"""Evaluate fixed-size masks as reconstruction and cue-replay diagnostics.

The White-name replay changes the prompt only and holds Black-name reasoning
fixed. Its contrast is not an on-policy effect and not a trace-only feature.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.masks import NodeMask, build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter, build_region_filter
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.probe import build_answer_probe
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.eval_monitorability_readout_controls import metrics, production_metrics

ROOT=Path('results/prompt_bias_v2/rethink0924/mask_feasibility')
DATA=Path('results/prompt_bias_v2/reduce_masks/reduce_masks_qwen3_8b/dataset.json')
OLD=DATA.parent/'masks'
SUFFIXES={'original':(PROBE_SUFFIX,['A','B']), 'answer_colon':('</think>\n\nFinal answer:',[' A',' B'])}


def binary_from_scores(scores,pool,n_keep):
    assert np.isfinite(scores[pool]).all(), 'Non-finite learned/TA scores in eligible pool'
    valid=np.flatnonzero(pool.flatten())
    # Stable deterministic tie handling; record how many threshold ties exist.
    rank=np.argsort(-scores.flatten()[valid],kind='stable')
    selected=valid[rank[:n_keep]]
    binary=np.ones(pool.shape,dtype=np.float32)
    binary[pool]=0
    binary.flat[selected]=1
    boundary=float(scores.flat[selected[-1]]) if len(selected) else None
    ties=int(np.sum(scores[pool]==boundary)) if boundary is not None else 0
    return binary,dict(threshold=boundary,n_threshold_ties=ties)


def mask_features(binary,pool,texts,n_prompt,name):
    kept=(binary>.5)&pool
    positions=np.arange(len(texts))
    names={name,*name.split()}
    name_prompt=np.array([j<n_prompt and text.strip() in names for j,text in enumerate(texts)])
    name_mention=np.array([bool(re.search(r'\b(?:'+ '|'.join(re.escape(x) for x in names)+r')\b',text)) for text in texts])
    n_reason=len(texts)-n_prompt
    late_key=positions>=n_prompt+int(np.floor(.75*n_reason))
    late_query=positions>=n_prompt+int(np.floor(.75*n_reason))
    features=dict(n_eligible=int(pool.sum()),n_kept=int(kept.sum()),actual_removal=float(1-kept.sum()/pool.sum()))
    for kind,selector in [('name_prompt_key',name_prompt[None,:]),('name_mention_key',name_mention[None,:]),('late_reasoning_key',late_key[None,:]),('late_reasoning_query',late_query[:,None])]:
        target=pool&selector
        count=int(target.sum())
        kept_count=int((kept&target).sum())
        features[kind]=dict(n_eligible=count,n_kept=kept_count,
            keep_rate=(kept_count/count if count else None),
            enrichment=((kept_count/count)/(kept.sum()/pool.sum()) if count and kept.sum() else None),
            fraction_of_kept=(kept_count/int(kept.sum()) if kept.sum() else None))
    return features


def build_context(tokenizer,path,i,rec,device):
    prefix,sentences,_,_,_,n_p=_build_prefix(tokenizer=tokenizer,prompt=None,data_path=str(path),prompt_index=i,base_answer_type='stored',analysis_timestep=rec['analysis_timestep'],analysis_sentence_step=None,min_sentence_length=10,sentence_chunk=1,sentences_after_prefix=0)
    return prefix.to(device),sentences,n_p


def run_eval(i):
    start=time.monotonic()
    records=json.loads(DATA.read_text()); rec=records[i]
    model,tok=load_model_eager('Qwen/Qwen3-8B',device='cuda')
    model.eval();model.config.use_cache=False
    dev=next(model.parameters()).device
    contexts={}
    for arm,path in [('black',DATA),('white_replay',ROOT/'replay_white_dataset.json')]:
        prefix,sentences,n_p=build_context(tok,path,i,rec,dev)
        contexts[arm]=(prefix,sentences,n_p)
    prefix,sentences,n_p=contexts['black'];S=len(sentences)
    assert len(contexts['white_replay'][1])==S
    assert contexts['white_replay'][2]==n_p
    texts=[tok.decode(prefix[0,s.start:s.end+1].tolist()) for s in sentences]
    for sb,sw in zip(sentences[n_p:],contexts['white_replay'][1][n_p:]):
        assert torch.equal(prefix[0,sb.start:sb.end+1],contexts['white_replay'][0][0,sw.start:sw.end+1])
    handles=install_clean_sdpa_forward(model)
    out=dict(example_id=i,uid=rec['uid'],is_positive=rec['is_positive'],name=rec['name'],trace_answer=rec['mask_target_letter'],
             source_rates=dict(p_black=rec['p_black'],p_white=rec['p_white'],delta=rec['delta_white']),
             sentence_texts=texts,n_prompt=n_p,n_sentences=S,prefix_tokens=int(prefix.shape[-1]),gender=rec['gender'],
             reasoning_name_mentions=sum(len(re.findall(r'\b(?:'+ '|'.join(re.escape(x) for x in {rec['name'],*rec['name'].split()})+r')\b',t)) for t in texts[n_p:]),pools={},
             replay_note='White prompt plus original Black-name CoT. No on-policy causal claim.')
    for pool_short,region in [('p2t','prompt_to_trace'),('t2t','trace_to_trace')]:
        combined=build_combined_filter(build_gap_filter(S,1,device=dev),build_mode_filter(S,S,'prefix',device=dev),build_causal_filter(S,device=dev),None)
        combined |= build_region_filter(region,n_p,S,device=dev,frozen_key_sentences=rec['frozen_key_sentences'] if pool_short=='p2t' else None)
        pool=(~combined).cpu().numpy()
        n_keep=int(round(.2*int(pool.sum())))
        masks={};paths={};details={}
        ones=np.ones((S,S),dtype=np.float32)
        empty=ones.copy();empty[pool]=0
        masks.update(clean=ones,empty=empty)
        mask_paths={'increase':ROOT/'masks'/f'ex{i:03d}_{pool_short}_increase_s80.json',
                    'decrease':OLD/f'ex{i:03d}_{pool_short}_think_s80.json',
                    'thought_anchors':ROOT/'masks_ta'/f'ex{i:03d}_thought_anchors.json'}
        for method,path in mask_paths.items():
            if not path.exists():
                raise FileNotFoundError(f'Required prespecified mask missing: {path}')
            nm=NodeMask.from_json(str(path)); scores=np.asarray(nm.scores,dtype=float)
            assert scores.shape==(S,S),(method,scores.shape,S)
            # These maps use exactly the same sentence spans and original prefix.
            assert [(x['start'],x['end']) for x in nm.sentences]==[(s.start,s.end) for s in sentences]
            binary,threshold_info=binary_from_scores(scores,pool,n_keep)
            masks[method]=binary;paths[method]=str(path)
            detail=dict(**threshold_info,score_readout=nm.metadata.get('score_readout'))
            if method in ['increase','decrease']:
                training_path=path.with_suffix('')/'training_metrics.jsonl'
                metrics_rows=[json.loads(line) for line in training_path.read_text().splitlines()]
                detail['last_training_metrics']=metrics_rows[-1]
                detail['first_training_metrics']=metrics_rows[0]
                detail['recorded_training_steps']=len(metrics_rows)
                detail['last_nonzero_task_grad_step']=max((x['step'] for x in metrics_rows if x.get('task_grad_norm',0)>0),default=None)
                detail['score_std']=float(scores[pool].std())
            details[method]=detail
        indices=np.flatnonzero(pool.flatten())
        for seed in range(10):
            rng=np.random.default_rng([924,i,seed,0 if pool_short=='p2t' else 1])
            binary=empty.copy();binary.flat[rng.choice(indices,n_keep,replace=False)]=1
            masks[f'random_{seed:02d}']=binary
        features={method:mask_features(binary,pool,texts,n_p,rec['name']) for method,binary in masks.items()}
        overlaps={}
        for other in ['decrease','thought_anchors']:
            if other in masks:
                a=(masks['increase']>.5)&pool;b=(masks[other]>.5)&pool
                overlaps[other]=dict(jaccard=float((a&b).sum()/(a|b).sum()),shared_fraction=float((a&b).sum()/a.sum()))
        evaluated={}
        for suffix_name,(suffix,letters) in SUFFIXES.items():
            probe=build_answer_probe(tok,suffix=suffix,answer_letters=letters)
            by_arm={}
            for arm,(prefix_arm,sentences_arm,_) in contexts.items():
                prefix_len=prefix_arm.shape[-1]
                full=torch.cat([prefix_arm,probe.make_continuation(dev)],dim=-1)
                mapping=torch.full((full.shape[-1],),-1,dtype=torch.long,device=dev)
                for j,sent in enumerate(sentences_arm):mapping[sent.start:sent.end+1]=j
                values={}
                for method,binary in masks.items():
                    value=production_metrics(model,full,probe,prefix_len,torch.tensor(binary,device=dev),mapping,combined)
                    value['log_odds_A_over_B']=float(np.log(max(value['conditional'][0],1e-30))-np.log(max(value['conditional'][1],1e-30)))
                    value['top_token_text']=tok.decode([value['top_token_id']])
                    values[method]=value
                by_arm[arm]=values
                del full,mapping
            contrasts={method:{'conditional_A':by_arm['black'][method]['conditional'][0]-by_arm['white_replay'][method]['conditional'][0],
                              'log_odds':by_arm['black'][method]['log_odds_A_over_B']-by_arm['white_replay'][method]['log_odds_A_over_B']} for method in masks}
            evaluated[suffix_name]=dict(suffix=suffix,answer_letters=letters,arms=by_arm,replay_contrasts=contrasts)
        out['pools'][pool_short]=dict(n_eligible=int(pool.sum()),n_keep=n_keep,actual_removal=1-n_keep/int(pool.sum()),
                                    mask_files=paths,training_and_threshold=details,features=features,overlap_with_increase=overlaps,evaluation=evaluated)
        print(f'ex{i:03d} {pool_short}: keep {n_keep}/{pool.sum()} | '+ ' '.join(f"{m}={evaluated['original']['arms']['black'][m]['conditional'][0]:.4f}" for m in ['clean','empty','increase','decrease','thought_anchors'] if m in masks),flush=True)
        torch.cuda.empty_cache()
    remove_handles(handles)
    out['elapsed_seconds']=time.monotonic()-start
    path=ROOT/'eval'/f'ex{i:03d}.json';path.write_text(json.dumps(out,indent=2))
    print(f'Wrote {path} in {out["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--example_id',type=int,required=True)
    run_eval(ap.parse_args().example_id)
