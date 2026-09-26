"""Descriptive positive pipeline check; no search over examples or objectives."""
from pathlib import Path
import json
import numpy as np
import torch
from utils.circuit_eval import install_clean_sdpa_forward,remove_handles
from utils.masks import NodeMask,build_gap_filter,build_mode_filter,build_causal_filter,build_combined_filter,build_region_filter
from expts.direct_answer_circuit_discovery.learn import load_model_eager,_build_prefix
from expts.direct_answer_circuit_discovery.probe import build_answer_probe
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.eval_monitorability_readout_controls import production_metrics
from expts.prompt_bias_circuit_discovery.rethink0924.eval_mask_feasibility import binary_from_scores

ROOT=Path('results/prompt_bias_v2/rethink0924/hint_positive_control')

def main():
    rec=json.loads((ROOT/'dataset.json').read_text())[0]
    model,tok=load_model_eager('Qwen/Qwen3-8B',device='cuda');model.eval();model.config.use_cache=False
    dev=next(model.parameters()).device
    prefix,sentences,_,_,_,n_p=_build_prefix(tokenizer=tok,prompt=None,data_path=str(ROOT/'dataset.json'),prompt_index=0,base_answer_type='stored',analysis_timestep=rec['analysis_timestep'],analysis_sentence_step=None,min_sentence_length=10,sentence_chunk=1,sentences_after_prefix=0)
    prefix=prefix.to(dev);S=len(sentences)
    combined=build_combined_filter(build_gap_filter(S,1,device=dev),build_mode_filter(S,S,'prefix',device=dev),build_causal_filter(S,device=dev),None)
    combined|=build_region_filter('prompt_to_trace',n_p,S,device=dev,frozen_key_sentences=rec['frozen_key_sentences'])
    pool=(~combined).cpu().numpy();n_keep=int(round(.2*pool.sum()))
    cue=np.zeros(S,dtype=bool);cue[rec['cue_key_sentences']]=True;cue_pool=pool&cue[None,:]
    ones=np.ones((S,S),dtype=np.float32);empty=ones.copy();empty[pool]=0
    masks={'clean':ones,'empty':empty};details={}
    for method,path in [('increase',ROOT/'masks/hint001_p2t_increase_s80.json'),('thought_anchors',ROOT/'masks_ta/hint001_thought_anchors.json')]:
        nm=NodeMask.from_json(str(path));scores=np.asarray(nm.scores,dtype=float)
        assert scores.shape==(S,S)
        assert [(x['start'],x['end']) for x in nm.sentences]==[(s.start,s.end) for s in sentences]
        masks[method],details[method]=binary_from_scores(scores,pool,n_keep)
    for seed in range(10):
        rng=np.random.default_rng([924,1001,seed]);m=empty.copy();m.flat[rng.choice(np.flatnonzero(pool.flatten()),n_keep,replace=False)]=1;masks[f'random_{seed:02d}']=m
    masks['increase_without_cue_reads']=masks['increase'].copy();masks['increase_without_cue_reads'][cue_pool]=0
    masks['empty_plus_cue_reads']=empty.copy();masks['empty_plus_cue_reads'][cue_pool]=1
    features={}
    for method,b in masks.items():
        kept=(b>.5)&pool
        features[method]=dict(n_kept=int(kept.sum()),n_eligible=int(pool.sum()),actual_removal=float(1-kept.sum()/pool.sum()),
                             n_cue_eligible=int(cue_pool.sum()),n_cue_kept=int((kept&cue_pool).sum()),
                             cue_enrichment=float((kept&cue_pool).sum()/cue_pool.sum()/(kept.sum()/pool.sum())) if kept.sum() else None)
    handles=install_clean_sdpa_forward(model);evaluated={}
    for suffix_name,suffix,letters in [('original',PROBE_SUFFIX,rec['all_letters']),('answer_colon','</think>\n\nFinal answer:',[' '+x for x in rec['all_letters']])]:
        probe=build_answer_probe(tok,suffix=suffix,answer_letters=letters);full=torch.cat([prefix,probe.make_continuation(dev)],-1)
        mapping=torch.full((full.shape[-1],),-1,dtype=torch.long,device=dev)
        for j,s in enumerate(sentences):mapping[s.start:s.end+1]=j
        values={}
        for method,b in masks.items():
            v=production_metrics(model,full,probe,prefix.shape[-1],torch.tensor(b,device=dev),mapping,combined)
            v['p_target']=v['conditional'][rec['all_letters'].index(rec['mask_target_letter'])];v['top_token_text']=tok.decode([v['top_token_id']]);values[method]=v
        evaluated[suffix_name]=dict(suffix=suffix,letters=letters,conditions=values)
    remove_handles(handles)
    training=[json.loads(s) for s in (ROOT/'masks/hint001_p2t_increase_s80/training_metrics.jsonl').read_text().splitlines()]
    out=dict(source_index=1,n_prompt=n_p,n_sentences=S,cue_sentences=rec['cue_key_sentences'],
             sentence_texts=[tok.decode(prefix[0,s.start:s.end+1].tolist()) for s in sentences],
             features=features,thresholds=details,last_training_metrics=training[-1],evaluation=evaluated)
    (ROOT/'eval.json').write_text(json.dumps(out,indent=2));print('Wrote',ROOT/'eval.json',flush=True)

if __name__=='__main__':main()
