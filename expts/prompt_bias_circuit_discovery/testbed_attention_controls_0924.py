"""Bounded whole-graph and complement-clamped readout controls; no mask fitting."""
import json, os, time
from pathlib import Path
import torch
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.probe import build_answer_probe
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.eval_monitorability_readout_controls import metrics, direct_metrics

DATA=Path('results/prompt_bias_v2/reduce_masks/reduce_masks_qwen3_8b/dataset.json')
OUT=Path('results/prompt_bias_v2/testbed_audit_0924/attention_controls.json')
IDS=[0,1,4,5,12,13,14,15]

def main():
    start=time.monotonic(); records=json.loads(DATA.read_text())
    model,tok=load_model_eager('Qwen/Qwen3-8B',device='cuda');model.eval();model.config.use_cache=False
    dev=next(model.parameters()).device;handles=install_clean_sdpa_forward(model)
    out=dict(job=os.environ.get('SLURM_JOB_ID'),ids=IDS,rows=[],scope='Every layer/head; token diagonal and first3 context-free chat keys remain; complete stored reasoning. No training.')
    for i in IDS:
        rec=records[i]
        prefix,sents,_,_,_,n_p=_build_prefix(tokenizer=tok,prompt=None,data_path=str(DATA),prompt_index=i,base_answer_type='stored',analysis_timestep=rec['analysis_timestep'],analysis_sentence_step=None,min_sentence_length=10,sentence_chunk=1,sentences_after_prefix=0)
        prefix=prefix.to(dev);L=prefix.shape[-1];P=len(rec['prompt_token_ids'])
        probe=build_answer_probe(tok,suffix=PROBE_SUFFIX,answer_letters=rec['all_letters'])
        full=torch.cat([prefix,probe.make_continuation(dev)],-1);N=full.shape[-1]
        t=torch.arange(N,device=dev);q=t[:,None];k=t[None,:]
        causal=(k<q)&(k>=3)
        p2r=(q>=P)&(q<L)&(k<P)&causal
        r2q=(q>=L)&(k>=P)&(k<L)&causal
        p2q=(q>=L)&(k<P)&causal
        # Keep suffix-internal history as part of a declared fixed readout.
        q2q=(q>=L)&(k>=L)&causal
        prefix_all=causal&(q<L)&(k<L)
        masks={
          'empty_all_prefix':prefix_all,
          'empty_all_including_probe':causal,
          'p2r_all_kept_complement_zero_readout_R':causal&~(p2r|r2q|q2q),
          'p2r_empty_complement_zero_readout_R':causal&~(r2q|q2q),
          'p2r_all_kept_complement_zero_readout_P_R':causal&~(p2r|r2q|p2q|q2q),
          'p2r_empty_complement_zero_readout_P_R':causal&~(r2q|p2q|q2q),
        }
        row=dict(example_id=i,answer=rec['mask_target_letter'],prefix_tokens=L,prompt_tokens=P,conditions={'clean':metrics(model,full,probe,L)})
        for name,block in masks.items():
            assert not block.diagonal().any()
            row['conditions'][name]=direct_metrics(model,full,probe,L,block)
        out['rows'].append(row);OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2))
        print(i,{k:round(v['conditional'][rec['all_letters'].index(rec['mask_target_letter'])],4) for k,v in row['conditions'].items()},flush=True)
    out['elapsed_seconds']=time.monotonic()-start;OUT.write_text(json.dumps(out,indent=2));remove_handles(handles)
if __name__=='__main__':main()
