"""Prespecified, answer-matched feasibility pilot: eight A traces, two pools."""
from pathlib import Path
import json
import re
import yaml
from transformers import AutoTokenizer
from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt
from expts.prompt_bias_circuit_discovery.gen_sweep_configs import CANONICAL
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX

DATA='results/prompt_bias_v2/reduce_masks/reduce_masks_qwen3_8b/dataset.json'
ROOT=Path('expts/prompt_bias_circuit_discovery/rethink0924/configs')
OUT=Path('results/prompt_bias_v2/rethink0924/mask_feasibility')


def main():
    records=json.loads(Path(DATA).read_text())
    ROOT.mkdir(parents=True,exist_ok=True)
    controls={r['uid']:r for r in json.loads(Path('results/prompt_bias_v2/full/blindspot_admission_prompts.json').read_text())}
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    replay_records=[]
    for rec in records:
        white=controls[rec['base_uid']]
        cf=dict(rec)
        for key in ['prompt','prompt_token_ids','question','question_with_choices','name']:
            cf[key]=white[key]
        name=white['name']; first,last=name.split()[0],name.split()[-1]
        forced=[(m.start(),m.end()) for m in re.finditer(rf'\b{re.escape(name)}\b|\b{re.escape(first)}\b|\b{re.escape(last)}\b',white['question'])]
        chunks,_=chunk_prompt(tok,white,'sentence',forced_spans=forced)
        chunks[0]['start']=3
        assert [c['kind'] for c in chunks]==rec['prompt_chunk_kinds'],rec['uid']
        cf['prompt_chunk_spans']=[[c['start'],c['end']] for c in chunks]
        cf['replay_note']='White control prompt, original Black-name reasoning tokens; not on-policy.'
        replay_records.append(cf)
    (OUT/'replay_white_dataset.json').write_text(json.dumps(replay_records))
    for i in range(0,16,2):
        rec=records[i]
        assert rec['mask_target_letter']=='A'
        common=dict(model_name='Qwen/Qwen3-8B',data_path=DATA,prompt_index=i,
                    analysis_timestep=rec['analysis_timestep'],sentences_after_prefix=0)
        for pool,region in [('p2t','prompt_to_trace'),('t2t','trace_to_trace')]:
            cfg=dict(CANONICAL,**common,answer_letters=rec['all_letters'],probe_suffix=PROBE_SUFFIX,
                     objective='answer_probe_reward_gap',target_letter=rec['mask_target_letter'],target_sparsity=.8,
                     learnable_region=region,output_dir=str(OUT/'masks'),file_name=f'ex{i:03d}_{pool}_increase_s80')
            if pool=='p2t': cfg['frozen_key_sentences']=rec['frozen_key_sentences']
            (ROOT/f'ex{i:03d}_{pool}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        ta=dict(**common,sentence_gap=1,sentence_chunk=1,mask_mode='prefix',device='cuda',seed=42,
                backend='sdpa',output_dir=str(OUT/'masks_ta'),file_name=f'ex{i:03d}')
        (ROOT/f'ex{i:03d}_ta.yaml').write_text(yaml.safe_dump(ta,sort_keys=False))
    plan=dict(dataset=DATA,example_ids=list(range(0,16,2)),n_inputs=8,n_traces=8,
              target='trace own answer A',checkpoint='whole reasoning',target_sparsity=.8,
              optimizer='unchanged canonical hybrid; fixed target_size_l2 penalty, no Lagrange multiplier',
              num_training_steps=1000,regions=['prompt_to_trace','trace_to_trace'],
              selection='All eight inputs of the existing reduce-mask pilot; first stored A trace each, fixed before new fitting.',
              evaluation=['clean','empty pool','own-answer-increasing SNP','existing opposite-answer SNP','native prefix-KL Thought Anchors','10 same-size random masks'],
              predeclared_features=['fraction of eligible name-key edges retained normalized by pool keep rate','fraction retained edges with late reasoning keys','retention of same-CoT Black-vs-White prompt-swap readout contrast'],
              limitations=['N=8 is a descriptive feasibility screen, not held-out classification','effect labels belong to inputs, not causally labeled traces','whole-trace teacher forcing permits answer-copying','prompt swaps replay one stored CoT and do not estimate on-policy behavior'])
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    print('Wrote 16 canonical SNP configurations and 8 native Thought Anchors configurations.')

if __name__=='__main__': main()
