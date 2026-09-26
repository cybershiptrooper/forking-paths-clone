"""Fixed historical professor-hint ID1; ID4 excluded for length before fitting."""
from pathlib import Path
import json
import yaml
from transformers import AutoTokenizer
from expts.prompt_bias_circuit_discovery.gen_sweep_configs import CANONICAL
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt

ROOT=Path('results/prompt_bias_v2/rethink0924/hint_positive_control')
CONFIG=Path('expts/prompt_bias_circuit_discovery/rethink0924/configs')

def main():
    source='data/collection/qwen3_8b/hinted_merged.json'
    rec=json.loads(Path(source).read_text())[1]
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    chunks,_=chunk_prompt(tok,rec,'sentence')
    chunks[0]['start']=3
    cue=[]
    for j,c in enumerate(chunks):
        if 'Stanford professor' in c['text']:
            c['kind']='cue';cue.append(j)
    assert len(cue)==1
    rec.update(analysis_timestep=rec['output_token_ids'].index(151668),reasoning_skip_tokens=2,
               prompt_chunk_spans=[[c['start'],c['end']] for c in chunks],prompt_chunk_kinds=[c['kind'] for c in chunks],
               n_prompt_chunks=len(chunks),frozen_key_sentences=[j for j,c in enumerate(chunks) if c['kind'] in ['choices_header','choice']],
               cue_key_sentences=cue,mask_target_letter=rec['clean_answer'],source_index=1)
    for sub in ['masks','masks_ta']: (ROOT/sub).mkdir(parents=True,exist_ok=True)
    dataset=ROOT/'dataset.json';dataset.write_text(json.dumps([rec]))
    plan=dict(source=source,source_index=1,selection='First entry of pre-existing hint_selection.json, fixed before new training.',
              omitted=dict(source_index=4,reason='11945 reasoning tokens exceeds bounded full-trace training cost; exclusion before new masks'),
              control_counts=rec['control_answer_counts'],hinted_answers=rec['all_sampled_answers'],
              analysis_timestep=rec['analysis_timestep'],objective='increase observed answer, unchanged reward-gap',target_sparsity=.8,
              region='prompt_to_trace',checkpoint='whole reasoning',n_training_steps=1000,
              no_outcome_selection='No selection or optimization using replay differences or new masks.')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    common=dict(model_name='Qwen/Qwen3-8B',data_path=str(dataset),prompt_index=0,analysis_timestep=rec['analysis_timestep'],sentences_after_prefix=0)
    cfg=dict(CANONICAL,**common,answer_letters=rec['all_letters'],probe_suffix=PROBE_SUFFIX,
             objective='answer_probe_reward_gap',target_letter=rec['mask_target_letter'],target_sparsity=.8,
             frozen_key_sentences=rec['frozen_key_sentences'],output_dir=str(ROOT/'masks'),file_name='hint001_p2t_increase_s80')
    (CONFIG/'hint001_p2t.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    ta=dict(**common,sentence_gap=1,sentence_chunk=1,mask_mode='prefix',device='cuda',seed=42,backend='sdpa',output_dir=str(ROOT/'masks_ta'),file_name='hint001')
    (CONFIG/'hint001_ta.yaml').write_text(yaml.safe_dump(ta,sort_keys=False))
    print('Fixed professor ID1 ready:',len(chunks),'prompt chunks, cue key',cue,'think cut',rec['analysis_timestep'])

if __name__=='__main__':main()
