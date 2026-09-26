"""Fixed 2x2 prompt/temperature audit of all released training positives in our screen.
No selection on new results; no API calls; no mask tuning.
"""
import argparse,ast,json,os,time
from pathlib import Path
from collections import defaultdict

ROOT=Path('results/prompt_bias_v2/testbed_audit_0924/prompt_replication')
SOURCE=Path('data/external/cot-proxy-tasks')

def prepare():
    local=json.loads(Path('results/prompt_bias_v2/rethink_0924/qwen32_screen/prompts.json').read_text())
    local={(r['qid'],r['arm']):r for r in local if r['family']=='scruples'}
    first={}
    for path in sorted((SOURCE/'datasets/4/qwen-3-32b/train').glob('*.json')):
        r=json.loads(path.read_text());key=(r['anecdote_id'],r['variant'])
        if key in local and r['label']=='sycophantic':first.setdefault(key,(r,str(path)))
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse((SOURCE/'src/tasks/scruples/prompts.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.targets[0],ast.Name)}
    assert len(first)==3,first.keys()
    rows=[];cases=[]
    for (qid,variant),(r,path) in sorted(first.items()):
        meta=json.loads((SOURCE/f'datasets/4/prompts/train/{qid}_{variant}.json').read_text())
        control=constants['CONTROL_PROMPT'].format(post_title=meta['title'],post_text=meta['text'])
        cases.append(dict(qid=qid,variant=variant,published_switch=meta['switch_rate'],source=path))
        for fmt in ['published','local']:
            for temp in [.6,.7]:
                for arm in ['control',variant]:
                    own=local[qid,arm]
                    content=(r['prompt'] if arm==variant else control) if fmt=='published' else own['prompt'].split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
                    rows.append(dict(qid=qid,variant=variant,arm=arm,prompt_format=fmt,temperature=temp,content=content,target='A' if variant=='suggest_wrong' else 'B',question=own['question'],all_letters=['A','B'],all_answers=['Yes','No'],dataset_type='multiple choice'))
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_commit='4482324b5e4a6277fa3bd544785cbd9875e11694',selection='All released Task4 train positive question/variant pairs present in the prior24-question Scruples screen; no result-dependent exclusions.',cases=cases,n_per_condition=50,model='Qwen/Qwen3-32B',max_tokens=8000,top_p=1.,top_k=-1,seed_scheme='40000000+1000*condition index+sample index',conditions=rows,limitations='Local vLLM reproduction; OpenRouter provider revision/defaults not supplied by release. No claim of identical serving.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare_only',action='store_true');args=ap.parse_args();rows=prepare()
    if args.prepare_only:print('Frozen',len(rows),'conditions x50 draws');return
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from utils.answer_utils import parse_answer
    start=time.monotonic();tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-32B',local_files_only=True)
    prompts=[tok.apply_chat_template([dict(role='user',content=r['content'])],tokenize=True,add_generation_prompt=True,enable_thinking=True) for r in rows]
    llm=LLM(model='Qwen/Qwen3-32B',tensor_parallel_size=2,gpu_memory_utilization=.9,max_model_len=max(map(len,prompts))+8064,max_num_seqs=128,seed=9241)
    params=[SamplingParams(n=50,temperature=r['temperature'],max_tokens=8000,seed=40000000+1000*i,top_p=1.,top_k=-1) for i,r in enumerate(rows)]
    outs=llm.generate([dict(prompt_token_ids=p) for p in prompts],params);records=[];parse_inputs=[]
    for i,(r,p,o) in enumerate(zip(rows,prompts,outs)):
        record=dict(r,prompt_token_ids=p,rollouts=[dict(text=c.text,token_ids=list(c.token_ids),finish_reason=c.finish_reason,seed=40000000+1000*i+c.index) for c in o.outputs]);records.append(record)
        for c in record['rollouts']:parse_inputs.append(dict(question=r['question'],output_text=c['text'],all_letters=r['all_letters'],all_answers=r['all_answers'],dataset_type=r['dataset_type']))
    (ROOT/'generations_unparsed.json').write_text(json.dumps(records))
    parsed=iter(parse_answer(llm,parse_inputs))
    for r in records:
        for c in r['rollouts']:
            p=next(parsed);c.update(answer=p['clean_answer'],raw_answer=p['raw_answer'])
    (ROOT/'generations.json').write_text(json.dumps(records))
    (ROOT/'run.json').write_text(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start)))
    print('Saved all1200 samples',flush=True)
if __name__=='__main__':main()
