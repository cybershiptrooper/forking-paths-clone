"""Audit the pinned public Qwen3-32B datasets; never modifies source data."""
from collections import Counter,defaultdict
import hashlib,json,subprocess
from pathlib import Path

SOURCE=Path('data/external/cot-proxy-tasks')
OUT=Path('results/prompt_bias_v2/testbed_audit_0924')
TARGET={'suggest_wrong':'A','suggest_right':'B','suggest_sarcastic':'A','suggest_sincere':'B'}

def thinking_text(value):
    if isinstance(value,str):return value
    if isinstance(value,list):
        assert all(isinstance(x,dict) and isinstance(x.get('text'),str) for x in value)
        return '\n'.join(x['text'] for x in value)
    raise TypeError(type(value))

def main():
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    out=dict(source_url='https://github.com/Centrattic/cot-proxy-tasks',source_commit=commit,tasks={},notes=['Counts are released files, not regenerated outcomes.','Base questions can have different labels under different hint variants.','Task4 release does not contain complete control and intervention sampling banks; do not reconstruct causal rates from the label-filtered rollouts.','Task5 OOD thinking is a list of reasoning blocks; normalize its text without dropping blocks.'])
    index=[]
    for task in [4,5]:
        subsets={};base_sets={};question_hashes={}
        for split in ['train','val','test','ood_test']:
            files=sorted((SOURCE/f'datasets/{task}/qwen-3-32b/{split}').glob('*.json'))
            rows=[]
            for path in files:
                r=json.loads(path.read_text());base=r.get('anecdote_id',r.get('question_id'));variant=r.get('variant',r.get('direction'))
                label=int(r['label'] in ['sycophantic','hint_following']);target=r.get('hint_letter') or TARGET[variant]
                text=thinking_text(r['thinking']);answer=r['answer'];assert answer in ['A','B','C','D']
                row=dict(source_file=str(path),base_id=base,variant=variant,label=label,answer=answer,target=target,agrees=answer==target,thinking_chars=len(text),thinking_sha256=hashlib.sha256(text.encode()).hexdigest(),thinking_type=type(r['thinking']).__name__)
                if task==5:row.update(p1=r['intv_rate'],p0=r['ctrl_rate'],effect=r['switch_rate'])
                rows.append(row);index.append(dict(task=task,split=split,**row))
            groups=defaultdict(list)
            for r in rows:groups[r['base_id']].append(r)
            pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
            neg_agree=sum(r['agrees'] for r in neg);pos_agree=sum(r['agrees'] for r in pos)
            tp=pos_agree/len(pos);tn=1-neg_agree/len(neg)
            subsets[split]=dict(n_rollouts=len(rows),n_base_questions=len(groups),n_question_variants=len({(r['base_id'],r['variant']) for r in rows}),label_rollouts=dict(Counter(r['label'] for r in rows)),base_questions_by_class={str(k):len({r['base_id'] for r in rows if r['label']==k}) for k in [0,1]},n_base_questions_with_both_labels=sum(len({r['label'] for r in rr})==2 for rr in groups.values()),hint_agree_rollouts_by_class={'0':neg_agree,'1':pos_agree},same_answer_base_questions_by_class={str(k):len({r['base_id'] for r in rows if r['label']==k and r['agrees']}) for k in [0,1]},thinking_formats=dict(Counter(r['thinking_type'] for r in rows)),answer_agreement_shortcut=dict(tpr=tp,tnr=tn,gmean2=tp*tn,auroc=(tp+tn)/2),empty_reasoning=sum(not r['thinking_chars'] for r in rows))
            base_sets[split]=set(groups)
        overlap={f'{a}__{b}':len(base_sets[a]&base_sets[b]) for a in base_sets for b in base_sets if a<b}
        out['tasks'][str(task)]=dict(splits=subsets,base_id_overlap=overlap)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'released_summary.json').write_text(json.dumps(out,indent=2))
    (OUT/'released_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
