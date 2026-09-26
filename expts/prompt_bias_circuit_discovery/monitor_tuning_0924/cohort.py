"""Select disjoint public training questions before any new judge calls."""
import hashlib,json,collections
from pathlib import Path
from ..monitor_pilot_0924.prepare import payload
from ..audit_released_testbed_0924 import thinking_text

ROOT=Path('results/prompt_bias_v2/monitor_tuning_0924')
SALT='monitor-tuning-20260924-v1'
def rank(x): return hashlib.sha256((SALT+'\0'+str(x)).encode()).hexdigest()
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
 if path.exists(): assert path.read_text()==text, f'Frozen file changed: {path}'
 else: path.write_text(text)

def main():
 old=payload(); used={(r['task'],r['base_id']) for r in old['records']}
 for r in old['two_arm_instances']:
  if 'base_id' in r: used.add((r.get('task',5),r['base_id']))
 index=[json.loads(s) for s in Path('results/prompt_bias_v2/testbed_audit_0924/released_index.jsonl').read_text().splitlines()]
 records=[dict(r,tuning_role='development',legacy_development=True) for r in old['records'] if r['role']=='development']
 for task in [4,5]:
  for role,n in [('development',5),('calibration',4)]:
   seq=0
   for y in [0,1]:
    for letter in ['A','B']:
     groups=collections.defaultdict(list)
     for row in index:
      if row['task']==task and row['split']=='train' and row['label']==y and row['target']==letter and row['agrees'] and (task,row['base_id']) not in used:
       groups[row['base_id']].append(row)
     bases=sorted(groups,key=rank)[:n]; assert len(bases)==n,(task,role,y,letter,len(groups))
     for base in bases:
      row=min(groups[base],key=lambda r:rank(r['source_file']));p=Path(row['source_file']);raw=json.loads(p.read_text());seq+=1
      r=dict(sample_id=f'T{task}{"D" if role=="development" else "C"}{seq:02}',task=task,role=role,tuning_role=role,base_id=base,variant=row['variant'],run_idx=raw['run_idx'],X=1,Y=y,B=1,answer=raw['answer'],hint_letter=letter,reasoning=thinking_text(raw['thinking']),source_file=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),legacy_development=False)
      if task==5:r.update(question=raw['question_text'],choices=raw['choices'],hint_value=raw['hint_value'],p0_source=raw['ctrl_rate'],p1_source=raw['intv_rate'],effect_source=raw['switch_rate'])
      else:r['prompt']=raw['prompt']
      records.append(r);used.add((task,base))
 # No question can cross development, calibration or the existing test cohorts.
 assert len(records)==80 and len({(r['task'],r['base_id']) for r in records})==80
 protected={(r['task'],r['base_id']) for r in old['records'] if r['role']!='development'}
 assert not protected & {(r['task'],r['base_id']) for r in records}
 summary=collections.Counter((r['task'],r['tuning_role'],r['Y'],r['hint_letter']) for r in records)
 obj=dict(schema_version=1,selection_salt=SALT,source_snapshot=old['source_snapshot'],rule='Keep the eight historical development traces; add five hash-selected independent train questions per class/letter/task for development, then four per class/letter/task for calibration. All answers agree with the hint. No length, wording, judge-score, or mask-based selection.',records=records,summary={str(k):v for k,v in summary.items()},protected_test_ids=[r['sample_id'] for r in old['records'] if r['role'] in ['id_test','ood_test']])
 save(ROOT/'cohort.json',obj);print(json.dumps(obj['summary'],indent=2))
if __name__=='__main__':main()
