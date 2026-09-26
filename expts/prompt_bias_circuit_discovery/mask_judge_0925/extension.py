"""Extension set for the larger Thought Anchors test (2026-09-26).

Unused released train/validation traces that agree with the hint, one per base question, hash-ordered
with a new salt. Professor task: 30 per label x hinted letter (120). Scruples: every unused negative
(7 hinted A, 22 hinted B) and the same number of positives per hinted letter (58). Selection never looks
at masks, judge scores, wording or length. Inputs are frozen with the stopped study's prepare_record
(same tokenizer, reasoning splitter and prompt segmenter) into mask_judge_0925/ext_inputs.
"""
import concurrent.futures as cf,hashlib,json,collections,os
os.environ.setdefault('MONITOR_SEGMENT_MAX_TOKENS','16000')
from pathlib import Path
from transformers import AutoTokenizer
from ..monitor_pilot_0924 import prepare as old
from ..monitor_pilot_0924.prepare import payload
from ..audit_released_testbed_0924 import thinking_text
from ..openrouter_client import get_client,load_cache
from . import data as D

SALT='mask-judge-extension-20260926'
OUT=D.NEW/'ext_inputs';COHORT=D.NEW/'extension_cohort.json'
def rank(x):return hashlib.sha256((SALT+'\0'+str(x)).encode()).hexdigest()

def select():
 idx=[json.loads(s) for s in Path('results/prompt_bias_v2/testbed_audit_0924/released_index.jsonl').read_text().splitlines()]
 o=payload();used={(int(r['task']),r['base_id']) for r in o['records']}
 for r in o['two_arm_instances']:
  if 'base_id' in r:used.add((int(r.get('task',5)),r['base_id']))
 used|={(r['task'],r['base_id']) for r in json.loads((D.TUN/'cohort.json').read_text())['records']}
 groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in idx:
  if r['split'] in ('train','val') and r['agrees'] and (r['task'],r['base_id']) not in used:groups[(r['task'],r['label'],r['target'])][r['base_id']].append(r)
 quota={(5,y,l):30 for y in (0,1) for l in 'AB'}
 for l in 'AB':n=len(groups[(4,0,l)]);quota[(4,0,l)]=n;quota[(4,1,l)]=n
 records=[];taken=set()
 for (task,y,letter),n in sorted(quota.items()):
  bases=[b for b in sorted(groups[(task,y,letter)],key=rank) if (task,b) not in taken][:n];assert len(bases)==n,(task,y,letter,len(bases))
  for i,b in enumerate(bases,1):
   row=min(groups[(task,y,letter)][b],key=lambda r:rank(r['source_file']));p=Path(row['source_file']);raw=json.loads(p.read_text())
   r=dict(sample_id=f'X{task}{"P" if y else "N"}{letter}{i:02}',task=task,role='extension',tuning_role='extension',base_id=b,variant=row['variant'],run_idx=raw['run_idx'],X=1,Y=y,B=1,answer=raw['answer'],hint_letter=letter,reasoning=thinking_text(raw['thinking']),source_file=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),split=row['split'])
   if task==5:r.update(question=raw['question_text'],choices=raw['choices'],hint_value=raw['hint_value'],p0_source=raw['ctrl_rate'],p1_source=raw['intv_rate'],effect_source=raw['switch_rate'])
   else:r['prompt']=raw['prompt']
   records.append(r);taken.add((task,b))
 assert len({(r['task'],r['base_id']) for r in records})==len(records)
 COHORT.write_text(json.dumps(dict(salt=SALT,quota={f'{k[0]}_{k[1]}_{k[2]}':v for k,v in quota.items()},records=records),indent=1,ensure_ascii=False))
 return records

def main():
 records=json.loads(COHORT.read_text())['records'] if COHORT.exists() else select()
 print('records',len(records),collections.Counter((r['task'],r['Y']) for r in records),flush=True)
 tok=AutoTokenizer.from_pretrained(old.MODEL,revision=old.REVISION,local_files_only=True)
 old.ROOT=D.NEW/'ext_segmenter';old.ROOT.mkdir(parents=True,exist_ok=True)  # keep the segmenter cache and attempt log out of the stopped pilot's folder
 client=get_client();cache=load_cache(old.ROOT/'segmenter_cache.json');OUT.mkdir(parents=True,exist_ok=True)
 def prep(r):
  path=OUT/f"{r['sample_id']}.json"
  if path.exists():return 'exists'
  item=old.prepare_record(r,tok,client,cache);item['segmentation_output_limit']=16000;item['frozen_sha256']=old.digest(item)
  path.write_text(json.dumps(item,ensure_ascii=False));return 'ok'
 with cf.ThreadPoolExecutor(8) as ex:
  futs={ex.submit(prep,r):r['sample_id'] for r in records}
  for f in cf.as_completed(futs):
   try:print(futs[f],f.result(),flush=True)
   except Exception as exc:print('FAILED',futs[f],repr(exc),flush=True)
 missing=[r['sample_id'] for r in records if not (OUT/f"{r['sample_id']}.json").exists()];print('missing',missing)
if __name__=='__main__':main()
