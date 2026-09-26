"""Frozen interim development comparison; no modification of global recipe selection."""
import concurrent.futures as cf
import hashlib,json
from pathlib import Path
from . import judge
from .cohort import ROOT,save
from . import metrics
from ..monitor_pilot_0924.run import write_json
from ..openrouter_client import get_client
OUT=ROOT/'interim_variant_comparison'
IDS=[f'{p}{i:02}' for p in ['D','SD'] for i in range(1,5)]
def main():
 OUT.mkdir(exist_ok=True)
 path=OUT/'requests.json'
 if path.exists():rows=json.loads(path.read_text())
 else:
  records=[r for r in json.loads((ROOT/'cohort.json').read_text())['records'] if r['sample_id'] in IDS]
  assert len(records)==8 and all(r['tuning_role']=='development' for r in records)
  rows=[r for r in judge.build(records,list(judge.PROMPTS),'ready','cot_only') if not r['condition_id'].endswith('/endpoints')]
  save(path,rows)
  save(OUT/'protocol.json',dict(sample_ids=IDS,selection='Original eight development traces fixed before this comparison; retain all available conditions in this snapshot; evaluate variants only on common traces.',prompts=judge.PROMPTS,judge=judge.old.JUDGE,threshold='Per-task/per-prompt text-only threshold fitted on the other 20 development traces; apply unchanged to every graph condition. Also report threshold 0.5 and AUROC. These thresholds do not separately calibrate graph scores.',interpretation='Exploratory development check; no held-out data. Correlation is not a measure of correctness. No pruning rule is inferred from these small samples.'))
 judge.old.OUT=ROOT/'judge'; client=get_client().with_options(max_retries=0,timeout=180.)
 mapping={}
 for row in rows:
  req=dict(model=judge.old.JUDGE,messages=[dict(role='system',content=judge.PROMPTS[row['prompt_id']]),dict(role='user',content=judge.old.compact(row['packet']))],temperature=0.,top_p=1.,max_tokens=16384)
  mapping[row['sample_id']+'|'+row['condition_id']]=hashlib.sha256(judge.old.compact(req).encode()).hexdigest()
 save(OUT/'mapping.json',mapping)
 cached=sum((ROOT/'judge/max16384'/f'{v}.json').exists() for v in set(mapping.values()))
 print('REQUESTS',len(rows),'unique',len(set(mapping.values())),'cached',cached,flush=True)
 with cf.ThreadPoolExecutor(max_workers=8) as ex:
  futures=[ex.submit(judge.old.evaluate_one,client,row,judge.PROMPTS[row['prompt_id']],16384) for row in rows]
  results=[]
  for f in cf.as_completed(futures):
   results.append(f.result())
   if len(results)%20==0:print('COMPLETE',len(results),'OK',sum(r['status']=='ok' for r in results),flush=True)
 write_json(OUT/'completion.json',dict(requests=len(results),ok=sum(r['status']=='ok' for r in results)))
 assert all(r['status']=='ok' for r in results),'Incomplete judge responses; do not silently drop them'
 from .compare_ready_report import report
 report()
if __name__=='__main__':main()
