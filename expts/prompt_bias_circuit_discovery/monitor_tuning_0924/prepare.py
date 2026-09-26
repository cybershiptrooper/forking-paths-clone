"""Preserve original semantic chunking, freeze spans, then queue development fits."""
import os
os.environ.setdefault('MONITOR_SEGMENT_MAX_TOKENS','16000')
import concurrent.futures as cf,json,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from ..monitor_pilot_0924 import prepare as old
from ..monitor_pilot_0924.run import write_json
from ..monitor_pilot_0924.expanded_worker import EXPANDED
from ..openrouter_client import get_client,load_cache
from .cohort import ROOT,save

def main():
 records=json.loads((ROOT/'cohort.json').read_text())['records']
 tok=AutoTokenizer.from_pretrained(old.MODEL,revision=old.REVISION,local_files_only=True)
 client=get_client();cache=load_cache(old.ROOT/'segmenter_cache.json')
 ledger_path=ROOT/'preparation.json'
 ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(submissions=[],failures=[],prepared=[])
 assert all(s['status']=='submitted' for s in ledger['submissions']),'Resolve ambiguous submission before restart'
 submitted={s['sample_id'] for s in ledger['submissions']}
 def prepare(r):
  path=ROOT/'inputs'/f"{r['sample_id']}.json"
  if path.exists():return json.loads(path.read_text())
  source=EXPANDED/'inputs'/f"{r['sample_id']}.json"
  if r['legacy_development']:item=json.loads(source.read_text())
  else:
   item=old.prepare_record(r,tok,client,cache);item['segmentation_output_limit']=16000;item['frozen_sha256']=old.digest(item)
  save(path,item);return item
 with cf.ThreadPoolExecutor(max_workers=8) as ex:
  futures={ex.submit(prepare,r):r for r in records}
  for f in cf.as_completed(futures):
   r=futures[f];sid=r['sample_id']
   try:item=f.result()
   except Exception as exc:
    ledger['failures'].append(dict(sample_id=sid,error=repr(exc)));write_json(ledger_path,ledger);print('FAILED',sid,repr(exc),flush=True);continue
   if sid not in ledger['prepared']:ledger['prepared'].append(sid)
   if r['tuning_role']=='development' and sid not in submitted:
    tasks=[]
    for retention in [.2,.5]:
     if r['legacy_development'] and retention==.2:continue
     for condition in ['rr_on','rr_off']:
      tasks.append(dict(sample_id=sid,kind='fits',condition=condition,retention=retention,recipe=f'{condition}_keep{int(100*retention)}'))
    if not r['legacy_development']:
     for condition in ['joint','rr_off']:tasks.append(dict(sample_id=sid,kind='ta',condition=condition,recipe=condition))
    manifest=ROOT/'manifests'/f'{sid}.json';save(manifest,tasks)
    partition=json.loads((EXPANDED/'partition_routing.json').read_text())['discovered_partition']
    entry=dict(sample_id=sid,manifest=str(manifest),tasks=len(tasks),status='submitting');ledger['submissions'].append(entry);write_json(ledger_path,ledger)
    job=subprocess.check_output(['sbatch','--parsable','--priority=10002',f'--partition={partition}',f'--array=0-{len(tasks)-1}','claude_scripts/monitor_tuning_0924.sbatch',str(manifest)],text=True).strip().split(';')[0]
    entry.update(job_id=job,status='submitted');submitted.add(sid)
    with (ROOT/f'monitor_{job}.log').open('a') as log:subprocess.Popen(['monitor_jobs',job],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('SUBMITTED',sid,job,len(tasks),flush=True)
   write_json(ledger_path,ledger);print('READY',sid,len(item['input_ids']),flush=True)
 ledger['unresolved_samples']=[r['sample_id'] for r in records if not (ROOT/'inputs'/f"{r['sample_id']}.json").exists()]
 ledger['status']='complete' if not ledger['unresolved_samples'] else 'preparation_failures';write_json(ledger_path,ledger)
if __name__=='__main__':main()
