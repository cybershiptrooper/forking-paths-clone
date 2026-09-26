"""Freeze development winners, calibrate separately, then evaluate protected tests.

This module never runs unless every planned development comparison is complete.
"""
import argparse,concurrent.futures as cf,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
from .cohort import ROOT,save
from . import judge
from .metrics import threshold,gmean,auc,summarize
from ..monitor_pilot_0924 import judge as old
from ..monitor_pilot_0924.prepare import payload
from ..monitor_pilot_0924.expanded_worker import EXPANDED
from ..monitor_pilot_0924.run import write_json
from ..openrouter_client import get_client

OUT=ROOT/'confirmation'

def choose():
 assert (ROOT/'primary_development_complete.json').exists(),'Primary development grid still running'
 opt=json.loads((ROOT/'optimizer_selection.json').read_text())
 if opt['winner']!='original':assert (ROOT/'selected_optimizer_judged.json').exists(),'Selected optimizer still running'
 metrics=summarize();metrics['results']=[r for r in metrics['results'] if '/cot_only/' in r['condition_id']];assert not metrics['failed_requests'],'Resolve missing semantic responses before recipe selection'
 expected=51+(24 if opt['winner']!='original' else 0)
 for task in [4,5]:assert sum(r['task']==task for r in metrics['results'])==expected,('Missing planned development conditions',task)
 choices={}
 for task in [4,5]:
  rows=[r for r in metrics['results'] if r['task']==task]
  order=lambda r:(-r['gmean2_cv'],-r['auroc'],r['condition_id'])
  text=min([r for r in rows if r['condition_id'].endswith('/text')],key=order)
  mask=min([r for r in rows if r['condition_id'].split('/')[2].startswith('snp')],key=order)
  choices[str(task)]=dict(text=text['condition_id'],mask=mask['condition_id'],development_gmean2_text=text['gmean2_cv'],development_gmean2_mask=mask['gmean2_cv'])
 result=dict(choices=choices,optimizer=opt['winner'],policy=json.loads((ROOT/'study_policy.json').read_text()),criterion='Maximize six-fold development G, then AUROC, then lexicographically smallest condition ID. No calibration/test outcome used.',development_metrics_sha256=hashlib.sha256((ROOT/'development_metrics.json').read_bytes()).hexdigest())
 save(OUT/'frozen_recipes.json',result);return result

def records():
 cal=[r for r in json.loads((ROOT/'cohort.json').read_text())['records'] if r['tuning_role']=='calibration']
 tests=[r for r in payload()['records'] if r['role'] in ['id_test','ood_test']]
 assert len(cal)==32 and len(tests)==96
 return cal+tests

def mask_spec(cid):
 p=cid.split('/');assert p[1]=='cot_only' and p[2].startswith('snp');return p[0],p[2],p[3],int(p[4][4:])/100,p[-1]=='endpoints'

def submit(recipes):
 tasks_normal=[];tasks_opt=[]
 for r in records():
  sid=r['sample_id'];path=ROOT/'inputs'/f'{sid}.json'
  if not path.exists():save(path,json.loads((EXPANDED/'inputs'/f'{sid}.json').read_text()))
  prompt,method,c,rho,endpoint=mask_spec(recipes['choices'][str(r['task'])]['mask'])
  # Reuse only complete fits. For incomplete original jobs, wait on their exact
  # scheduler IDs rather than creating a second writer or duplicating work.
  source=judge.source_path(sid,method,c,rho)
  complete=source is not None and json.loads(source.read_text()).get('final',False)
  if not complete:
   suffix='_'+method[4:] if method.startswith('snp_') else ''
   task=dict(sample_id=sid,kind='fits',condition=c,retention=rho,recipe=f'{c}_keep{int(100*rho)}{suffix}')
   if suffix:
    variant=method[4:];peak,sparse={'peak10_dense':(10,False),'peak100_dense':(100,False),'peak10_sparse':(10,True)}[variant];task.update(penalty_peak=peak,sparse_init=sparse);tasks_opt.append(task)
   else:tasks_normal.append(task)
  background=c if c.endswith('_off') else 'joint'
  ta=judge.source_path(sid,'ta',c,rho)
  if ta is None or not json.loads(ta.read_text()).get('complete',False):tasks_normal.append(dict(sample_id=sid,kind='ta',condition=background,recipe=background))
 # Held-out original jobs may still be queued/running. If they can supply the
 # exact original 80% fit or TA file, use them and raise only their priority.
 original_map={}
 ledger=json.loads((EXPANDED/'preparation_ledger.json').read_text())
 for batch in ledger['submissions']:
  for i,t in enumerate(json.loads(Path(batch['manifest']).read_text())):original_map[(t['sample_id'],t['kind'],t['condition'])]=f"{batch['job_id']}_{i}"
 queue=dict(line.split('|') for line in subprocess.check_output(['squeue','-r','--me','-h','-o','%i|%T'],text=True).splitlines())
 wait=[];remaining=[]
 for t in tasks_normal:
  jid=original_map.get((t['sample_id'],t['kind'],t['condition']))
  reusable=t['kind']=='ta' or t.get('retention')==.2
  if reusable and jid and queue.get(jid) in ['PENDING','RUNNING']:
   wait.append(jid)
   if queue[jid]=='PENDING':subprocess.run(['scontrol','update',f'JobId={jid}','Priority=10004'],check=True)
  else:remaining.append(t)
 partition=json.loads((EXPANDED/'partition_routing.json').read_text())['discovered_partition']
 ledger_path=OUT/'launch.json'
 assert not ledger_path.exists(),'Inspect existing submission ledger before resuming'
 launch=dict(status='submitting',reuse_jobs=wait,batches=[]);write_json(ledger_path,launch)
 for name,tasks,script in [('normal',remaining,'monitor_tuning_0924'),('optimizer',tasks_opt,'monitor_optimizer_0924')]:
  if not tasks:continue
  manifest=OUT/f'tasks_{name}.json';save(manifest,tasks)
  entry=dict(name=name,manifest=str(manifest),tasks=len(tasks),status='submitting');launch['batches'].append(entry);write_json(ledger_path,launch)
  jid=subprocess.check_output(['sbatch','--parsable','--priority=10004',f'--partition={partition}',f'--array=0-{len(tasks)-1}',f'slurm_scripts/{script}.sbatch',str(manifest)],text=True).strip().split(';')[0]
  entry.update(job_id=jid,status='submitted');write_json(ledger_path,launch);wait.append(jid)
 launch['status']='submitted';write_json(ledger_path,launch)
 if wait:subprocess.run(['monitor_jobs',*wait],check=True)


def conditions(choice):
 prompt,method,c,rho,endpoint=mask_spec(choice['mask'])
 rows=[dict(id='best_text',prompt=choice['text'].split('/')[0],method=None),dict(id='matched_text',prompt=prompt,method=None),dict(id='selected_mask',prompt=prompt,method=method,condition=c,retention=rho,endpoint=endpoint),dict(id='thought_anchors',prompt=prompt,method='ta',condition=c,retention=rho,endpoint=endpoint)]
 for seed in range(5):rows.append(dict(id=f'random{seed}',prompt=prompt,method='random',condition=c,retention=rho,endpoint=endpoint,seed=seed))
 rows.append(dict(id='alternate_graph_display',prompt=prompt,method=method,condition=c,retention=rho,endpoint=not endpoint))
 return rows

def evaluate(rows,recipes,phase,only_conditions=None):
 old.OUT=OUT/'judge';old.OUT.mkdir(parents=True,exist_ok=True);client=get_client().with_options(max_retries=0,timeout=180.);requests=[];mapping={}
 for r in rows:
  record=json.loads((ROOT/'inputs'/f"{r['sample_id']}.json").read_text())
  for spec in conditions(recipes['choices'][str(r['task'])]):
   if only_conditions is not None and spec['id'] not in only_conditions:continue
   packet=old.packet(record,'cot_only')
   if spec['method']:
    ev=judge.evidence(record,spec['condition'],spec['method'],spec['retention'],spec['endpoint'],random_seed=spec.get('seed',0));assert ev is not None,('Missing mask',r['sample_id'],spec);packet['attention_evidence']=ev
   row=dict(sample_id=r['sample_id'],condition_id=spec['id'],packet=packet,request_index=len(requests));requests.append((row,judge.PROMPTS[spec['prompt']]))
   req=dict(model=old.JUDGE,messages=[dict(role='system',content=judge.PROMPTS[spec['prompt']]),dict(role='user',content=old.compact(packet))],temperature=0.,top_p=1.,max_tokens=16384)
   mapping[r['sample_id']+'|'+spec['id']]=hashlib.sha256(old.compact(req).encode()).hexdigest()
 save(OUT/f'{phase}_mapping.json',mapping)
 # Freeze packets and exact prompts before any call in this phase.
 save(OUT/f'{phase}_requests.json',[dict(row=r,system_prompt=p) for r,p in requests])
 with cf.ThreadPoolExecutor(max_workers=8) as ex:
  results=list(ex.map(lambda rp:old.evaluate_one(client,*rp,16384),requests))
 assert all(r['status']=='ok' for r in results),'Incomplete confirmation responses: do not drop samples'
 scores={}
 for key,sha in mapping.items():scores[key]=json.loads((old.OUT/'max16384'/f'{sha}.json').read_text())['parsed']['probability']
 return scores

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--select-only',action='store_true');a=ap.parse_args();OUT.mkdir(exist_ok=True)
 assert json.loads((ROOT/'judge_prompts.json').read_text())==judge.PROMPTS,'Judge prompts changed after development'
 path=OUT/'frozen_recipes.json';recipes=json.loads(path.read_text()) if path.exists() else choose()
 if a.select_only:print(json.dumps(recipes,indent=2));return
 path=OUT/'launch.json'
 if not path.exists():submit(recipes)
 else:
  launch=json.loads(path.read_text());assert launch['status']=='submitted' and all(b['status']=='submitted' for b in launch['batches']);jobs=launch['reuse_jobs']+[b['job_id'] for b in launch['batches']]
  if jobs:subprocess.run(['monitor_jobs',*jobs],check=True)
 allrows=records();cal=[r for r in allrows if r['role']=='calibration'];tests=[r for r in allrows if r['role']!='calibration']
 scores=evaluate(cal,recipes,'calibration');thresholds={}
 for task in [4,5]:
  rr=[r for r in cal if r['task']==task]
  for spec in conditions(recipes['choices'][str(task)]):
   cid=spec['id'];thresholds[f'{task}|{cid}']=float(threshold([r['Y'] for r in rr],[scores[r['sample_id']+'|'+cid] for r in rr]))
 save(OUT/'frozen_thresholds.json',thresholds)
 scores=evaluate(tests,recipes,'test');save(OUT/'test_scores.json',scores)
 from .confirm_metrics import report
 report()
 from .repeat_confirmation import run as repeat_confirmation
 repeat_confirmation()
if __name__=='__main__':main()
