"""Wait via monitor_jobs, select by actor likelihood only, then expand the winner."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from .cohort import ROOT,save
from ..monitor_pilot_0924.judge import ROOT as PILOT
from ..monitor_pilot_0924.expanded_worker import EXPANDED
from ..monitor_pilot_0924.run import write_json

def main():
 launch=json.loads((ROOT/'optimizer_diagnostic_launch.json').read_text());job=launch['job_id']
 legacy=json.loads((ROOT/'legacy_development_priority.json').read_text())
 subprocess.run(['monitor_jobs',job,*[r['job_id'] for r in legacy]],check=True)
 scores=[]
 for config in ['original','peak10_dense','peak100_dense','peak10_sparse']:
  values=[]
  for sid in launch['diagnostic_ids']:
   for c in ['rr_on','rr_off']:
    if config=='original':
     d=EXPANDED/'fits'/sid/c;mp=d/f'{c}_mask.json';cp=d/'controls.json'
     if not mp.exists():
      mp=PILOT/sid/f'{c}_mask.json';z=json.loads((PILOT/sid/'controls.json').read_text());random=[z[f'{c}/random{i}']['trace_nll'] for i in range(5)]
     else:
      z=json.loads(cp.read_text());random=[z[f'random{i}']['trace_nll'] for i in range(5)]
    else:
     d=ROOT/'fits'/sid/f'{c}_keep20_{config}';assert (d/'complete.json').exists();mp=d/f'{c}_mask.json';z=json.loads((d/'controls.json').read_text());random=[z[f'random{i}']['trace_nll'] for i in range(5)]
    mask=json.loads(mp.read_text());assert mask['final'];nll=mask['diagnostics']['trace_nll'];values.append(dict(sample_id=sid,condition=c,nll=nll,random_mean=float(np.mean(random)),excess=nll-float(np.mean(random))))
  scores.append(dict(config=config,mean_excess=float(np.mean([r['excess'] for r in values])),rows=values))
 winner=min(scores,key=lambda r:(r['mean_excess'],r['config']))['config']
 selection=dict(criterion=launch['criterion'],winner=winner,scores=scores,uses_judge_or_labels=False)
 save(ROOT/'optimizer_selection.json',selection);print(json.dumps(selection),flush=True)
 if winner=='original':return
 config={'peak10_dense':(10,False),'peak100_dense':(100,False),'peak10_sparse':(10,True)}[winner]
 records=[r for r in json.loads((ROOT/'cohort.json').read_text())['records'] if r['tuning_role']=='development'];tasks=[]
 for r in records:
  for rho in [.2,.5]:
   for c in ['rr_on','rr_off']:
    recipe=f'{c}_keep{int(100*rho)}_{winner}'
    if (ROOT/'fits'/r['sample_id']/recipe/'complete.json').exists():continue
    tasks.append(dict(sample_id=r['sample_id'],kind='fits',condition=c,retention=rho,penalty_peak=config[0],sparse_init=config[1],recipe=recipe))
 manifest=ROOT/'manifests/selected_optimizer_development.json';save(manifest,tasks)
 path=ROOT/'selected_optimizer_launch.json'
 if path.exists():raise RuntimeError('Do not submit duplicate selected-optimizer work')
 partition=json.loads((EXPANDED/'partition_routing.json').read_text())['discovered_partition']
 write_json(path,dict(status='submitting',manifest=str(manifest),tasks=len(tasks)))
 job=subprocess.check_output(['sbatch','--parsable','--priority=10004',f'--partition={partition}',f'--array=0-{len(tasks)-1}','claude_scripts/monitor_optimizer_0924.sbatch',str(manifest)],text=True).strip().split(';')[0]
 write_json(path,dict(status='submitted',job_id=job,manifest=str(manifest),tasks=len(tasks)))
 print('SELECTED OPTIMIZER EXPANSION',winner,job,len(tasks),flush=True)
 with (ROOT/f'monitor_{job}.log').open('a') as log:subprocess.Popen(['monitor_jobs',job],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
if __name__=='__main__':main()
