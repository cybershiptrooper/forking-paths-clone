"""Evaluate completed development batches without polling Slurm or touching test outcomes."""
import json,subprocess,sys,time
from pathlib import Path
from .cohort import ROOT
from ..monitor_pilot_0924.run import write_json

def run_judge():
 subprocess.run([sys.executable,'-m','expts.prompt_bias_circuit_discovery.monitor_tuning_0924.judge','--mode','ready'],check=True)
 subprocess.run([sys.executable,'-m','expts.prompt_bias_circuit_discovery.monitor_tuning_0924.metrics'],check=True)

def main():
 preparation=json.loads((ROOT/'preparation.json').read_text());assert preparation['status']=='complete'
 jobs=[r['job_id'] for r in preparation['submissions']]
 legacy=[r['job_id'] for r in json.loads((ROOT/'legacy_development_priority.json').read_text())]
 # Scheduler state is delegated to the cluster's array-aware monitor helper.
 subprocess.run(['monitor_jobs',*jobs,*legacy],check=True)
 run_judge()
 write_json(ROOT/'primary_development_complete.json',dict(status='primary_grid_judged',time=time.time(),test_judge_calls=0))
 # Optimizer expansion is independently supervised. If it is queued now, wait
 # on that exact job. Otherwise leave an explicit continuation marker.
 selected=ROOT/'selected_optimizer_launch.json'
 if selected.exists():
  launch=json.loads(selected.read_text());assert launch['status']=='submitted';subprocess.run(['monitor_jobs',launch['job_id']],check=True);run_judge()
  write_json(ROOT/'selected_optimizer_judged.json',dict(status='complete',time=time.time(),job_id=launch['job_id'],test_judge_calls=0))
 else:
  selection=json.loads((ROOT/'optimizer_selection.json').read_text())
  assert selection['winner']=='original','Selected optimizer expansion has not been submitted'
 subprocess.run([sys.executable,'-m','expts.prompt_bias_circuit_discovery.monitor_tuning_0924.confirm'],check=True)
if __name__=='__main__':main()
