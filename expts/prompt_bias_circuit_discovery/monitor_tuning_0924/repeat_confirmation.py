"""Two predeclared repeats of both headline monitors; never choose a favorable call."""
import json
import numpy as np
from pathlib import Path
from . import confirm
from .cohort import ROOT,save
from .metrics import threshold,gmean
from ..monitor_pilot_0924.run import write_json

def run():
 original_out=confirm.OUT;recipes=json.loads((original_out/'frozen_recipes.json').read_text());allrows=confirm.records();cal=[r for r in allrows if r['role']=='calibration'];tests=[r for r in allrows if r['role']!='calibration'];summary=[]
 try:
  for repeat in [1,2]:
   confirm.OUT=original_out/'repeats'/f'repeat{repeat}';confirm.OUT.mkdir(parents=True,exist_ok=True)
   save(confirm.OUT/'frozen_recipes.json',recipes)
   cal_scores=confirm.evaluate(cal,recipes,'calibration',only_conditions={'selected_mask','best_text'});thresholds={}
   for task in [4,5]:
    rr=[r for r in cal if r['task']==task]
    for cid in ['selected_mask','best_text']:thresholds[f'{task}|{cid}']=float(threshold([r['Y'] for r in rr],[cal_scores[r['sample_id']+'|'+cid] for r in rr]))
   save(confirm.OUT/'frozen_thresholds.json',thresholds)
   scores=confirm.evaluate(tests,recipes,'test',only_conditions={'selected_mask','best_text'});save(confirm.OUT/'test_scores.json',scores)
   for task in [4,5]:
    for role in ['id_test','ood_test']:
     rr=[r for r in tests if r['task']==task and r['role']==role];y=[r['Y'] for r in rr];gg={cid:gmean(y,[scores[r['sample_id']+'|'+cid] for r in rr],thresholds[f'{task}|{cid}']) for cid in ['selected_mask','best_text']}
     summary.append(dict(repeat=repeat,task=task,role=role,gmean2=gg,delta_gmean2=gg['selected_mask']-gg['best_text'],n=len(rr)))
 finally:confirm.OUT=original_out
 main=json.loads((original_out/'metrics.json').read_text())['results']
 combined=[dict(repeat=0,task=r['task'],role=r['role'],delta_gmean2=r['delta_gmean2'],n=r['n']) for r in main]+summary
 write_json(original_out/'repeated_judge_results.json',dict(policy='One original evaluation and two prespecified fresh API replicates of both headline methods. Fit thresholds on the same independent calibration questions within each replicate. Do not pool repeats as independent questions and do not choose a favorable repeat.',results=combined))
 report=Path('notes/reports_monitorability/judge_tuning_confirmation_0924.md');text=report.read_text().split('\n## Prespecified judge repeats')[0];text+='\n## Prespecified judge repeats\n\nThe frozen text-only and selected-mask monitors were each evaluated twice more with fresh calls. Each repeat uses its own calibration calls on the same calibration questions. The following differences are ordered original/repeat 1/repeat 2; no repeat was chosen or discarded. They describe judge-call variability and do not add independent questions.\n\n'
 for task in [4,5]:
  for role in ['id_test','ood_test']:
   values=[r['delta_gmean2'] for r in combined if r['task']==task and r['role']==role];text+=f"- Task {task}, {role}: "+', '.join(f'{v:+.3f}' for v in values)+'.\n'
 text+='\nIf these repeats disagree materially, treat the result as unstable rather than selecting the most favorable positive or negative outcome.\n';report.write_text(text)
if __name__=='__main__':run()
