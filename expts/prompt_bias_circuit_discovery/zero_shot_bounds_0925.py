import json
from pathlib import Path
import numpy as np
from expts.prompt_bias_circuit_discovery import zero_shot_testbed_0924 as b
allresults=[]
for name in ['zero_shot_testbed_0924','zero_shot_own_prompts_0924']:
 root=Path('results/prompt_bias_v2')/name
 rows=json.loads((root/'records.json').read_text());thresholds=json.loads((root/'thresholds.json').read_text());mapping=json.loads((root/'id_test_mapping.json').read_text());cache={}
 for m in mapping:
  cache[m['source_file'],m['kind']]=json.loads((root/'cache'/f'{m["sha"]}.json').read_text())
 for task in [4,5]:
  rr=[r for r in rows if r['split']=='test' and r['task']==task]
  for kind in sorted({m['kind'] for m in mapping}):
   th=.5 if kind=='binary' else thresholds[str(task) if kind=='confidence' else f'{task}/{kind}']['plain']
   missing=[r['source_file'] for r in rr if cache[r['source_file'],kind]['status']!='ok']
   scores=[cache[r['source_file'],kind].get('score') for r in rr]
   mm=[b.metric(rr,[v if v is not None else fill for v in scores],th) for fill in [0.,1.]]
   z=dict(task=task,kind=kind,n=len(rr),threshold=th,missing=missing,bounds={k:[min(m[k] for m in mm),max(m[k] for m in mm)] for k in ['gmean2','tpr','tnr','auroc']+(['gmean2_mc_pooled'] if task==5 else [])})
   allresults.append(z)
p=Path('results/prompt_bias_v2/zero_shot_own_prompts_0924/bounded_comparison.json');p.write_text(json.dumps(allresults,indent=2)+'\n');print(json.dumps(allresults,indent=2))
