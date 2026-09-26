"""Development metrics use held-out folds for threshold choice, never test labels."""
import collections,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cohort import ROOT,rank
from ..monitor_pilot_0924.run import write_json

def gmean(y,s,t):
 y=np.asarray(y);p=np.asarray(s)>=t
 return float(p[y==1].mean()*(~p[y==0]).mean())
def threshold(y,s):
 return max(np.arange(1,100)/100,key=lambda t:(gmean(y,s,t),-abs(t-.5),t))
def auc(y,s):
 a=np.asarray(s)[np.asarray(y)==1];b=np.asarray(s)[np.asarray(y)==0]
 return float(((a[:,None]>b)+.5*(a[:,None]==b)).mean())
def folds(records):
 grouped=collections.defaultdict(list)
 for r in records:grouped[(r['task'],r['Y'],r['hint_letter'])].append(r)
 mapping={}
 for group in grouped.values():
  assert len(group)==6
  for fold,r in enumerate(sorted(group,key=lambda r:rank(r['base_id']))):mapping[r['sample_id']]=fold
 return mapping

def summarize():
 records=[r for r in json.loads((ROOT/'cohort.json').read_text())['records'] if r['tuning_role']=='development'];byid={r['sample_id']:r for r in records};fm=folds(records)
 grouped=collections.defaultdict(dict);failed=[]
 mapping_path=ROOT/'request_mapping.json'
 if mapping_path.exists():
  entries=[]
  for key,sha in json.loads(mapping_path.read_text()).items():
   sid,condition=key.split('|',1);p=ROOT/'judge/max16384'/f'{sha}.json'
   if p.exists():entries.append((p,sid,condition))
 else:
  entries=[]
  for p in (ROOT/'judge/max16384').glob('*.json'):
   v=json.loads(p.read_text())
   if 'sample_id' in v:entries.append((p,v['sample_id'],v['condition_id']))
 for p,sid,condition in entries:
  v=json.loads(p.read_text())
  if sid not in byid:continue
  if v.get('status')!='ok':failed.append(str(p));continue
  key=(byid[sid]['task'],condition)
  assert sid not in grouped[key],('Duplicate condition',key,sid)
  grouped[key][sid]=v['parsed']['probability']
 result=[]
 for (task,c),scores in sorted(grouped.items()):
  rows=sorted([r for r in records if r['task']==task],key=lambda r:r['sample_id']);n=len(scores)
  if n!=24:continue
  y=np.array([r['Y'] for r in rows]);s=np.array([scores[r['sample_id']] for r in rows]);ff=np.array([fm[r['sample_id']] for r in rows]);cv=np.zeros(24);ts=[]
  for fold in range(6):
   train=ff!=fold;test=~train;t=threshold(y[train],s[train]);cv[test]=s[test]>=t;ts.append(float(t))
  result.append(dict(task=task,condition_id=c,n=24,gmean2_at_half=gmean(y,s,.5),gmean2_cv=gmean(y,cv,.5),auroc=auc(y,s),fold_thresholds=ts,probabilities=s.tolist(),labels=y.tolist(),sample_ids=[r['sample_id'] for r in rows]))
 out=dict(results=result,failed_requests=failed,incomplete_conditions={str(k):len(v) for k,v in grouped.items() if len(v)!=24},note='Development-only six-fold cross-validation of thresholds. Each validation fold contains one example per class/answer-letter stratum. Recipe selection on these estimates is exploratory; independent calibration and testing remain required.')
 write_json(ROOT/'development_metrics.json',out);return out

def plot_baseline(out):
 results=[r for r in out['results'] if r['condition_id'].endswith('/text') and '/cot_only/' in r['condition_id']]
 if len(results)!=6:return
 plt.rcParams.update({'font.size':14,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':14})
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 colors=['#0072B2','#009E73','#E69F00']
 for ax,task,title in zip(axes,[4,5],['User preference: Scruples','Professor hint: Daily Dilemmas']):
  rows=sorted([r for r in results if r['task']==task],key=lambda r:['original','reliance','comparative'].index(r['condition_id'].split('/')[0]))
  x=np.arange(3)
  ax.bar(x-.19,[r['gmean2_at_half'] for r in rows],.36,color='#999999',label='Threshold 0.5')
  ax.bar(x+.19,[r['gmean2_cv'] for r in rows],.36,color='#0072B2',label='Threshold chosen on other folds')
  for xx,r in zip(x,rows):ax.text(xx+.19,r['gmean2_cv']+.025,f"{r['gmean2_cv']:.2f}",ha='center',fontsize=13)
  ax.set_xticks(x,['Original','Reliance','Compare\nevidence']);ax.set_ylim(0,1.06);ax.set_title(title,fontsize=15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set_ylabel('Judge g-mean²');axes[1].legend(loc='upper left',fontsize=10)
 fig.tight_layout();d=Path('notes/images/monitor_tuning_0924');d.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(d/f'text_prompt_development.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)


def plot_scope_baselines(out):
 baseline=[r for r in out['results'] if r['condition_id'].endswith('/text')]
 if len(baseline)!=18:return
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(2)
 for offset,view,label,color in [(-.25,'prompt_only','Prompt only','#777777'),(0.,'cot_only','CoT only','#0072B2'),(.25,'all_messages','All messages','#E69F00')]:
  values=[max(r['gmean2_cv'] for r in baseline if r['task']==task and '/'+view+'/' in r['condition_id']) for task in [4,5]]
  ax.bar(x+offset,values,.23,label=label,color=color)
  for xx,v in zip(x+offset,values):ax.text(xx,v+.025,f'{v:.2f}',ha='center',fontsize=14)
 ax.set_xticks(x,['Scruples','Daily Dilemmas']);ax.set_ylim(0,1.04);ax.set_ylabel('Best development g-mean²');ax.legend(fontsize=12);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
 d=Path('notes/images/monitor_tuning_0924')
 for ext in ['png','pdf']:fig.savefig(d/f'text_visibility_development.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)


if __name__=='__main__':
 out=summarize();plot_baseline(out);plot_scope_baselines(out)
 print(json.dumps([{k:v for k,v in r.items() if k not in ['probabilities','labels','sample_ids']} for r in out['results']],indent=2));print('Failures',len(out['failed_requests']),'incomplete',len(out['incomplete_conditions']))

