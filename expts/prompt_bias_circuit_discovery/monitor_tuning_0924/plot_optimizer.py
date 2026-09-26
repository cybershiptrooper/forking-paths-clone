"""Show every predeclared D01 optimizer result at the final fixed checkpoint."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .cohort import ROOT
from ..monitor_pilot_0924.prepare import ROOT as PILOT

def main():
 labels=['Original','Penalty 10\ndense start','Penalty 100\ndense start','Penalty 10\nsparse start','Random']
 colors=['#777777','#E69F00','#D55E00','#0072B2','#009E73'];data=[]
 for c in ['rr_on','rr_off']:
  orig=json.loads((PILOT/'D01'/f'{c}_mask.json').read_text());vals=[orig['diagnostics']['trace_nll']]
  for variant in ['peak10_dense','peak100_dense','peak10_sparse']:
   p=ROOT/'fits/D01'/f'{c}_keep20_{variant}'/f'{c}_mask.json';m=json.loads(p.read_text());assert m['final'];vals.append(m['diagnostics']['trace_nll'])
  controls=json.loads((PILOT/'D01/controls.json').read_text());random=[controls[f'{c}/random{i}']['trace_nll'] for i in range(5)];vals.append(float(np.mean(random)));data.append(dict(condition=c,nll=vals,random_draws=random))
 (ROOT/'optimizer_D01_plot_data.json').write_text(json.dumps(data,indent=2))
 plt.rcParams.update({'font.size':14,'xtick.labelsize':12,'ytick.labelsize':14})
 fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
 for ax,row,title in zip(axes,data,['Outside connections on','Outside connections off']):
  x=np.arange(5);ax.bar(x,row['nll'],color=colors)
  for xx,v in zip(x,row['nll']):ax.text(xx,v+.15,f'{v:.2f}',ha='center',fontsize=13)
  ax.scatter(np.full(5,4.),row['random_draws'],color='black',s=12,zorder=3);ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title(title,fontsize=15);ax.set_ylim(0,10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set_ylabel('Reasoning NLL per token (lower is better)');fig.tight_layout();d=Path('notes/images/monitor_tuning_0924')
 for ext in ['png','pdf']:fig.savefig(d/f'optimizer_D01.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
