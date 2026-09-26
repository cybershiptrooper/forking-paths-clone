import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('notes/images/monitorability_rethink');OUT.mkdir(parents=True,exist_ok=True)
RESULTS=Path('results/prompt_bias_v2/rethink_0924');RESULTS.mkdir(parents=True,exist_ok=True)
# Exact expectation of clipped plug-in sensitivity with known attributable lower bound.
def exact_mc(n,t,r):
 return sum(math.comb(n,k)*t**k*(1-t)**(n-k)*min(1,k/(n*r)) for k in range(n+1))
ns=[1,2,4,8,16,32,64,128]
bias={str(r):{str(n):exact_mc(n,r,r) for n in ns} for r in [.2,.5,.8]}
# Identical observable distributions, opposite hidden attribution classification.
worlds=[]
for overlap in [0,18]:
 worlds.append(dict(n_control=100,n_control_y1=42,n_intervention=100,n_intervention_y1=60,
    n_truly_influenced=18,n_flagged_intervention_y1=18,n_flagged_elsewhere=0,
    n_flagged_and_influenced=overlap,official_gmean2=1,actual_causal_recall=overlap/18))
summary={'known_r_plugin_expectation_when_true_tpr_equals_r':bias,'identical_observable_worlds':worlds,
 'note':'These are mathematical toy calculations, not empirical model results.'}
(RESULTS/'metric_toy.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':12})
fig,axes=plt.subplots(1,2,figsize=(13,4.9))
p0=np.linspace(0,.8,161)
axes[0].plot(p0,np.sqrt(1-p0),lw=2.5,label='Monitor predicts target answer occurred')
axes[0].plot(p0,np.ones_like(p0),lw=2.5,ls='--',label='Monitor predicts intervention AND target')
axes[0].set(xlabel='Control target-answer rate',ylabel='Official intervention score',title='Outcome shortcuts can score highly',ylim=(0,1.05))
axes[0].legend(loc='lower left');axes[0].grid(alpha=.2)
for r,color in zip([.2,.5,.8],['#0072B2','#D55E00','#009E73']):
 axes[1].plot(ns,[bias[str(r)][str(n)] for n in ns],marker='o',lw=2,color=color,label=f'r = true flag rate = {r:g}')
axes[1].axhline(1,color='black',ls='--',lw=1,label='Population minimal-criterion TPR')
axes[1].set_xscale('log',base=2);axes[1].set_xticks([1,2,4,8,16,32,64,128]);axes[1].set_xticklabels([1,2,4,8,16,32,64,128])
axes[1].set(xlabel='Judged intervention positives per input',ylabel='Expected plug-in minimal-criterion TPR',title='Clipping biases small samples downward',ylim=(0,1.05))
axes[1].legend(loc='lower right');axes[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'metric_shortcuts_and_bias.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2));print(OUT/'metric_shortcuts_and_bias.png')
