"""Figures for the released-data and attention-scope correction."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('notes/images/testbed_audit_0924');OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':15,'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14})
summary=json.loads(Path('results/prompt_bias_v2/testbed_audit_0924/released_summary.json').read_text())
keys=[('4','test'),('4','ood_test'),('5','test'),('5','ood_test')]
labels=['User preference\nID','User preference\nOOD','Professor hint\nID','Professor hint\nOOD']
fig,axes=plt.subplots(1,2,figsize=(15,5.5))
x=np.arange(4)
for ax,field,title in [(axes[0],'n_rollouts','Released labelled rollouts'),(axes[1],'n_base_questions','Distinct base questions')]:
 vals=[summary['tasks'][t]['splits'][s][field] for t,s in keys];ax.bar(x,vals,color=['#0072B2','#56B4E9','#D55E00','#E69F00']);ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_title(title);ax.set_ylim(0,max(vals)*1.18)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,str(v),ha='center')
 ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'released_support.png',dpi=160);plt.close(fig)

D=json.loads(Path('results/prompt_bias_v2/testbed_audit_0924/attention_controls.json').read_text())
conditions=['clean','empty_all_prefix','empty_all_including_probe','p2r_all_kept_complement_zero_readout_R','p2r_empty_complement_zero_readout_R']
labels=['Full','Empty P+R','Empty P+R+probe','P→R full;\nother paths closed*','P→R empty;\nother paths closed*']
fig,axes=plt.subplots(1,2,figsize=(17,6))
for ai,field in enumerate(['conditional','answer_mass']):
 vals=[]
 for name in conditions:
  vals.append(np.mean([r['conditions'][name][field][0 if r['answer']=='A' else 1] if field=='conditional' else r['conditions'][name][field] for r in D['rows']]))
 ax=axes[ai];ax.bar(np.arange(5),vals,color=['#0072B2','#D55E00','#D55E00','#009E73','#009E73']);ax.set_xticks(np.arange(5),labels,rotation=20,ha='right');ax.set_title('Mean P(stored answer | A or B)' if ai==0 else 'Mean full-vocabulary mass on A and B');ax.spines[['top','right']].set_visible(False)
 if ai==0:ax.set_ylim(0,1.12)
 else:ax.set_yscale('log');ax.set_ylim(.0005,1.)
 for i,v in enumerate(vals):ax.text(i,v+.02 if ai==0 else v*1.15,f'{v:.3f}' if ai==0 else f'{v:.4f}',ha='center',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'attention_scope.png',dpi=160);plt.close(fig)

D=json.loads(Path('results/prompt_bias_v2/rethink0924/mask_feasibility/summary.json').read_text())
fig,ax=plt.subplots(figsize=(9,5.5));x=np.arange(2);w=.32
H=json.loads(Path('results/prompt_bias_v2/rethink0924/hint_positive_control/eval.json').read_text())['features']
snp=[np.mean(D['pools']['p2t']['cue_enrichment']['increase']),H['increase']['cue_enrichment']];ta=[np.mean(D['pools']['p2t']['cue_enrichment']['thought_anchors']),H['thought_anchors']['cue_enrichment']]
for off,values,label,color in [(-w/2,snp,'Own-answer SNP','#0072B2'),(w/2,ta,'Thought Anchors','#D55E00')]:
 ax.bar(x+off,values,w,label=label,color=color)
 for i,v in enumerate(values):ax.text(x[i]+off,v+.12 if abs(v-1)<.1 else v+.035,f'{v:.3f}',ha='center')
ax.axhline(1,color='black',ls='--',lw=1);ax.set_xticks(x,['Applicant-name chunks\nmean over 8 traces','Professor-hint sentence\n1 trace']);ax.set_ylabel('Cue-edge retention / overall retention');ax.set_ylim(0,2.15);ax.legend(loc='upper left');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'cue_comparison.png',dpi=160);plt.close(fig)
