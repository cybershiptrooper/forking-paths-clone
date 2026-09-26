"""Figures for the 2026-09-26 follow-ups: sentence gap 1 versus 0, and Thought Anchors across trace sets."""
import json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import data as D, analyze as A, judge_ablation as J
FIG=A.FIG

def frame(ax):
 for s in ax.spines.values():s.set_color('k');s.set_linewidth(1.1)

def gap_figure():
 g=json.loads((D.NEW/'gap1_compare.json').read_text());dev=json.loads((A.OUT/'summary_development.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(15,4.6),gridspec_kw=dict(width_ratios=[1,2.3]))
 ax=axes[0];rows=g['rows']
 for i,(k,lab) in enumerate([('learned','learned mask'),('random','random masks'),('empty','no reads between chunks')]):
  for j,(gap,col) in enumerate([('gap0','#D55E00'),('gap1','#009E73')]):
   v=[r[gap][k] for r in rows];x=i+(j-.5)*.35
   ax.bar(x,np.mean(v),.33,color=col,edgecolor='k',linewidth=.5,label=('sentence gap 0 (used before)' if gap=='gap0' else 'sentence gap 1') if i==0 else None)
   ax.scatter(np.full(len(v),x)+np.random.default_rng(0).uniform(-.08,.08,len(v)),v,s=8,color='k',alpha=.4,zorder=3)
 ax.axhline(np.mean([r['gap1']['full'] for r in rows]),ls=':',color='k',label='all reads kept')
 ax.set_xticks(range(3),['learned mask','random masks','no reads\nbetween chunks']);ax.set_ylabel('reasoning NLL, nats per token');ax.set_title('Reasoning likelihood, 24 professor traces, 80% removed',fontsize=11);ax.legend(fontsize=8.5);frame(ax)
 ax=axes[1];cfgs=[c for c in J.CONFIGS if dev['configs'][c].get('gap1_vs_gap0')]
 for i,c in enumerate(cfgs):
  a=dev['configs'][c]['gap1_vs_gap0']['auroc']
  for off,k,col,mk in [(-.24,'text','#000000','s'),(-.08,'snp_sparse','#D55E00','o'),(.08,'snp_gap1','#009E73','o')]:
   ax.scatter(i+off,a[k]['professor'],color=col,marker=mk,s=40,edgecolor='k',linewidth=.4,zorder=3)
  rv=[a['random0']['professor'],a['random1']['professor']];ax.plot([i+.24]*2,[min(rv),max(rv)],color='#999',lw=3);ax.scatter(i+.24,np.mean(rv),color='#999',s=26,zorder=3)
 ax.set_xticks(range(len(cfgs)),[A.CONFIG_LABEL[c] for c in cfgs],rotation=30,ha='right');ax.set_ylim(.75,1);ax.set_ylabel('AUROC, professor task')
 for col,mk,lab in [('#000000','s','text only'),('#D55E00','o','SNP, sentence gap 0'),('#009E73','o','SNP, sentence gap 1'),('#999999','o','random mask (2 seeds, range)')]:ax.scatter([],[],color=col,marker=mk,label=lab)
 ax.legend(ncol=4,fontsize=8.5,loc='lower left');ax.set_title('Judge AUROC on the 24 professor development traces',fontsize=11);ax.grid(axis='y',alpha=.25);frame(ax)
 fig.tight_layout();[fig.savefig(FIG/f'gap1_vs_gap0.{e}',dpi=160,bbox_inches='tight') for e in ('png','pdf')];plt.close(fig)

def sets_figure():
 pooled=json.loads((A.OUT/'heldout_pooled.json').read_text());fig,axes=plt.subplots(1,2,figsize=(13,4.3),sharey=True)
 for ax,cfg in zip(axes,['base','legacy']):
  xs=[];labels=[]
  for i,st in enumerate(['development','calibration','extension']):
   e=json.loads((A.OUT/f'summary_{st}.json').read_text())['configs'][cfg]['ta']
   for off,k,col in [(-.12,'gain_over_text','#E69F00'),(.12,'gain_over_random','#8c6d31')]:
    g=e[k]['mean'];ax.errorbar(i+off,g['point'],yerr=[[g['point']-g['ci'][0]],[g['ci'][1]-g['point']]],fmt='o' if k=='gain_over_text' else 'D',color=col,capsize=3,mfc=col if k=='gain_over_text' else 'white')
   labels.append(f"{st} ({e['n']})")
  for off,k,col in [(-.12,'ta_minus_text','#E69F00'),(.12,'ta_minus_random','#8c6d31')]:
   p,lo,hi=pooled[f'{cfg}|{k}'];ax.errorbar(3+off,p,yerr=[[p-lo],[hi-p]],fmt='o' if 'text' in k else 'D',color=col,capsize=3,mfc=col if 'text' in k else 'white')
  labels.append('calibration +\nextension pooled')
  ax.axhline(0,color='k',lw=.8);ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(A.CONFIG_LABEL[cfg],fontsize=11);ax.grid(axis='y',alpha=.25);frame(ax)
 axes[0].set_ylabel('Thought Anchors AUROC gain, mean of tasks')
 axes[1].errorbar([],[],fmt='o',color='#E69F00',label='minus text only');axes[1].errorbar([],[],fmt='D',color='#8c6d31',mfc='white',label='minus random masks');axes[1].legend(fontsize=9)
 fig.tight_layout();[fig.savefig(FIG/f'ta_across_sets.{e}',dpi=160,bbox_inches='tight') for e in ('png','pdf')];plt.close(fig)

if __name__=='__main__':
 gap_figure();sets_figure()
