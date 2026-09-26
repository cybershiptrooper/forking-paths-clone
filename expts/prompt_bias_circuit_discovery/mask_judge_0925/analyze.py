"""Metrics and figures for the judge ablations (plan.json). No model or API calls."""
import json,collections,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import data as D, judge_ablation as J
from ..monitor_tuning_0924 import metrics as TM

FIG=Path('notes/images/mask_judge_0925');OUT=D.NEW/'analysis'
METHODS=['snp_sparse','snp_original','ta','snp_gap1']
NAMES={'text':'text only','snp_sparse':'SNP, revised optimizer','snp_original':'SNP, original optimizer','ta':'Thought Anchors','random':'random mask','snp_gap1':'SNP, revised optimizer, sentence gap 1'}
CONFIG_LABEL={'base':'base','format_raw':'format: raw list','format_inline':'format: inline','format_topk':'format: top 15',
 'description_minimal':'description: minimal','description_mechanism':'description: no usage guide','answer_probability':'answer: probability only',
 'answer_structured':'answer: structured','no_cue_marks':'no cue marks','framing_reliance':'framing: reliance','view_all_messages':'judge also sees prompt and answer','view_cot_action':'judge also sees the final answer','background_on':'outside attention on','legacy':'stopped-study setup'}
COLOR={'text':'#000000','snp_sparse':'#0072B2','snp_original':'#56B4E9','ta':'#E69F00','random':'#999999','snp_gap1':'#009E73'}

def scores(set_name,configs,replicate=0):
 sids=J.sets()[set_name];out=collections.defaultdict(dict);fails=collections.Counter()
 for cfg in configs:
  for r in J.build(cfg,sids,J.CONDITIONS,replicate):
   v=J.lookup(r)
   if v is None:continue
   if v.get('status')=='ok':out[(cfg,r['condition'])][r['sample_id']]=float(v['parsed']['probability'])
   else:fails[(cfg,r['condition'])]+=1
 return out,fails

def auroc(y,s):
 y=np.asarray(y);s=np.asarray(s,float);a=s[y==1];b=s[y==0]
 if len(a)==0 or len(b)==0:return np.nan
 return float(((a[:,None]>b)+.5*(a[:,None]==b)).mean())

def task_mean_auc(sids,sc,T):
 v=[auroc([T[s]['Y'] for s in sids if T[s]['task']==t],[sc[s] for s in sids if T[s]['task']==t]) for t in (4,5)]
 return v,float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan

def compare(sc_by_cond,T,conds,draws=4000,seed=0):
 """AUROC per task and mean for each condition on the traces every listed condition has; paired bootstrap."""
 common=sorted(set.intersection(*[set(sc_by_cond[c]) for c in conds]))
 res=dict(n=len(common),n_by_task={t:sum(T[s]['task']==t for s in common) for t in (4,5)},auroc={})
 for c in conds:
  per,mean=task_mean_auc(common,sc_by_cond[c],T);res['auroc'][c]=dict(scruples=per[0],professor=per[1],mean=mean)
 rng=np.random.default_rng(seed);strata=collections.defaultdict(list)
 for s in common:strata[(T[s]['task'],T[s]['Y'])].append(s)
 boots={c:[] for c in conds}
 for _ in range(draws):
  pick=[x for v in strata.values() for x in rng.choice(v,len(v))]
  for c in conds:
   per,mean=task_mean_auc(pick,sc_by_cond[c],T);boots[c].append([per[0],per[1],mean])
 res['boot']={c:np.asarray(v) for c,v in boots.items()}
 return res

def diff(res,a,b,col=2):
 """a minus b (b may be a list averaged); returns point, 95% interval."""
 bs=b if isinstance(b,list) else [b]
 pa=res['auroc'][a][['scruples','professor','mean'][col]];pb=np.mean([res['auroc'][x][['scruples','professor','mean'][col]] for x in bs])
 d=res['boot'][a][:,col]-np.mean([res['boot'][x][:,col] for x in bs],axis=0)
 return dict(point=float(pa-pb),ci=[float(np.nanpercentile(d,2.5)),float(np.nanpercentile(d,97.5))])

def gmean_cv(sc,T):
 recs=[r for r in json.loads((D.TUN/'cohort.json').read_text())['records'] if r['tuning_role']=='development'];fm=TM.folds(recs);out={}
 for t in (4,5):
  rows=[r for r in recs if r['task']==t and r['sample_id'] in sc]
  if len(rows)<24:out[t]=None;continue
  y=np.array([r['Y'] for r in rows]);s=np.array([sc[r['sample_id']] for r in rows]);ff=np.array([fm[r['sample_id']] for r in rows]);cv=np.zeros(len(rows))
  for f in range(6):
   tr=ff!=f;cv[~tr]=s[~tr]>=TM.threshold(y[tr],s[tr])
  out[t]=TM.gmean(y,cv,.5)
 return out

def frame(ax):
 for s in ax.spines.values():s.set_visible(True);s.set_color('k');s.set_linewidth(1.1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--set',default='development');a=ap.parse_args()
 T=D.traces();configs=list(J.CONFIGS);sc,fails=scores(a.set,configs)
 FIG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 summary=dict(set=a.set,failures={f'{k[0]}/{k[1]}':v for k,v in fails.items()},configs={})
 for cfg in configs:
  bycond={c:sc.get((cfg,c),{}) for c in ['text','snp_sparse','snp_original','ta','random0','random1','snp_gap1']}
  entry={}
  for m in METHODS:
   if len(bycond[m])<10:continue
   conds=['text',m,'random0','random1'];res=compare(bycond,T,conds)
   entry[m]=dict(n=res['n'],n_by_task=res['n_by_task'],auroc=res['auroc'],
     gain_over_text={k:diff(res,m,'text',i) for i,k in enumerate(['scruples','professor','mean'])},
     gain_over_random={k:diff(res,m,['random0','random1'],i) for i,k in enumerate(['scruples','professor','mean'])},
     random_over_text={k:diff(res,['random0','random1'][0],'text',i) for i,k in enumerate(['scruples','professor','mean'])},
     mean_abs_change_vs_text=float(np.mean([abs(bycond[m][s]-bycond['text'][s]) for s in bycond[m] if s in bycond['text']])),
     random_mean_abs_change_vs_text=float(np.mean([abs(bycond['random0'][s]-bycond['text'][s]) for s in bycond['random0'] if s in bycond['text'] and s in bycond[m]])))
  # does the score shift a mask causes (masked minus text-only) point toward the label?
  entry['shift_auroc']={}
  if len(bycond['snp_gap1'])>=10:  # sentence gap 1 versus gap 0, same traces and configuration
   res=compare(bycond,T,['text','snp_gap1','snp_sparse','random0','random1'])
   entry['gap1_vs_gap0']=dict(n=res['n'],auroc=res['auroc'],gap1_minus_gap0={k:diff(res,'snp_gap1','snp_sparse',i) for i,k in enumerate(['scruples','professor','mean'])})
  for c in ['snp_sparse','snp_original','ta','random0','random1','snp_gap1']:
   common=[x for x in bycond[c] if x in bycond['text']]
   if len(common)<10:continue
   sh={x:bycond[c][x]-bycond['text'][x] for x in common}
   per=[auroc([T[x]['Y'] for x in common if T[x]['task']==t],[sh[x] for x in common if T[x]['task']==t]) for t in (4,5)]
   entry['shift_auroc'][c]=dict(n=len(common),scruples=per[0],professor=per[1],mean=float(np.nanmean(per)))
  if a.set=='development':entry['gmean2_cv']={c:gmean_cv(bycond[c],T) for c in bycond if len(bycond[c])>=48}
  summary['configs'][cfg]=entry
 # replicate noise floor for the base configuration
 rep,_=scores(a.set,['base'],replicate=1)
 noise={}
 for c in ['text','ta','snp_sparse']:
  r0=sc.get(('base',c),{});r1=rep.get(('base',c),{})
  if len(r1)>=10:
   res=compare({'r0':r0,'r1':r1},T,['r0','r1'])
   noise[c]=dict(n=res['n'],auroc_rep0=res['auroc']['r0'],auroc_rep1=res['auroc']['r1'],diff_mean=diff(res,'r1','r0',2),
                 mean_abs_prob_change=float(np.mean([abs(r1[s]-r0[s]) for s in r1 if s in r0])))
 summary['replicate_noise']=noise
 (OUT/f'summary_{a.set}.json').write_text(json.dumps(summary,indent=1))
 plot(summary,a.set)
 return summary

def plot(summary,set_name):
 cfgs=[c for c in J.CONFIGS if 'ta' in summary['configs'][c]]
 plt.rcParams.update({'font.size':11})
 fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
 for ax,(key,title) in zip(axes,[('scruples','Scruples (user preference), 24 traces'),('professor','Daily Dilemmas (professor hint), 24 traces'),('mean','mean of the two tasks')]):
  for i,cfg in enumerate(cfgs):
   e=summary['configs'][cfg];ref=e['ta']['auroc']
   ax.scatter(i-.24,ref['text'][key],color=COLOR['text'],marker='s',s=36,zorder=4)
   rv=[ref['random0'][key],ref['random1'][key]];ax.plot([i+.24,i+.24],[min(rv),max(rv)],color=COLOR['random'],lw=3,zorder=3);ax.scatter(i+.24,np.mean(rv),color=COLOR['random'],s=30,zorder=4)
   for off,m in [(-.08,'ta'),(.08,'snp_sparse')]:
    if m in e:
     ax.scatter(i+off,e[m]['auroc'][m][key],color=COLOR[m],s=46,zorder=5,edgecolor='k',linewidth=.5)
  ax.axhline(.5,ls=':',color='#777');ax.set_ylabel('AUROC');ax.set_title(title,fontsize=11);ax.set_ylim(.35,1.0);ax.grid(axis='y',alpha=.25);frame(ax)
 axes[-1].set_xticks(range(len(cfgs)),[CONFIG_LABEL[c] for c in cfgs],rotation=30,ha='right')
 for m,mk in [('text','s'),('ta','o'),('snp_sparse','o'),('random','o')]:axes[0].scatter([],[],color=COLOR[m],marker=mk,label=NAMES[m]+(' (2 seeds, range)' if m=='random' else ''))
 axes[0].legend(ncol=4,fontsize=9,loc='lower left')
 fig.tight_layout();[fig.savefig(FIG/f'judge_auroc_{set_name}.{e}',dpi=160,bbox_inches='tight') for e in ('png','pdf')];plt.close(fig)
 # gains with intervals (mean over tasks)
 fig,ax=plt.subplots(figsize=(13,4.6))
 for i,cfg in enumerate(cfgs):
  e=summary['configs'][cfg]
  for off,m,k in [(-.2,'ta','gain_over_text'),(-.07,'snp_sparse','gain_over_text'),(.07,'ta','gain_over_random'),(.2,'snp_sparse','gain_over_random')]:
   if m not in e:continue
   g=e[m][k]['mean'];mk='o' if k=='gain_over_text' else 'D'
   ax.errorbar(i+off,g['point'],yerr=[[g['point']-g['ci'][0]],[g['ci'][1]-g['point']]],fmt=mk,color=COLOR[m],ms=6,capsize=2,mfc=COLOR[m] if k=='gain_over_text' else 'white')
 nz=summary.get('replicate_noise',{}).get('text')
 if nz:ax.axhspan(-abs(nz['diff_mean']['point']),abs(nz['diff_mean']['point']),color='#ddd',zorder=0)
 ax.axhline(0,color='k',lw=.8);ax.set_xticks(range(len(cfgs)),[CONFIG_LABEL[c] for c in cfgs],rotation=30,ha='right');ax.set_ylabel('AUROC difference, mean of tasks')
 for m,k,mk,fc in [('ta','gain_over_text','o',None),('snp_sparse','gain_over_text','o',None),('ta','gain_over_random','D','white'),('snp_sparse','gain_over_random','D','white')]:
  ax.errorbar([],[],fmt=mk,color=COLOR[m],mfc=fc or COLOR[m],label=f"{NAMES[m]} minus {'text only' if k=='gain_over_text' else 'random mask'}")
 ax.legend(ncol=2,fontsize=9);ax.grid(axis='y',alpha=.25);frame(ax);fig.tight_layout()
 [fig.savefig(FIG/f'judge_gains_{set_name}.{e}',dpi=160,bbox_inches='tight') for e in ('png','pdf')];plt.close(fig)

if __name__=='__main__':
 s=main()
 for cfg,e in s['configs'].items():
  for m in METHODS:
   if m in e:
    g=e[m]['gain_over_text']['mean'];r=e[m]['gain_over_random']['mean'];au=e[m]['auroc']
    print(f"{cfg:22s} {m:12s} n={e[m]['n']:2d} text {au['text']['mean']:.3f} method {au[m]['mean']:.3f} random {np.mean([au['random0']['mean'],au['random1']['mean']]):.3f} | -text {g['point']:+.3f} [{g['ci'][0]:+.2f},{g['ci'][1]:+.2f}] -random {r['point']:+.3f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] | |dp| {e[m]['mean_abs_change_vs_text']:.3f} vs random {e[m]['random_mean_abs_change_vs_text']:.3f}")
 print('failures',s['failures']);print('noise',json.dumps(s['replicate_noise'])[:600])

def plot_shift_and_heldout():
 """Left: AUROC of the score shift a mask causes (masked minus text-only) against the label, per
 configuration. Right: held-out calibration traces for the two configurations run there."""
 T=D.traces();dev=json.loads((OUT/'summary_development.json').read_text())
 cfgs=[c for c in J.CONFIGS if 'shift_auroc' in dev['configs'][c]]
 fig,axes=plt.subplots(1,2,figsize=(15,4.8),gridspec_kw=dict(width_ratios=[2.3,1]))
 ax=axes[0]
 for i,cfg in enumerate(cfgs):
  sa=dev['configs'][cfg]['shift_auroc']
  for off,c,col,mk in [(-.2,'ta',COLOR['ta'],'o'),(0,'snp_sparse',COLOR['snp_sparse'],'o'),(.13,'random0',COLOR['random'],'o'),(.26,'random1',COLOR['random'],'o')]:
   if c in sa:ax.scatter(i+off,sa[c]['mean'],color=col,marker=mk,s=40,edgecolor='k',linewidth=.4,zorder=3)
 ax.axhline(.5,ls=':',color='#555');ax.set_xticks(range(len(cfgs)),[CONFIG_LABEL[c] for c in cfgs],rotation=30,ha='right');ax.set_ylim(.2,.9)
 ax.set_ylabel('AUROC of (masked minus text-only score)\nagainst the label, mean of tasks');ax.set_title('Does the mask move the judge toward the right answer? (48 development traces)',fontsize=11)
 for c,lab in [('ta',NAMES['ta']),('snp_sparse',NAMES['snp_sparse']),('random',NAMES['random']+', 2 seeds')]:ax.scatter([],[],color=COLOR[c],edgecolor='k',linewidth=.4,label=lab)
 ax.legend(fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.25);frame(ax)
 ax=axes[1];p=OUT/'summary_calibration.json'
 if p.exists():
  cal=json.loads(p.read_text());x=0;ticks=[]
  for cfg in ['legacy','base']:
   e=cal['configs'].get(cfg,{}).get('ta')
   if not e:continue
   au=e['auroc'];vals=[('text',au['text']['mean'],COLOR['text']),('ta',au['ta']['mean'],COLOR['ta']),('random',np.mean([au['random0']['mean'],au['random1']['mean']]),COLOR['random'])]
   if 'snp_sparse' in cal['configs'][cfg]:vals.insert(2,('snp_sparse',cal['configs'][cfg]['snp_sparse']['auroc']['snp_sparse']['mean'],COLOR['snp_sparse']))
   for j,(n,v,col) in enumerate(vals):ax.bar(x+j*.8,v,.7,color=col,edgecolor='k',linewidth=.5);ax.text(x+j*.8,v+.01,f'{v:.2f}',ha='center',fontsize=9)
   ticks.append((x+(len(vals)-1)*.4,CONFIG_LABEL[cfg]));x+=len(vals)*.8+1
  ax.set_xticks([t[0] for t in ticks],[t[1] for t in ticks]);ax.set_ylim(.4,1);ax.axhline(.5,ls=':',color='#555')
  ax.set_ylabel('AUROC, mean of tasks');ax.set_title('Held-out calibration traces (32)',fontsize=11);frame(ax)
  for c,lab in [('text',NAMES['text']),('ta',NAMES['ta']),('random',NAMES['random']+', mean of 2 seeds')]:ax.bar([0],[0],color=COLOR[c],edgecolor='k',linewidth=.5,label=lab)
  ax.legend(fontsize=8.5,loc='upper right')
 fig.tight_layout();[fig.savefig(FIG/f'judge_shift_and_heldout.{e}',dpi=160,bbox_inches='tight') for e in ('png','pdf')];plt.close(fig)
