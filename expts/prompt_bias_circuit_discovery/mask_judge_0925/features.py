"""Step 1: does the mask itself carry label signal about reliance on the cue, beyond the text?

For every trace and mask, with C the reasoning chunks that refer to the suggestion (cue_marks) and
the eligible reasoning-to-reasoning cells (reader chunk, earlier-or-same source chunk):
  cue_read_enrichment      kept fraction of cells reading a cue chunk from a later chunk / kept fraction of all between-chunk cells
  late_cue_read_enrichment same, restricted to readers in the final quarter of the reasoning chunks
  cue_read_score_rank      mean within-trace percentile of the mask's continuous score (SNP gate alpha,
                           Thought Anchors KL, random) over cells reading a cue chunk from a later chunk
Traces without a cue chunk get the neutral value (1 for enrichments, 0.5 for the rank).
Higher values are read as more reliance; the direction is fixed before looking.
Text-only comparators: whether any chunk refers to the cue, the fraction of chunks that do, and the
relative position of the last cue reference.
"""
import json,collections
import numpy as np
from . import data as D, cue_marks as C

def auroc(y,s):
 y=np.asarray(y);s=np.asarray(s,float);a=s[y==1];b=s[y==0]
 return float(((a[:,None]>b)+.5*(a[:,None]==b)).mean())

def boot_ci(y,s,groups,n=4000,seed=0):
 """Percentile interval, resampling traces within each (task, label) stratum."""
 rng=np.random.default_rng(seed);y=np.asarray(y);s=np.asarray(s,float);groups=np.asarray(groups);vals=[]
 idx={g:np.where(groups==g)[0] for g in set(groups.tolist())}
 for _ in range(n):
  pick=np.concatenate([rng.choice(v,len(v)) for v in idx.values()])
  vals.append(auroc(y[pick],s[pick]))
 return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

def mask_features(sid,condition,method,seed=0):
 m=D.mask(sid,condition,method,seed)
 if m is None:return None
 ids=[c['id'] for c in D.reasoning_chunks(sid)];order={c:i for i,c in enumerate(ids)};n=len(ids)
 cue=set(C.load(sid));late={c for c in ids if order[c]>=n-max(1,round(n/4))}
 b=np.asarray(m['binary'],float);score=np.asarray(m['score'],float)
 rank=score.argsort().argsort()/(len(score)-1)
 between=np.array([q!=k for q,k in m['pairs']]);base=b[between].mean()
 cue_cells=[i for i,(q,k) in enumerate(m['pairs']) if k in cue and q!=k]
 late_cells=[i for i in cue_cells if m['pairs'][i][0] in late]
 enr=lambda cells:float(b[cells].mean()/base) if cells and base>0 else 1.  # no between-chunk cell kept: neutral
 return dict(cue_read_enrichment=enr(cue_cells),late_cue_read_enrichment=enr(late_cells),
             cue_read_score_rank=float(rank[cue_cells].mean()) if cue_cells else .5,n_cue_cells=len(cue_cells))

def text_features(sid):
 ids=[c['id'] for c in D.reasoning_chunks(sid)];cue=C.load(sid)
 return dict(mentions_cue=float(bool(cue)),cue_chunk_fraction=len(cue)/len(ids),
             last_cue_position=max((ids.index(c)+1)/len(ids) for c in cue) if cue else 0.)

METHODS=[('snp_sparse',None),('snp_original',None),('ta',None),('ta_all_cells',None)]+[('random',s) for s in range(20)]

def main():
 T=D.traces();out=dict(rows=[],results=[])
 for role_set,name in [(('development',),'development'),(('calibration',),'calibration')]:
  sids=sorted(s for s,v in T.items() if v['role'] in role_set)
  for condition in ['rr_off','rr_on']:
   feats=collections.defaultdict(dict)
   for sid in sids:
    for method,seed in METHODS:
     f=mask_features(sid,condition,method,seed or 0)
     if f is not None:feats[(method,seed)][sid]=f
   for sid in sids:
    for k,v in text_features(sid).items():feats[('text',k)][sid]={k:v}
   for task in [4,5,'both']:
    keep=[s for s in sids if task=='both' or T[s]['task']==task]
    for key,by in feats.items():
     have=[s for s in keep if s in by]
     if len(have)<len(keep):continue  # compare methods only on the complete set
     y=[T[s]['Y'] for s in have];groups=[f"{T[s]['task']}_{T[s]['Y']}" for s in have]
     names=[key[1]] if key[0]=='text' else ['cue_read_enrichment','late_cue_read_enrichment','cue_read_score_rank']
     for f in names:
      s=[by[x][f] for x in have]
      r=dict(set=name,condition=condition,task=task,method=key[0],seed=key[1] if key[0]=='random' else None,feature=f,n=len(have),n_pos=int(sum(y)),auroc=auroc(y,s))
      if key[0]!='random':r['ci95']=boot_ci(y,s,groups)
      out['results'].append(r)
 # random masks: summarise the 20 seeds as a null distribution per feature
 summary=[]
 grouped=collections.defaultdict(list)
 for r in out['results']:
  k=(r['set'],r['condition'],r['task'],r['method'],r['feature'])
  grouped[k].append(r)
 for k,rs in grouped.items():
  if k[3]=='random':
   a=[r['auroc'] for r in rs];summary.append(dict(set=k[0],condition=k[1],task=k[2],method='random',feature=k[4],n=rs[0]['n'],auroc_mean=float(np.mean(a)),auroc_5_95=[float(np.percentile(a,5)),float(np.percentile(a,95))],seeds=len(a)))
  else:summary.append(dict(rs[0]))
 out['summary']=summary
 path=D.NEW/'features.json';path.write_text(json.dumps(out,indent=1))
 for r in summary:
  if r['set']=='development' and r['task']!='both':
   a=r.get('auroc',r.get('auroc_mean'));ci=r.get('ci95',r.get('auroc_5_95'))
   print(f"{r['condition']:6s} task{r['task']} n={r['n']:2d} {r['method']:13s} {r['feature']:26s} AUROC {a:.3f} [{ci[0]:.2f},{ci[1]:.2f}]")


TEXT_F=['mentions_cue','cue_chunk_fraction','last_cue_position']
MASK_F=['cue_read_enrichment','late_cue_read_enrichment','cue_read_score_rank']

def loo_auc(X,y):
 """Leave-one-out AUROC of a standardised L2 logistic regression (C=1), fitted within one task."""
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import make_pipeline
 X=np.asarray(X,float);y=np.asarray(y);p=np.zeros(len(y))
 for i in range(len(y)):
  tr=np.arange(len(y))!=i
  if X[tr].std(0).max()==0:p[i]=.5;continue
  m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000)).fit(X[tr],y[tr]);p[i]=m.predict_proba(X[i:i+1])[0,1]
 return auroc(y,p)

def added_value(condition='rr_off',role='development',methods=('snp_sparse','ta','ta_all_cells','snp_original'),random_seeds=20):
 """Does adding a mask's cue-read features to the text statistics raise leave-one-out AUROC?"""
 T=D.traces();out=[]
 for task in (4,5):
  sids=sorted(s for s,v in T.items() if v['role']==role and v['task']==task)
  for method in list(methods)+['random']:
   seeds=range(random_seeds) if method=='random' else [0]
   vals=[]
   for seed in seeds:
    rows=[(s,mask_features(s,condition,method,seed)) for s in sids]
    rows=[(s,f) for s,f in rows if f is not None]
    if len(rows)<len(sids):break
    y=[T[s]['Y'] for s,_ in rows];tf=[[text_features(s)[k] for k in TEXT_F] for s,_ in rows]
    both=[t+[f[k] for k in MASK_F] for t,(_,f) in zip(tf,rows)]
    vals.append((loo_auc(tf,y),loo_auc(both,y),loo_auc([[f[k] for k in MASK_F] for _,f in rows],y)))
   if not vals:continue
   v=np.asarray(vals)
   out.append(dict(task=task,method=method,n=len(sids),text_only=float(v[:,0].mean()),text_plus_mask=float(v[:,1].mean()),mask_only=float(v[:,2].mean()),
                   text_plus_mask_5_95=[float(np.percentile(v[:,1],5)),float(np.percentile(v[:,1],95))] if method=='random' else None))
 return out

def plot(set_name='development',condition='rr_off'):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from pathlib import Path
 S=json.loads((D.NEW/'features.json').read_text())['summary']
 fig,axes=plt.subplots(1,2,figsize=(15,4.8),sharey=True)
 labels={'cue_read_enrichment':'cue reads kept\n(relative rate)','late_cue_read_enrichment':'cue reads kept,\nfinal quarter','cue_read_score_rank':'mask score of\ncue reads (rank)',
         'mentions_cue':'any cue\nmention','cue_chunk_fraction':'fraction of chunks\nmentioning cue','last_cue_position':'position of last\ncue mention'}
 colors={'snp_sparse':'#0072B2','ta':'#E69F00','ta_all_cells':'#D55E00','text':'#000000'}
 names={'snp_sparse':'SNP, revised optimizer','ta':'Thought Anchors (between-chunk top 20%)','ta_all_cells':'Thought Anchors (all cells top 20%)','text':'text statistic'}
 for ax,task,title in zip(axes,[4,5],['Scruples (user preference)','Daily Dilemmas (professor hint)']):
  rows=[r for r in S if r['set']==set_name and r['condition']==condition and r['task']==task]
  feats=MASK_F+TEXT_F
  for i,f in enumerate(feats):
   rnd=[r for r in rows if r['method']=='random' and r['feature']==f]
   if rnd:ax.fill_between([i-.35,i+.35],*rnd[0]['auroc_5_95'],color='#ccc',zorder=0)
   ms=[m for m in ['snp_sparse','ta','ta_all_cells'] if f in MASK_F] or ['text']
   for j,m in enumerate(ms):
    r=[x for x in rows if x['method']==m and x['feature']==f]
    if not r:continue
    r=r[0];x=i+(j-(len(ms)-1)/2)*.22
    ax.errorbar(x,r['auroc'],yerr=[[r['auroc']-r['ci95'][0]],[r['ci95'][1]-r['auroc']]],fmt='o',color=colors[m],capsize=2,ms=6)
  ax.axhline(.5,ls=':',color='#555');ax.axvline(len(MASK_F)-.5,color='#888',lw=.8)
  ax.set_xticks(range(len(feats)),[labels[f] for f in feats],fontsize=8.5);ax.set_title(title+', 24 traces',fontsize=11);ax.set_ylim(0,1.02)
  for s in ax.spines.values():s.set_color('k');s.set_linewidth(1.1)
 axes[0].set_ylabel('AUROC against released label')
 for m in ['snp_sparse','ta','ta_all_cells','text']:axes[1].errorbar([],[],fmt='o',color=colors[m],label=names[m])
 axes[1].fill_between([],[],[],color='#ccc',label='20 random masks, 5-95%');axes[1].legend(fontsize=8,loc='lower right')
 fig.tight_layout();d=Path('notes/images/mask_judge_0925');d.mkdir(parents=True,exist_ok=True)
 for e in ('png','pdf'):fig.savefig(d/f'mask_features_{set_name}.{e}',dpi=160,bbox_inches='tight')

if __name__=='__main__':
 main();plot()
 av=added_value();Path=__import__('pathlib').Path
 (D.NEW/'features_added_value.json').write_text(json.dumps(av,indent=1))
 for r in av:print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
