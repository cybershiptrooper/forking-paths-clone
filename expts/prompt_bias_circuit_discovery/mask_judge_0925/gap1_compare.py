"""Sentence gap 1 versus gap 0 revised-optimizer masks on the 24 professor development traces (no API calls)."""
import json,statistics as st
import numpy as np
from . import data as D, features as F

def nll(d):
 m=json.loads((d/'rr_off_mask.json').read_text());c=json.loads((d/'controls.json').read_text())
 return dict(learned=m['diagnostics']['trace_nll'],random=st.mean(c[f'random{i}']['trace_nll'] for i in range(5)),full=c['full']['trace_nll'],empty=c['empty']['trace_nll'],clean=c['clean']['trace_nll'])

def main():
 T=D.traces();sids=sorted(s for s,v in T.items() if v['role']=='development' and v['task']==5)
 rows=[]
 for s in sids:
  g1=D.NEW/'fits'/s/'rr_off_keep20_peak10_sparse_gap1'
  g0=next(p for p in [D.NEW/'fits'/s/'rr_off_keep20_peak10_sparse',D.TUN/'fits'/s/'rr_off_keep20_peak10_sparse'] if (p/'complete.json').exists())
  a=D.mask(s,'rr_off','snp_gap1');b=D.mask(s,'rr_off','snp_sparse')
  k1={p for p,on in zip(a['pairs'],a['binary']) if on};k0={p for p,on in zip(b['pairs'],b['binary']) if on and p[0]!=p[1]}
  nb=len(a['pairs']);exp=len(k1)*len(k0)/nb  # expected overlap of two independent random sets of these sizes
  rows.append(dict(sample_id=s,Y=T[s]['Y'],gap1=nll(g1),gap0=nll(g0),between_cells=nb,kept_gap1=len(k1),kept_gap0_between=len(k0),overlap=len(k1&k0),overlap_expected_random=exp,
                   jaccard=len(k1&k0)/len(k1|k0)))
 ex1=[r['gap1']['learned']-r['gap1']['random'] for r in rows];ex0=[r['gap0']['learned']-r['gap0']['random'] for r in rows]
 out=dict(rows=rows,summary=dict(
  gap1_learned_minus_random=dict(mean=float(np.mean(ex1)),beat_random=int(sum(x<0 for x in ex1))),
  gap0_learned_minus_random=dict(mean=float(np.mean(ex0)),beat_random=int(sum(x<0 for x in ex0))),
  mean_nll=dict(gap1_learned=float(np.mean([r['gap1']['learned'] for r in rows])),gap0_learned=float(np.mean([r['gap0']['learned'] for r in rows])),
                gap1_random=float(np.mean([r['gap1']['random'] for r in rows])),gap0_random=float(np.mean([r['gap0']['random'] for r in rows])),
                full=float(np.mean([r['gap1']['full'] for r in rows])),gap1_empty=float(np.mean([r['gap1']['empty'] for r in rows])),gap0_empty=float(np.mean([r['gap0']['empty'] for r in rows]))),
  between_chunk_overlap=dict(observed=float(np.mean([r['overlap'] for r in rows])),expected_random=float(np.mean([r['overlap_expected_random'] for r in rows])),jaccard=float(np.mean([r['jaccard'] for r in rows])))))
 feats={}
 y=[T[s]['Y'] for s in sids]
 for m in ['snp_gap1','snp_sparse','ta']:
  fs=[F.mask_features(s,'rr_off',m) for s in sids]
  feats[m]={f:dict(auroc=F.auroc(y,[x[f] for x in fs]),ci95=F.boot_ci(y,[x[f] for x in fs],[str(v) for v in y])) for f in ['cue_read_enrichment','late_cue_read_enrichment','cue_read_score_rank']}
 out['features_professor']=feats
 (D.NEW/'gap1_compare.json').write_text(json.dumps(out,indent=1))
 print(json.dumps(out['summary'],indent=1));print(json.dumps({m:{f:round(v['auroc'],3) for f,v in d.items()} for m,d in feats.items()},indent=1))
if __name__=='__main__':main()
