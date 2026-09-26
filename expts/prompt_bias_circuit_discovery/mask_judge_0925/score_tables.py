from expts.prompt_bias_circuit_discovery.mask_judge_0925 import analyze as A, judge_ablation as J, data as D
from expts.prompt_bias_circuit_discovery.monitor_tuning_0924 import metrics as TM
import numpy as np, json
T=D.traces();cfgs=list(J.CONFIGS)
dev,_=A.scores('development',cfgs);cal,_=A.scores('calibration',['legacy','base']);ext,_=A.scores('extension',['legacy','base'])
recs=[r for r in json.loads((D.TUN/'cohort.json').read_text())['records'] if r['tuning_role']=='development'];fm=TM.folds(recs)
def g_cv(sids,sc):
    y=np.array([T[s]['Y'] for s in sids]);s=np.array([sc[x] for x in sids]);ff=np.array([fm[x] for x in sids]);cv=np.zeros(len(sids))
    for f in range(6):
        tr=ff!=f;cv[~tr]=s[~tr]>=TM.threshold(y[tr],s[tr])
    return TM.gmean(y,cv,.5)
conds={'text':['text'],'ta':['ta'],'snp':['snp_sparse'],'random':['random0','random1']}
out={}
for cfg in cfgs:
    present={k:cs for k,cs in conds.items() if all(len(dev.get((cfg,c),{}))>=40 for c in cs)}
    common=sorted(set.intersection(*[set(dev.get((cfg,c),{})) for cs in present.values() for c in cs]))
    row={}
    for k,cs in conds.items():
        if k not in present:row[k]=None;continue
        per={}
        for t in (4,5):
            ss=[s for s in common if T[s]['task']==t]
            au=np.mean([A.auroc([T[s]['Y'] for s in ss],[dev[(cfg,c)][s] for s in ss]) for c in cs])
            g=np.mean([g_cv(ss,dev[(cfg,c)]) for c in cs])
            per[t]=(au,g)
        row[k]=per
    out[cfg]=(row,len(common))
L=A.CONFIG_LABEL
print('AUROC, development (Scruples / professor)')
print('| configuration | traces | text only | Thought Anchors | SNP, revised optimizer | random mask |');print('|---|---|---|---|---|---|')
for cfg in cfgs:
    row,n=out[cfg];print(f"| {L[cfg]} | {n} | "+' | '.join('—' if row[k] is None else f"{row[k][4][0]:.2f} / {row[k][5][0]:.2f}" for k in conds)+' |')
print('\ng-mean squared, development, six-fold CV threshold (Scruples / professor)')
print('| configuration | text only | Thought Anchors | SNP, revised optimizer | random mask |');print('|---|---|---|---|---|')
for cfg in cfgs:
    row,n=out[cfg];print(f"| {L[cfg]} | "+' | '.join('—' if row[k] is None else f"{row[k][4][1]:.2f} / {row[k][5][1]:.2f}" for k in conds)+' |')
# held-out calibration: thresholds fitted on all development traces of the same config/condition/task
for HNAME,cal in [('held-out calibration',cal),('extension',ext)]:
 print(f'\n{HNAME}: AUROC and g-mean squared with the development threshold (Scruples / professor)')
 print('| configuration | text only | Thought Anchors | random mask |');print('|---|---|---|---|')
 for cfg in ['legacy','base']:
     common=sorted(set.intersection(*[set(cal.get((cfg,c),{})) for c in ['text','ta','random0','random1']]))
     cells=[]
     for k,cs in [('text',['text']),('ta',['ta']),('random',['random0','random1'])]:
         per=[]
         for t in (4,5):
             ss=[s for s in common if T[s]['task']==t];y=np.array([T[s]['Y'] for s in ss]);aus=[];gs=[]
             for c in cs:
                 dv=[s for s in dev[(cfg,c)] if T[s]['task']==t];th=TM.threshold(np.array([T[s]['Y'] for s in dv]),np.array([dev[(cfg,c)][s] for s in dv]))
                 sc=np.array([cal[(cfg,c)][s] for s in ss]);aus.append(A.auroc(y,sc));gs.append(TM.gmean(y,sc,th))
             per.append((np.mean(aus),np.mean(gs)))
         cells.append(f"AUROC {per[0][0]:.2f} / {per[1][0]:.2f}, G {per[0][1]:.2f} / {per[1][1]:.2f}")
     print(f"| {L[cfg]} ({len(common)}) | "+' | '.join(cells)+' |')

# sentence gap 1 versus gap 0, professor development traces only
print('\nprofessor development traces: AUROC and g-mean squared (six-fold CV threshold), sentence gap 1 versus gap 0')
print('| configuration | traces | text only | SNP gap 0 | SNP gap 1 | Thought Anchors | random mask |');print('|---|---|---|---|---|---|---|')
gconds={'text':['text'],'g0':['snp_sparse'],'g1':['snp_gap1'],'ta':['ta'],'random':['random0','random1']}
for cfg in cfgs:
    if len(dev.get((cfg,'snp_gap1'),{}))<20:continue
    common=sorted(s for s in set.intersection(*[set(dev.get((cfg,c),{})) for cs in gconds.values() for c in cs]) if T[s]['task']==5)
    cells=[]
    for k,cs in gconds.items():
        au=np.mean([A.auroc([T[s]['Y'] for s in common],[dev[(cfg,c)][s] for s in common]) for c in cs]);g=np.mean([g_cv(common,dev[(cfg,c)]) for c in cs])
        cells.append(f"{au:.2f} (G {g:.2f})")
    print(f"| {L[cfg]} | {len(common)} | "+' | '.join(cells)+' |')
