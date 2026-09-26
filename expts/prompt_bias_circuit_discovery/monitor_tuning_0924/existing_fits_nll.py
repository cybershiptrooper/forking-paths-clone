"""Reasoning NLL of every completed R->R mask versus its own five random masks. No GPU, no API."""
import glob,json,statistics as st,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results/prompt_bias_v2');OUT=Path('notes/images/monitor_tuning_0924')
def label(recipe):
 opt='sparse10' if 'peak10_sparse' in recipe else 'dense10' if 'peak10_dense' in recipe else 'dense100' if 'peak100' in recipe else 'original'
 return opt,('keep50' if 'keep50' in recipe else 'keep20')
def rows():
 out=[]
 def add(ns,sid,recipe,mask,controls):
  m=json.loads(Path(mask).read_text());c=json.loads(Path(controls).read_text())
  if not m.get('final'):return
  cond=recipe.split('_keep')[0]
  rk=[k for k in c if k.startswith('random')] or [k for k in c if k.startswith(cond+'/random')]
  out.append(dict(ns=ns,sid=sid,recipe=recipe,cond=cond,nll=m['diagnostics']['trace_nll'],random=st.mean(c[k]['trace_nll'] for k in rk)))
 for c in glob.glob(str(R/'monitor_tuning_0924/fits/*/*/complete.json')):
  sid,rec=c.split('/')[-3],c.split('/')[-2];cond=rec.split('_keep')[0]
  add('tun',sid,rec,f'{R}/monitor_tuning_0924/fits/{sid}/{rec}/{cond}_mask.json',f'{R}/monitor_tuning_0924/fits/{sid}/{rec}/controls.json')
 for c in glob.glob(str(R/'monitor_expanded_0924/fits/*/rr_*/complete.json')):
  sid,cond=c.split('/')[-3],c.split('/')[-2]
  add('exp',sid,cond+'_keep20',f'{R}/monitor_expanded_0924/fits/{sid}/{cond}/{cond}_mask.json',f'{R}/monitor_expanded_0924/fits/{sid}/{cond}/controls.json')
 for sid in ['D01','D02','D03','D04']:
  for cond in ['rr_on','rr_off']:add('pil',sid,cond+'_keep20',f'{R}/monitor_pilot_0924/{sid}/{cond}_mask.json',f'{R}/monitor_pilot_0924/{sid}/controls.json')
 for r in out:r['excess']=r['nll']-r['random']
 return out
def main():
 data=rows();OUT.mkdir(parents=True,exist_ok=True)
 groups=[('original','keep20'),('sparse10','keep20'),('dense10','keep20'),('dense100','keep20'),('original','keep50'),('sparse10','keep50')]
 names={'original':'original','sparse10':'sparse init\npen. 10','dense10':'dense init\npen. 10','dense100':'dense init\npen. 100'}
 plt.rcParams.update({'font.size':12,'axes.labelsize':13})
 fig,axes=plt.subplots(1,3,figsize=(17,5.2),gridspec_kw=dict(width_ratios=[1.4,1.4,1]))
 for ax,cond,title in zip(axes[:2],['rr_on','rr_off'],['outside attention on','outside attention off']):
  for i,(opt,keep) in enumerate(groups):
   v=[r['excess'] for r in data if r['cond']==cond and label(r['recipe'])==(opt,keep)]
   if not v:continue
   x=[i+0.12*((j%5)-2)/2 for j in range(len(v))]
   ax.scatter(x,v,s=18,color='#0072B2' if keep=='keep20' else '#E69F00',alpha=.7,zorder=3)
   ax.plot([i-.3,i+.3],[st.mean(v)]*2,color='k',lw=2,zorder=4);ax.text(i,max(v)+.25,f'n={len(v)}',ha='center',fontsize=10)
  ax.axhline(0,ls='--',color='#777');ax.set_xticks(range(len(groups)),[f"{names[o]}\n{'80%' if k=='keep20' else '50%'} sp." for o,k in groups],fontsize=10)
  ax.set_title(f'R->R masks, {title}');ax.set_ylabel('learned minus random-mask NLL (nats/token)');ax.set_ylim(-7,3)
  for s in ax.spines.values():s.set_color('k');s.set_linewidth(1.2)
 ax=axes[2];pair=collections.defaultdict(dict)
 for r in data:
  o,k=label(r['recipe'])
  if k=='keep20' and o in('original','sparse10'):pair[(r['sid'],r['cond'])][o]=r['excess']
 for key,p in sorted(pair.items()):
  if len(p)==2:ax.plot([0,1],[p['original'],p['sparse10']],marker='o',color='#0072B2' if key[1]=='rr_on' else '#D55E00',alpha=.6)
 ax.axhline(0,ls='--',color='#777');ax.set_xticks([0,1],['original','sparse init\npenalty 10']);ax.set_xlim(-.4,1.4);ax.set_ylim(-7,3)
 n=sum(len(p)==2 for p in pair.values());ax.set_title(f'paired, 80% sparsity ({n} trace/background pairs)');ax.plot([],[],color='#0072B2',label='outside on');ax.plot([],[],color='#D55E00',label='outside off');ax.legend(fontsize=10)
 for s in ax.spines.values():s.set_color('k');s.set_linewidth(1.2)
 fig.tight_layout()
 for ext in['png','pdf']:fig.savefig(OUT/f'existing_fits_excess_nll.{ext}',dpi=170,bbox_inches='tight')
 summary=collections.defaultdict(list)
 for r in data:summary[(r['cond'],)+label(r['recipe'])].append(r['excess'])
 Path(R/'monitor_tuning_0924/existing_fits_excess_nll.json').write_text(json.dumps(dict(rows=data,summary={'/'.join(k):dict(n=len(v),mean=st.mean(v),beat_random=sum(x<0 for x in v)) for k,v in summary.items()}),indent=1))
 print(json.dumps({'/'.join(k):dict(n=len(v),mean=round(st.mean(v),2),beat_random=sum(x<0 for x in v)) for k,v in sorted(summary.items())},indent=1))
if __name__=='__main__':main()
