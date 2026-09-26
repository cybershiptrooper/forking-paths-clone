"""Report paired interim judge comparisons on frozen development packets."""
import json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cohort import ROOT
from .metrics import threshold,gmean,auc
from ..monitor_pilot_0924.run import write_json
OUT=ROOT/'interim_variant_comparison'
METHODS=['snp','snp_peak10_sparse','ta','random']
LABELS=['Original SNP','Revised SNP','Thought Anchors','Random']
CONFIGS=['rr_on/keep50','rr_off/keep50','rr_on/keep20','rr_off/keep20']
def corr(a,b):
 if np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def ranks(a):return [sum(v<x for v in a)+(sum(v==x for v in a)-1)/2 for x in a]
def report():
 protocol=json.loads((OUT/'protocol.json').read_text());prompts=list(protocol['prompts']);mapping=json.loads((OUT/'mapping.json').read_text());scores={}
 for key,sha in mapping.items():
  d=json.loads((ROOT/'judge/max16384'/f'{sha}.json').read_text());assert d['status']=='ok';scores[key]=d['parsed']['probability']
 cohort={r['sample_id']:r for r in json.loads((ROOT/'cohort.json').read_text())['records']}
 required=['text']+[m+'/'+c for m in METHODS for c in CONFIGS]
 ids=[sid for sid in protocol['sample_ids'] if all(sid+'|'+p+'/cot_only/'+c in scores for p in prompts for c in required)]
 assert ids
 baseline=json.loads((ROOT/'development_metrics.json').read_text())['results'];thresholds={}
 for task in [4,5]:
  yy=[cohort[s]['Y'] for s in ids if cohort[s]['task']==task];assert set(yy)=={0,1},('Insufficient labels',task,yy)
  for p in prompts:
   b=next(r for r in baseline if r['task']==task and r['condition_id']==p+'/cot_only/text')
   train=[i for i,s in enumerate(b['sample_ids']) if s not in protocol['sample_ids']];assert len(train)==20
   thresholds[task,p]=float(threshold(np.array(b['labels'])[train],np.array(b['probabilities'])[train]))
 rows=[]; correlations=[]
 for p in prompts:
  for c in required:
   for task in [4,5]:
    ss=[s for s in ids if cohort[s]['task']==task];y=[cohort[s]['Y'] for s in ss];v=[scores[s+'|'+p+'/cot_only/'+c] for s in ss];t=thresholds[task,p]
    rows.append(dict(prompt=p,variant=c,task=task,n=len(ss),sample_ids=ss,labels=y,probabilities=v,threshold=t,gmean2=gmean(y,v,t),gmean2_at_half=gmean(y,v,.5),auroc=auc(y,v)))
   text=[scores[s+'|'+p+'/cot_only/text'] for s in ids];v=[scores[s+'|'+p+'/cot_only/'+c] for s in ids];y=[cohort[s]['Y'] for s in ids]
   correlations.append(dict(prompt=p,variant=c,n=len(ids),pearson_with_text=corr(v,text),spearman_with_text=corr(ranks(v),ranks(text)),mean_absolute_score_change=float(np.mean(np.abs(np.array(v)-text))),mean_score_change_positive=statistics.mean(a-b for s,a,b in zip(ids,v,text) if cohort[s]['Y']==1),mean_score_change_negative=statistics.mean(a-b for s,a,b in zip(ids,v,text) if cohort[s]['Y']==0)))
 out=dict(sample_ids=ids,excluded_missing=[s for s in protocol['sample_ids'] if s not in ids],results=rows,correlations=correlations,thresholds={f'{t}/{p}':v for (t,p),v in thresholds.items()});write_json(OUT/'results.json',out)
 def macro(p,c,k):return statistics.mean(r[k] for r in rows if r['prompt']==p and r['variant']==c)
 plt.rcParams.update({'font.size':14,'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14})
 image_dir=Path('notes/images/monitor_variant_comparison_0924');image_dir.mkdir(exist_ok=True,parents=True)
 colors=['#999999','#0072B2','#009E73','#E69F00'];fig,axes=plt.subplots(2,3,figsize=(17,10),sharey=True)
 for j,p in enumerate(prompts):
  for ax,k in zip(axes[:,j],['gmean2','auroc']):
   for i,m in enumerate(METHODS):ax.bar(np.arange(4)+(i-1.5)*.19,[macro(p,m+'/'+c,k) for c in CONFIGS],width=.18,color=colors[i],label=LABELS[i])
   ax.axhline(macro(p,'text',k),color='black',linestyle='--',label='Text only');ax.set_xticks(range(4),['50%\non','50%\noff','80%\non','80%\noff']);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  axes[0,j].set_title(p.capitalize());axes[1,j].set_xlabel('Sparsity / outside-mask attention')
 axes[0,0].set_ylabel('Mean task g-mean²');axes[1,0].set_ylabel('Mean task AUROC');handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,fontsize=14);fig.tight_layout(rect=(0,0,1,.94))
 for ext in ['png','pdf']:fig.savefig(image_dir/f'judge_variants.{ext}',dpi=150,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
 for ax,p in zip(axes,prompts):
  for i,m in enumerate(METHODS):
   z=[next(r for r in correlations if r['prompt']==p and r['variant']==m+'/'+c)['pearson_with_text'] for c in CONFIGS];ax.bar(np.arange(4)+(i-1.5)*.19,[v if v is not None else np.nan for v in z],width=.18,color=colors[i],label=LABELS[i])
  ax.set_title(p.capitalize());ax.set_xticks(range(4),['50%\non','50%\noff','80%\non','80%\noff']);ax.set_ylim(-1,1.05);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_xlabel('Sparsity / outside-mask attention')
 axes[0].set_ylabel('Pearson correlation with text-only score');fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,fontsize=14);fig.tight_layout(rect=(0,0,1,.89))
 for ext in ['png','pdf']:fig.savefig(image_dir/f'text_correlation.{ext}',dpi=150,bbox_inches='tight')
 plt.close(fig)
 lines=['# Completed development masks provide an interim judge comparison','', 'This experiment compares 50% and 80% sparsity and outside-mask attention on/off to decide whether the larger training grid can be reduced. It uses the original eight development traces, never the calibration or protected ID/OOD test sets. Requests were frozen before this comparison. It evaluates original SNP, revised SNP with sparse initialization and penalty coefficient 10, Thought Anchors, and one matched random mask per condition, using all three existing judge prompts.','', '## Experiment','',f'The common complete set contains **{len(ids)} traces**: '+', '.join(ids)+'. Missing-condition exclusions: '+(', '.join(out['excluded_missing']) or 'none')+'. The same traces enter every comparison. Completion-based exclusions can bias this snapshot; these traces were also previously used for development and are not fresh validation.','', 'Task 4 is the Scruples user-preference cue task; task 5 is the Daily Dilemmas professor-hint task. All samples have an intervention and agree with the suggested answer. Labels are the released susceptibility proxies, not verified trace-level causal reliance. Each judge sees the full stored CoT, a task description, and optionally the graph; it does not see the original prompt, final answer, or label. R→R masks supply reasoning-node connections. On/off describes mutable attention outside that region, not whether the judge sees text.','',r'For each task, $G=\mathrm{TPR}\times\mathrm{TNR}$. Each prompt uses a text-only threshold selected on the other 20 development traces of the same task, excluding all eight designated comparison traces. The threshold then stays fixed across mask variants. This separates threshold fitting from the small comparison, but does not calibrate each graph variant separately. AUROC measures positive-versus-negative score ordering, with ties worth one half. Plots average the two task metrics equally; they do not pool unlike tasks. Raw threshold-0.5 results also appear in the JSON.','', '## Results','', '![Judge variant comparison](../images/monitor_variant_comparison_0924/judge_variants.png)','',f'**Figure 1.** Judge performance on the same {len(ids)} development traces for every bar. Columns use the three frozen judge prompts. Top: g-mean² with independently fitted text-only thresholds. Bottom: AUROC. The dashed line is text-only. The four x-axis groups cross sparsity with outside-mask attention; colors compare graph methods. Small sample counts make these descriptive estimates unsuitable for a conclusive ranking.','']
 for p in prompts:
  vals=[(c,macro(p,'snp_peak10_sparse/'+c,'gmean2'),macro(p,'snp_peak10_sparse/'+c,'auroc')) for c in CONFIGS]
  lines += [f"- **{p}:** text-only G={macro(p,'text','gmean2'):.3f}, AUC={macro(p,'text','auroc'):.3f}. Revised SNP (G, AUC): "+'; '.join(f'{c}: ({g:.3f}, {a:.3f})' for c,g,a in vals)+'.']
 lines += ['', '![Text score correlation](../images/monitor_variant_comparison_0924/text_correlation.png)','',f'**Figure 2.** Pearson correlation between graph-assisted and text-only judge scores on the same {len(ids)} traces, across both tasks. This measures similarity to text-only judgments, not correctness or causal faithfulness. Task-level differences can contribute to pooled correlation. Spearman correlation, absolute score changes, and mean changes by label are recorded in the result JSON.','', '## Takeaways','', 'The completed snapshot contains 402 valid request-condition responses and seven matched traces: three Scruples and four Daily Dilemmas. Revised SNP at 80% sparsity with outside attention off raises mean task G from 0.50 to 0.75 under the comparative prompt; Thought Anchors matches it. That gain comes from one corrected decision in the three-trace Scruples subset. The other two prompts show no gain for that variant. Revised SNP at 50% sparsity with outside attention off instead lowers mean task G from 0.50 to 0.25 under the reliance prompt. Original SNP at 50% sparsity with outside attention on also reaches 0.75 under the comparative prompt. These results do not support discarding the original optimizer or declaring a winning sparsity/background based on reasoning likelihood alone. Revised-SNP score correlations with text-only range from 0.909 to 0.999 across the twelve prompt/configuration combinations. Correlation includes task differences and does not prove that graphs carry no additional information.', '', 'This interim comparison cannot establish a robust winning sparsity or outside-mask setting: each task has only a few traces, graph-specific calibration has not occurred, and the examples have already supported development decisions. High text correlation is compatible with useful threshold crossings; low correlation is compatible with harm. A larger fixed, balanced development comparison is needed before performance-based pruning. No training jobs or frozen held-out recipes are changed by this script.','', '## Reproduction','', '- Run `uv run python -m expts.prompt_bias_circuit_discovery.monitor_tuning_0924.compare_ready` from the repository root. Existing exact-request responses are reused; the frozen request set does not grow on rerun.','- Code: `expts/prompt_bias_circuit_discovery/monitor_tuning_0924/compare_ready.py` and `compare_ready_report.py`.','- Frozen packets, prompts, model, sample IDs, mappings, completion status, and numeric results: `results/prompt_bias_v2/monitor_tuning_0924/interim_variant_comparison/{requests,protocol,mapping,completion,results}.json`.','- Raw judge cache: `results/prompt_bias_v2/monitor_tuning_0924/judge/max16384/`.','- Source labels and records: `results/prompt_bias_v2/monitor_tuning_0924/cohort.json`; frozen inputs in `inputs/`; masks in `fits/`, with original-mask/TA fallbacks resolved by `judge.source_path`.','- Judge: `google/gemini-3.8-flash`, temperature 0, top-p 1, maximum 16,384 completion tokens. Full system prompts and exact visible packets are stored in the frozen protocol and requests files.','']
 Path('notes/reports_monitorability/judge_variant_comparison_0924.md').write_text('\n'.join(lines));print('REPORT COMPLETE',len(ids),'matched traces',flush=True)
if __name__=='__main__':report()
