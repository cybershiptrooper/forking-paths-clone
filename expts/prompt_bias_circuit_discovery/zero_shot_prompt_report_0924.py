"""Render the fixed zero-shot comparison after both API-only studies finish."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('results/prompt_bias_v2/zero_shot_testbed_0924')
OWN=Path('results/prompt_bias_v2/zero_shot_own_prompts_0924')
REPORT=Path('notes/reports_monitorability/zero_shot_prompt_comparison_0924.md')
def report():
 if not all((r/'results.json').exists() for r in [BASE,OWN]):
  print('Both complete result files are required; no partial-subset ranking.',flush=True);return
 data=json.loads((BASE/'results.json').read_text())+json.loads((OWN/'results.json').read_text());records={r['source_file']:r for r in json.loads((BASE/'records.json').read_text())};order=['binary','confidence','reliance','comparative'];labels=['Simple\nbinary','Simple\nconfidence','Reliance','Comparative'];summaries=[]
 rng=np.random.default_rng(20260924)
 for task in [4,5]:
  b=next(r for r in data if r['task']==task and r['kind']=='confidence');files=b['source_files'];rr=[records[f] for f in files];y=np.array([r['label'] for r in rr]);groups=sorted({r['base_id'] for r in rr});indices=np.array([groups.index(r['base_id']) for r in rr]);w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
  def boots(row):
   assert row['source_files']==files
   pred=np.array(row['scores'])>=row['threshold'];stats=np.stack([np.bincount(indices,weights=a,minlength=len(groups)) for a in [(y==1)&pred,(y==1),(y==0)&(~pred),(y==0)]],axis=1);s=w@stats;valid=(s[:,1]>0)&(s[:,3]>0);out=np.full(len(w),np.nan);out[valid]=s[valid,0]/s[valid,1]*s[valid,2]/s[valid,3];return out
  bb=boots(b)
  for kind in order:
   row=next(r for r in data if r['task']==task and r['kind']==kind);bs=boots(row);delta=bs-bb
   summaries.append(dict(task=task,kind=kind,n=row['n'],base_questions=row['base_questions'],threshold=row['threshold'],gmean2=row['gmean2'],g_ci=np.nanquantile(bs,[.025,.975]).tolist(),auroc=None if kind=='binary' else row['auroc'],delta_vs_simple_confidence=row['gmean2']-b['gmean2'],delta_ci=np.nanquantile(delta,[.025,.975]).tolist(),tpr=row['tpr'],tnr=row['tnr'],gmean2_mc_pooled=row.get('gmean2_mc_pooled')))
 (OWN/'paired_comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
 plt.rcParams.update({'font.size':14,'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14});fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
 for ax,task,title in zip(axes,[4,5],['Scruples ID','Daily Dilemmas ID']):
  z=[next(r for r in summaries if r['task']==task and r['kind']==k) for k in order];vals=np.array([r['gmean2'] for r in z]);ax.bar(range(4),vals,color=['#999999','#0072B2','#009E73','#E69F00']);ax.vlines(range(4),[r['g_ci'][0] for r in z],[r['g_ci'][1] for r in z],color='black');ax.set_xticks(range(4),labels);ax.set_title(title);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for x,v in enumerate(vals):ax.text(x,v+.025,f'{v:.3f}',ha='center',fontsize=13)
 axes[0].set_ylabel('ID test g-mean²');fig.tight_layout();im=Path('notes/images/zero_shot_prompt_comparison_0924');im.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(im/f'comparison.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
 lines=['# Zero-shot judge prompts compared on the full released ID data','', 'The user requested a replication of the testbed zero-shot judge baseline and a comparison with our existing prompts. GPU training remains stopped. This API-only experiment uses Gemini 3.8 Flash for every condition, the exact same complete released reasoning text, and no few-shot examples. The simple prompt is our reconstruction of the published procedure, not an unpublished author prompt. The reliance and comparative prompts reuse our earlier reasoning instructions, removing graph-specific text and using the same 0–100 score schema. No prompt was changed using results from this comparison.','', '## Experiment','', 'Validation comprises 1,408 Scruples and 930 Daily Dilemmas rollouts. ID testing comprises 1,937 Scruples and 1,218 Daily Dilemmas rollouts. All released examples are retained, including negatives that disagree with the hint; there is no answer-agreement or trace-length filter. Each request contains only a task description and raw CoT, without the original prompt, final-answer field, chunk IDs, attention graphs, labels, or resampling rates. The binary baseline returns a class directly; confidence conditions return scores from 0 to 100.','', 'The judge is `google/gemini-3.8-flash`, temperature 0, top-p 1, maximum completion length 16,384 tokens. The first successful semantic response is cached; malformed semantic outputs are retained as failures rather than resampled. All responses must succeed before aggregation.','',r'Primary performance is $G=\mathrm{TPR}\times\mathrm{TNR}$ on released proxy labels. Each confidence condition selects a threshold independently on its released validation set from $\{0.01,\ldots,0.99\}$ to maximize G; ties prefer the value closest to 0.5, then the larger value. Thresholds are frozen before that condition is tested. The binary condition has no fitted threshold. Test results do not select or revise prompts.','', '## Results','', '![Zero-shot prompt comparison](../images/zero_shot_prompt_comparison_0924/comparison.png)','', '**Figure 1.** Unadjusted g-mean² on all released ID-test rollouts. Vertical lines show percentile intervals from 2,000 resamples of base questions, keeping rollouts from the same base question together. Intervals condition on the fitted validation thresholds; they do not include threshold-estimation or judge-resampling uncertainty.','']
 for task,name in [(4,'Scruples'),(5,'Daily Dilemmas')]:
  lines += [f'**{name}:**','']
  for r in [z for z in summaries if z['task']==task]:
   lines.append(f"- {r['kind']}: G={r['gmean2']:.3f}; TPR={r['tpr']:.3f}; TNR={r['tnr']:.3f}; threshold={r['threshold']:.2f}. Difference from simple confidence: {r['delta_vs_simple_confidence']:+.3f}, paired 95% cluster interval [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}].")
  lines += ['']
 lines += ['## Interpretation and replication limits','', 'Compare the paired differences, not the most favorable single prompt or a threshold chosen on test. These are only ID results and use released susceptibility proxy labels, not verified per-trace causal reliance. Multiple prompt comparisons remain exploratory. The test split includes questions reserved by the stopped mask study; future tuning informed by these results must not call those questions unseen.','', 'The blogpost used GPT-5.2 with reasoning, whereas this experiment uses the allowed Flash judge. Exact author prompts and evaluator code were not released. The post also reports an adjusted intervention score. Task 4 lacks the complete control/intervention probabilities needed to reconstruct it. Task 5 saves a pooled minimal-criterion sensitivity score, using the mean positive attributable fraction; this is an explicit aggregation assumption, not a verified reproduction of the authors’ score. The primary chart therefore uses ordinary G consistently. The separate result JSON includes both fixed-0.5 and validation-calibrated scores and the task-5 sensitivity estimates.','', '## Reproduction','', '- Runner: `expts/prompt_bias_circuit_discovery/zero_shot_testbed_0924.py` for the simple baseline; `zero_shot_own_prompts_0924.py` in the same directory for our two prompts.','- Data source: `data/external/cot-proxy-tasks`, revision `4482324b5e4a6277fa3bd544785cbd9875e11694`.','- Exact frozen prompts, records, request mappings, responses, thresholds, and numeric results: `results/prompt_bias_v2/zero_shot_testbed_0924/` and `results/prompt_bias_v2/zero_shot_own_prompts_0924/`.','- Paired differences and intervals: `zero_shot_own_prompts_0924/paired_comparison.json` under the results root.','- Regenerate this report without API calls or GPUs: `uv run python -m expts.prompt_bias_circuit_discovery.zero_shot_prompt_report_0924`.','']
 REPORT.write_text('\n'.join(lines));print('Comparison report complete',flush=True)
if __name__=='__main__':report()
