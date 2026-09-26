"""Descriptive report of every prespecified case; no example/setting selection."""
from pathlib import Path
import json
from statistics import mean
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/prompt_bias_v2/rethink0924/mask_feasibility')
HINT=Path('results/prompt_bias_v2/rethink0924/hint_positive_control')
METHODS=['clean','empty','random','thought_anchors','increase','decrease']
LABELS={'clean':'Full pool','empty':'Empty pool','random':'Random 20%','thought_anchors':'Thought Anchors','increase':'Own-answer SNP','decrease':'Opposite-answer SNP'}

def method_values(pool,suffix,arm,method):
    entries=pool['evaluation'][suffix]['arms'][arm]
    if method=='random':
        random=[entries[k] for k in entries if k.startswith('random_')]
        return dict(conditional=[mean(x['conditional'][j] for x in random) for j in range(2)],
                    answer_mass=mean(x['answer_mass'] for x in random),raw=[mean(x['raw'][j] for x in random) for j in range(2)])
    return entries[method]

def auc(labels,values):
    pos=[v for y,v in zip(labels,values) if y];neg=[v for y,v in zip(labels,values) if not y]
    return mean(float(p>n)+.5*float(p==n) for p in pos for n in neg)

def feature(pool,method,name):
    if method=='random':
        x=[pool['features'][k][name]['enrichment'] for k in pool['features'] if k.startswith('random_')]
        return mean(v for v in x if v is not None) if any(v is not None for v in x) else None
    return pool['features'][method][name]['enrichment']

def main():
    paths=[ROOT/'eval'/f'ex{i:03d}.json' for i in range(0,16,2)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise SystemExit('Report requires every fixed admission case; missing: '+', '.join(missing))
    rows=[json.loads(p.read_text()) for p in paths]
    summary=dict(n_inputs=8,labels=[r['is_positive'] for r in rows],uids=[r['uid'] for r in rows],input_details=[{k:r[k] for k in ['example_id','uid','gender','prefix_tokens','n_sentences','n_prompt','reasoning_name_mentions','source_rates']} for r in rows],pools={},hint=None)
    for pool_name in ['p2t','t2t']:
        ps=[r['pools'][pool_name] for r in rows]
        pooled={}
        for suffix in ['original','answer_colon']:
            pooled[suffix]={}
            for method in METHODS:
                vals=[method_values(p,suffix,'black',method) for p in ps]
                pooled[suffix][method]=dict(mean_conditional_A=mean(v['conditional'][0] for v in vals),
                    mean_raw_A=mean(v['raw'][0] for v in vals),mean_answer_mass=mean(v['answer_mass'] for v in vals),
                    per_input_conditional_A=[v['conditional'][0] for v in vals],per_input_answer_mass=[v['answer_mass'] for v in vals])
        cue={method:[feature(p,method,'name_prompt_key') for p in ps] for method in ['random','thought_anchors','increase','decrease']} if pool_name=='p2t' else {}
        aucs={m:auc(summary['labels'],v) for m,v in cue.items() if all(x is not None for x in v)}
        random_aucs=[auc(summary['labels'],[feature(p,f'random_{seed:02d}','name_prompt_key') for p in ps]) for seed in range(10)] if pool_name=='p2t' else []
        training=[p['training_and_threshold']['increase']['last_training_metrics'] for p in ps]
        summary['pools'][pool_name]=dict(readout=pooled,cue_enrichment=cue,cue_enrichment_auc=aucs,random_cue_auc_by_seed=random_aucs,
            threshold_ties={m:[p['training_and_threshold'][m]['n_threshold_ties'] for p in ps] for m in ['increase','decrease','thought_anchors']},
            actual_removal=[p['actual_removal'] for p in ps],last_training_metrics=training,
            overlap_with_increase=[p['overlap_with_increase'] for p in ps],
            replay_contrast={suffix:{method:[p['evaluation'][suffix]['replay_contrasts'][method] for p in ps] for method in ['clean','empty','thought_anchors','increase','decrease']} for suffix in ['original','answer_colon']})
    if (HINT/'eval.json').exists():summary['hint']=json.loads((HINT/'eval.json').read_text())
    accounting=ROOT/'allocation_accounting.json'
    if accounting.exists():summary['allocation']=json.loads(accounting.read_text())
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14})
    fig,axes=plt.subplots(3,2,figsize=(19,17),layout='constrained')
    colors=['#0072B2' if r['is_positive'] else '#D55E00' for r in rows]
    offsets=np.linspace(-.18,.18,8)
    for j,pool_name in enumerate(['p2t','t2t']):
        pools=[r['pools'][pool_name] for r in rows]
        for ri,pool in enumerate(pools):
            x=np.arange(len(METHODS))+offsets[ri]
            vals=[method_values(pool,'original','black',m) for m in METHODS]
            axes[0,j].plot(x,[v['conditional'][0] for v in vals],'-o',color=colors[ri],alpha=.65,markersize=6,lw=.8)
            axes[1,j].plot(x,[v['answer_mass'] for v in vals],'-o',color=colors[ri],alpha=.65,markersize=6,lw=.8)
        for ax in axes[:2,j]:
            ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=28,ha='right')
        axes[0,j].set_title('Prompt → reasoning' if pool_name=='p2t' else 'Earlier reasoning → reasoning')
        axes[0,j].set_ylabel('Conditional P(A | A or B)')
        axes[0,j].set_ylim(-.02,1.03)
        axes[1,j].set_ylabel('Raw P(A) + P(B)')
        axes[1,j].set_ylim(bottom=0)
        feat='name_prompt_key' if pool_name=='p2t' else 'late_reasoning_key'
        methods=['random','thought_anchors','increase','decrease']
        for ri,pool in enumerate(pools):
            axes[2,j].plot(np.arange(4)+offsets[ri],[feature(pool,m,feat) for m in methods],'-o',color=colors[ri],alpha=.65,markersize=6,lw=.8)
        axes[2,j].set_xticks(range(4),[LABELS[m] for m in methods],rotation=28,ha='right')
        axes[2,j].axhline(1,color='black',ls='--',lw=1)
        axes[2,j].set_ylabel('Name-key retention / pool keep rate' if pool_name=='p2t' else 'Last-quarter key retention / pool keep rate')
        axes[2,j].set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    axes[0,0].legend(handles=[Line2D([0],[0],marker='o',color='#0072B2',label='Effect-input proxy'),Line2D([0],[0],marker='o',color='#D55E00',label='Small-effect proxy')],loc='lower left',bbox_to_anchor=(0,1.06),ncol=2,frameon=False)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    imdir=Path('notes/images/rethink0924');imdir.mkdir(parents=True,exist_ok=True)
    fig.savefig(imdir/'mask_feasibility.png',dpi=150,bbox_inches='tight');plt.close(fig)
    hint_text='The fixed professor-hint control has not completed. No result is inferred from its absence.'
    if summary['hint']:
        h=summary['hint'];cond=h['evaluation']['original']['conditions'];ran=[v for k,v in cond.items() if k.startswith('random_')]
        hint_text=f"The fixed professor-hint example has clean conditional target probability {cond['clean']['p_target']:.4f}, empty-pool probability {cond['empty']['p_target']:.4f}, own-answer SNP probability {cond['increase']['p_target']:.4f}, Thought Anchors probability {cond['thought_anchors']['p_target']:.4f}, and random-mask mean {mean(v['p_target'] for v in ran):.4f}. The cue-edge enrichment is {h['features']['increase']['cue_enrichment']:.3f} for own-answer SNP and {h['features']['thought_anchors']['cue_enrichment']:.3f} for Thought Anchors. Removing retained cue reads from the SNP mask gives target probability {cond['increase_without_cue_reads']['p_target']:.4f}; the change concerns only the modeled prompt-to-reasoning reads. The final-answer probe and other paths can still read the cue."
        alt=h['evaluation']['answer_colon']['conditions']
        hint_text+=f" The original clean suffix assigns total raw option mass {cond['clean']['answer_mass']:.3f}; the colon suffix assigns {alt['clean']['answer_mass']:.3f}, despite conditional target probability {alt['clean']['p_target']:.4f}. A low probability of an immediate option letter does not establish malformed generation: the model can first emit formatting or explanatory tokens. Thought Anchors preferentially retains the explicit cue on this fixed example, while own-answer SNP stays closer to uniform retention. This is a positive check of cue-sensitive ranking for Thought Anchors; it neither establishes a decision-causing path nor distinguishes proxy classes."
        fig,ax=plt.subplots(1,3,figsize=(21,5),layout='constrained')
        meth=['clean','empty','random','thought_anchors','increase'];lab=[LABELS[m] for m in meth]
        prob=[mean(v['p_target'] for v in ran) if m=='random' else cond[m]['p_target'] for m in meth]
        mass=[mean(v['answer_mass'] for v in ran) if m=='random' else cond[m]['answer_mass'] for m in meth]
        for a,vals,title in zip(ax,[prob,mass],['Conditional P(stored answer | options)','Raw probability of any option letter']):
            a.bar(lab,vals,color='#0072B2');a.set_ylim(0,1.03);a.set_title(title);a.tick_params(axis='x',rotation=25);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
        cue_meth=['random','thought_anchors','increase']
        cue_vals=[mean(v['cue_enrichment'] for k,v in h['features'].items() if k.startswith('random_')) if m=='random' else h['features'][m]['cue_enrichment'] for m in cue_meth]
        ax[2].bar([LABELS[m] for m in cue_meth],cue_vals,color='#0072B2');ax[2].set_title('Cue retention / pool keep rate');ax[2].axhline(1,color='black',ls='--',lw=1);ax[2].tick_params(axis='x',rotation=25);ax[2].grid(axis='y',alpha=.2);ax[2].set_axisbelow(True)
        fig.savefig(imdir/'hint_positive_control.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=[]
    for pool_name in ['p2t','t2t']:
        p=summary['pools'][pool_name];o=p['readout']['original'];a=p['readout']['answer_colon'];train=p['last_training_metrics']
        lines.append(f"For **{pool_name}**, mean conditional A probability is {o['clean']['mean_conditional_A']:.4f} with the full pool, {o['empty']['mean_conditional_A']:.4f} with the empty pool, {o['increase']['mean_conditional_A']:.4f} with own-answer SNP, {o['thought_anchors']['mean_conditional_A']:.4f} with Thought Anchors, and {o['random']['mean_conditional_A']:.4f} for random masks. The existing opposite-answer SNP gives {o['decrease']['mean_conditional_A']:.4f}. The own-answer SNP minus random difference is {o['increase']['mean_conditional_A']-o['random']['mean_conditional_A']:+.4f}. Under the predeclared alternative suffix, the same difference is {a['increase']['mean_conditional_A']-a['random']['mean_conditional_A']:+.4f}.")
        lines.append(f"The binary evaluation removes {min(p['actual_removal']):.4f}–{max(p['actual_removal']):.4f} of eligible edges after rounding to whole pairs. The last logged training sparsity ranges from {min(t['sparsity'] for t in train):.4f} to {max(t['sparsity'] for t in train):.4f}, and expected sparsity from {min(t['sparsity_expected'] for t in train):.4f} to {max(t['sparsity_expected'] for t in train):.4f}. Those training values concern the optimized Hard-Concrete masks; they are not the sparsity or task loss of the final rank-thresholded binary mask.")
    p2t=summary['pools']['p2t']
    auc_text='; '.join(f"{LABELS[m]}: {v:.4f}" for m,v in p2t['cue_enrichment_auc'].items())
    hint_image='\n\n![Fixed professor-hint pipeline control](../../images/rethink0924/hint_positive_control.png)\n\n*Figure 2. The first fixed professor-hint example from the historical selection, evaluated at the complete reasoning trace with the same 80-percent removal target and own-answer objective. The first two panels separate the conditional option ratio from full-vocabulary option mass. The third panel compares hint-key retention after normalizing by the pool keep rate; the dashed line is uniform retention in expectation. A single selected example checks the pipeline and does not estimate generalization.*' if summary['hint'] else ''
    report=rf'''# Whole-trace masks test answer reconstruction and cue retention on eight fixed admission inputs

We tested whether the existing fixed-sparsity mask recipe supplies more than an answer-preserving subnetwork before spending on a larger monitor dataset. The experiment fits own-answer-increasing masks using subnetwork probing (SNP), compares native Thought Anchors and existing opposite-answer masks, and measures cue retention beside answer reconstruction. Every admission example and training setting was fixed before the new results. The experiment contains eight distinct applications, so the class comparisons are descriptive feasibility checks rather than held-out monitor evaluations. A parallel provenance audit found that the historical 16-rollout screen fully overlaps positions 35–50 of the claimed 64-rollout confirmation bank. The effect/small-effect colors therefore use historical proxy labels without the claimed independent confirmation. We retain those prespecified labels and cases; we do not relabel the pilot after inspecting masks. Fresh or demonstrably nonoverlapping effect estimates are required before a larger class-discrimination evaluation. The [main report, Section 3](../principled_dataset_and_mask_plan.md#3-what-the-existing-qwen3-8b-data-support) gives the seed audit and corrected support counts.

## Experiment

The admission cohort contains the eight inputs from the earlier reduce-mask pilot: four historical effect-input proxies and four historical small-estimated-effect proxies. We use one stored A/admit trace per input, making the observed answer constant across the two proxy classes. The effects label applications; the experiment has no individual-trace causal labels. We replay complete stored reasoning from Qwen3-8B up to its closing thinking token.

We fit 16 new masks: eight traces times two separate eligible pools, prompt-to-reasoning and earlier-reasoning-to-reasoning. Every fit targets a larger conditional probability of the observed A answer, using the existing reward-gap objective, 1,000 steps, the canonical hybrid optimizer, and the fixed target-size squared penalty. The prescribed removal fraction is 0.80. We do not introduce a Lagrange multiplier, change the objective using a counterfactual, or select checkpoints or examples using new mask outcomes. Prompt-to-reasoning masks keep the existing option-key exclusions. Unassigned tokens, within-sentence reads, prompt-to-prompt reads and answer-probe reads retain their existing behavior.

The training loss uses the answer-token conditional distribution $q$ and the existing prescribed penalty schedule:

$$\mathcal{{L}}(\alpha)=\mathbb{{E}}_{{M\sim\mathrm{{HC}}(\alpha)}}\left[-q_y(M)+\max_{{a\ne y}}q_a(M)\right]+\frac{{\lambda_t}}{{n}}\left(\sum_{{e\in K}}\Pr_\alpha[M_e>0]-0.2n\right)^2,\qquad n=|K|.$$

Here $y$ is the stored answer, $K$ is the eligible pool, and HC is the Hard-Concrete mask distribution. The fixed schedule ramps $\lambda_t$ to 1,000; it is not an optimized dual variable. This objective optimizes the conditional option ratio, not raw full-vocabulary option mass. For the two-option admission task, its answer term is $1-2q_A$.

Each evaluation compares the full and empty eligible pools, the new own-answer mask, the existing opposite-answer mask, native Thought Anchors, and ten uniformly random masks with exactly the same retained-pair count. Rank thresholding removes 80 percent of the eligible pairs, up to integer rounding, regardless of the last training sparsity. Exact score ties follow the original pair order; the raw evaluations and aggregate summary record threshold tie counts so a tied ranking is not mistaken for identified importance. We report the training and thresholded evaluations separately.

Native Thought Anchors suppresses a source-sentence column for mapped queries while computing next-token KL over the supplied prefix; unassigned tokens retain their existing attention. Its scoring intervention includes prompt-to-prompt and within-sentence reads. The eventual score map is then filtered to the evaluation pool and thresholded. Therefore a Thought Anchors score is not the isolated causal effect of an eligible pair, and Thought Anchors does not optimize the answer objective.

```mermaid
flowchart LR
    P[Prompt and cue] --> R[Supplied reasoning]
    T[Fixed reasoning token identities] --> R
    R --> A[Appended answer suffix]
    P --> A
    M[Learned mask] -. changes eligible reads .-> R
```

*The final suffix can read the supplied reasoning and the prompt outside either learned pool. Preserving the forced answer therefore does not establish that the retained pairs caused the original decision.*

For next-token logits, we record the conditional answer ratio $q_A=p_A/(p_A+p_B)$ and raw option mass $p_A+p_B$. The original suffix is `</think>\n\n**Final answer (`. A fixed alternative suffix is `</think>\n\nFinal answer:` followed by option tokens with leading spaces. The second suffix measures readout sensitivity; we do not select between suffixes using proxy-class performance.

For name-key retention, define $K$ as the set of eligible edges, $C\subseteq K$ as edges whose source is an isolated applicant-name chunk, and $M\subseteq K$ as the retained set. The degree-normalized enrichment is

$$E_{{\mathrm{{name}}}}=\frac{{|M\cap C|/|C|}}{{|M|/|K|}}.$$

A value of one equals uniform retention in expectation. The last-quarter reasoning-key enrichment uses the same formula, with $C$ defined by the final quarter of reasoning sentences. That position-based quantity describes mask location; it does not label sentences as decision-causing or as answer copying.

We also replay the original Black-name reasoning under the paired White-name prompt with aligned prompt chunks and identical reasoning tokens. The stored replay contrast measures sensitivity of a supplied trace to its prompt. It is not an on-policy outcome-rate difference, individual causal ground truth, a trace-only feature, or an optimization objective. All replay results remain in the JSON, including failures and near-zero differences.

## Results

![Fixed-sparsity admission-mask reconstruction and cue retention](../../images/rethink0924/mask_feasibility.png)

*Figure 1. All eight prespecified admission inputs. Blue denotes historical effect-input proxies and orange denotes historical small-estimated-effect proxies; the later seed audit invalidated the claimed independence of their confirmation bank. Each line tracks one input through the displayed conditions. All supplied traces end in A/admit. The first row shows the forced next-letter ratio, and the second shows full-vocabulary mass of A and B. Random values average ten masks per input. The third row shows applicant-name key enrichment for prompt-to-reasoning masks and last-quarter reasoning-key enrichment for reasoning-to-reasoning masks. The dashed line is uniform retention in expectation. The figure describes reconstruction and mask features; it does not estimate held-out monitor accuracy.*

{(chr(10)*2).join(lines)}

The prespecified, higher-name-enrichment-means-effect ranking gives these descriptive AUROCs across the eight applications: {auc_text}. These are rankings of a single stated feature on the same eight development applications, without a fitted classifier. The random-feature entry above averages ten independently sampled masks before ranking; the ten individual random draws have AUROCs ranging from {min(p2t['random_cue_auc_by_seed']):.3f} to {max(p2t['random_cue_auc_by_seed']):.3f}. There are only four historical positive and four historical negative proxy inputs, and their confirmation bank overlaps the screening bank. Thought Anchors has low absolute applicant-name enrichment, {min(p2t['cue_enrichment']['thought_anchors']):.3f}–{max(p2t['cue_enrichment']['thought_anchors']):.3f}, despite its descriptive class ranking. Its admission masks mostly exclude applicant-name reads. Degree normalization does not control all differences in trace length, wording, or demographic cues. The values do not establish generalization, and alternative feature choices or orientations must not be selected from these results and reported as held-out performance.

## Fixed professor-hint positive control

The source is historical professor-hint example 1, the first entry of the existing hint selection. The suggested-answer condition appends a Stanford professor recommendation. The control model chose C on 16 of 16 stored control rollouts; the hinted model chose the suggested A answer on 10 of 16 stored rollouts. The stored reasoning has 2,303 tokens before its closing thinking token. The next historical selection, example 4, has 11,945 reasoning tokens and was excluded for the stated compute bound before fitting new masks. We did not search for another example after seeing new results.

The professor example uses the same whole-trace own-answer objective and 80-percent removal budget, with its hint sentence explicitly included in the learnable prompt-to-reasoning pool. Actual option chunks remain frozen. The example is a selected pipeline check, not a fresh validation set.

{hint_text}{hint_image}

## Interpretation and stopping criteria

Answer reconstruction, cue-sensitive routes, and proxy-class discrimination are separate questions. A mask can improve its training objective by removing evidence against an answer. A mask can preserve the answer because the suffix reads the supplied conclusion. A mask can still contain useful monitor features even when removing the eligible pool leaves the forced answer flat. None of those observations alone establishes or refutes monitor utility.

The user chose a stronger trace-level validation requirement before scaling. Fresh application-level proxy labels alone do not satisfy that gate. Hold the larger proxy-class mask batch until a prespecified trace-level test shows what behavior the mask evidence can identify, with positive and negative controls that distinguish cue influence from merely copying a supplied answer. Do not scale the present recipe based on an improvement in conditional answer probability. Any later proxy-class monitor experiment additionally needs fresh or demonstrably nonoverlapping effect estimates, grouped held-out applications, a frozen feature/monitor policy, and a comparison against task-context-plus-CoT and same-size random-mask evidence. The primary decision must be the incremental monitoring value over those baselines at a prespecified threshold or ranking metric. Keep the independent base application as the statistical unit.

Stop scaling this extraction recipe if its apparent monitor gain disappears when holding the answer constant, controlling the mask density and prompt information, or evaluating held-out applications. Also stop if only a tuned readout or an objective chosen from class-separation results supplies the gain. Failure of the explicit-hint check would additionally limit the claim that these masks recover a readily specified cue under whole-trace replay. A negative eight-input feature comparison cannot prove that every possible mask representation is uninformative; it can justify withholding a larger training budget until a sharper positive check succeeds.

Both jobs completed successfully. Slurm allocation time was 4.281 GPU-hours for the admission experiment and 0.706 GPU-hours for the hint control, totaling 4.987 GPU-hours including model loading, Thought Anchors, and evaluation. The two experiment jobs are complete; this ablation queued no judge calls.

## Appendix: reproduction and complete artifacts

- Prespecified admission plan: `results/prompt_bias_v2/rethink0924/mask_feasibility/plan.json`.
- Admission source: `results/prompt_bias_v2/reduce_masks/reduce_masks_qwen3_8b/dataset.json`.
- Admission configs and scripts: `expts/prompt_bias_circuit_discovery/rethink0924/`.
- Admission launcher: `claude_scripts/rethink0924/mask_feasibility.sbatch`; job 87045, array 0–7 initially limited to four concurrent one-GPU tasks. After the separate 32B screen completed, the scheduler limit increased to six using its freed GPUs. The same sixteen fits and scientific settings remained fixed.
- Hint plan and source selection: `results/prompt_bias_v2/rethink0924/hint_positive_control/plan.json`.
- Hint launcher: `claude_scripts/rethink0924/hint_positive_control.sbatch`; job 87051, one GPU.
- Each job uses priority 10000, the default partition, no custom CPU or memory request, and a one-hour limit. The extra hint job and later use of two freed GPUs give a peak of seven mask GPUs. The first four admission inputs required about 8–17 minutes per SNP fit; the longer professor trace required about 33 minutes for SNP and about 42 minutes for its complete job. Larger model or longer trace costs need a separate measurement. Future-budget reproduction remains at four admission GPUs; the allocation history is in `results/prompt_bias_v2/rethink0924/mask_feasibility/scheduler_history.json`.
- Admission per-input evaluations: `results/prompt_bias_v2/rethink0924/mask_feasibility/eval/ex000.json` through `ex014.json`, even indices only. These contain raw probabilities, both suffixes, both prompt replays, every random mask, retained-edge counts, threshold ties, overlap, and final training diagnostics.
- Completed allocation accounting: `results/prompt_bias_v2/rethink0924/mask_feasibility/allocation_accounting.json`.
- Aggregation: `results/prompt_bias_v2/rethink0924/mask_feasibility/summary.json`.
- Hint evaluation: `results/prompt_bias_v2/rethink0924/hint_positive_control/eval.json`.
- Logs: `logs/rethink0924/mask_feasibility_87045_*.log` and `logs/rethink0924/hint_positive_control_87051.log`.

```bash
uv run python -m expts.prompt_bias_circuit_discovery.rethink0924.build_mask_feasibility
sbatch --priority=10000 --array=0-7%4 claude_scripts/rethink0924/mask_feasibility.sbatch
uv run python -m expts.prompt_bias_circuit_discovery.rethink0924.build_hint_positive_control
sbatch --priority=10000 claude_scripts/rethink0924/hint_positive_control.sbatch
uv run python -m expts.prompt_bias_circuit_discovery.rethink0924.report_mask_feasibility
```
'''
    dest=Path('notes/reports_monitorability/rethink0924/mask_feasibility.md');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(report)
    print('Wrote',dest,ROOT/'summary.json')

if __name__=='__main__':main()
