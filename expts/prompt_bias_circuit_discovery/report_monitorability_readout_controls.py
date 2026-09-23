"""Summarize the small readout-route diagnostic, without population inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    ('clean', 'Clean'),
    ('production_no_both', 'Drop prompt→reasoning + earlier reasoning'),
    ('production_no_both_probe', 'Also drop probe→prompt (mapped tokens)'),
    ('strict_no_both_probe_keep_start3', 'Close unmapped routes; keep first 3 tokens'),
    ('probe_no_prompt', 'Probe reads reasoning only'),
    ('probe_no_reasoning', 'Probe reads prompt only'),
    ('probe_isolated', 'Probe reads no prefix tokens'),
    ('probe_isolated_keep_start3', 'Probe reads first 3 prefix tokens only'),
    ('probe_isolated_keep_unassigned', 'Probe reads unmapped prefix tokens only'),
    ('all_token_self_only', 'Every token reads itself only'),
    ('suffix_only_matching_rope', 'Suffix alone at matching positions'),
]


def write_report(results_path):
    source = Path(results_path)
    data = json.loads(source.read_text())
    assert len(data['rows']) == len(data['ids']) * 3, 'Report only complete diagnostics.'
    summary = {'n_inputs':len({r['uid'] for r in data['rows']}), 'n_traces':len(data['ids']),
               'elapsed_seconds':data['elapsed_seconds'], 'groups':{}, 'parity_max':{}, 'option_reversal':[]}
    cuts = ['prompt', 'half', 'think']
    for cut in cuts:
        summary['groups'][cut] = {}
        for answer in ['A','B']:
            rows = [r for r in data['rows'] if r['cut']==cut and r['trace_answer']==answer]
            conditions = {}
            for name in sorted({name for r in rows for name in r['conditions']}):
                use = [r for r in rows if name in r['conditions']]
                conditions[name] = dict(n=len(use),
                    conditional_p_trace=mean(r['conditions'][name]['conditional'][r['letters'].index(answer)] for r in use),
                    conditional_p_A=mean(r['conditions'][name]['conditional'][0] for r in use),
                    raw_p_A=mean(r['conditions'][name]['raw'][0] for r in use),
                    raw_p_B=mean(r['conditions'][name]['raw'][1] for r in use),
                    answer_mass=mean(r['conditions'][name]['answer_mass'] for r in use))
            summary['groups'][cut][answer] = conditions
    for key in {key for r in data['rows'] for key in r['validation']}:
        summary['parity_max'][key] = max(r['validation'].get(key, 0) for r in data['rows'])
    for name in ['direct_identity','production_identity']:
        summary['parity_max'][name+'_conditional'] = max(abs(a-b) for r in data['rows'] for a,b in zip(r['conditions'][name]['conditional'],r['conditions']['clean']['conditional']))
    for reversed_rec in data['reversed_options']:
        clean = next(r['conditions']['clean'] for r in data['rows'] if r['uid']==reversed_rec['uid'] and r['cut']=='prompt')
        summary['option_reversal'].append(dict(example_id=reversed_rec['example_id'],
            original_conditional_p_admit=clean['conditional'][0], reversed_conditional_p_admit=reversed_rec['raw']['conditional'][1],
            original_answer_mass=clean['answer_mass'], reversed_answer_mass=reversed_rec['raw']['answer_mass']))
    source.with_name('summary.json').write_text(json.dumps(summary,indent=2))

    plt.rcParams.update({'font.size':14, 'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14})
    fig,axes=plt.subplots(2,3,figsize=(23,15),sharey=True,layout='constrained')
    colors={'A':'#0072B2','B':'#D55E00'}
    y=np.arange(len(CONDITIONS)); height=.36
    for j,cut in enumerate(cuts):
        for answer,offset in [('A',-height/2),('B',height/2)]:
            values=summary['groups'][cut][answer]
            axes[0,j].barh(y+offset,[values[k]['conditional_p_trace'] for k,_ in CONDITIONS],height=height,color=colors[answer],label=f'Stored {answer} traces')
            axes[1,j].barh(y+offset,[values[k]['answer_mass'] for k,_ in CONDITIONS],height=height,color=colors[answer])
        axes[0,j].set_title({'prompt':'Prompt only','half':'Half of stored reasoning','think':'End of stored reasoning'}[cut])
        axes[0,j].set_xlim(0,1.03)
        axes[0,j].set_xticks([0,.25,.5,.75,1])
        axes[0,j].set_xlabel('Mean P(stored answer | A or B)')
        axes[1,j].set_xscale('log')
        axes[1,j].set_xlim(.001,1.05)
        axes[1,j].set_xlabel('Mean P(A) + P(B), full vocabulary')
        for ax in axes[:,j]:
            ax.set_yticks(y,[label for _,label in CONDITIONS])
            ax.grid(axis='x',alpha=.22)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].invert_yaxis()
    axes[0,0].legend(loc='lower left',bbox_to_anchor=(0,1.04),ncol=2,frameon=False)
    image_path=Path('notes/images/monitorability_design_audit/readout_controls.png')
    image_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(image_path,dpi=150,bbox_inches='tight')
    plt.close(fig)

    end=summary['groups']['think']
    def grouped(name,metric='conditional_p_trace'):
        return f"{end['A'][name][metric]:.4f} for the four A traces and {end['B'][name][metric]:.4f} for the four B traces"
    pooled=lambda name,metric: mean(end[a][name][metric] for a in ['A','B'])
    parity=summary['parity_max']
    reverse='; '.join(f"input {r['example_id']}: {r['original_conditional_p_admit']:.3f} → {r['reversed_conditional_p_admit']:.3f}" for r in summary['option_reversal'])
    report=rf'''# Attention masks change answer readout, while normalization hides low answer-token probability

We ran a small diagnostic to explain why removing prompt-to-reasoning and reasoning-to-reasoning attention had left forced answers stable. The experiment checks the production masking implementation against explicit token masks, closes unassigned-token routes, separates prompt and reasoning reads at the answer suffix, and measures full-vocabulary answer probability. The completed run contains eight stored traces from four admission inputs; the results diagnose the implementation and readout, not monitorability or population-level effects.

## Experiment

The model is Qwen3-8B in evaluation mode with bfloat16 weights. Two inputs have positive estimated Black-name versus White-name admission effects, and two inputs have zero observed effect in the source confirmation samples. Each input contributes one stored A/admit trace and one stored B/reject trace. The source dataset is the newer reduce-mask dataset, not the original 179-example analysis-point dataset. The trace IDs are {data['ids']}.

Each trace is cut at the prompt, halfway through its reasoning sentences, and immediately before the stored closing thinking token. We append the same deterministic answer suffix as the mask experiments and read the next-token probabilities for A and B. We replay stored reasoning tokens under each mask. We do not sample new reasoning under the intervention.

For a suffix position with logits $\ell$, the raw answer probabilities and conditional readout are

$$p_A=\operatorname{{softmax}}(\ell)_A,\qquad p_B=\operatorname{{softmax}}(\ell)_B,\qquad m=p_A+p_B,\qquad q_A=\frac{{p_A}}{{m}},\quad q_B=\frac{{p_B}}{{m}}.$$

The conditional probability of the stored answer is $q_A$ for an A trace and $q_B$ for a B trace. The figure also reports $m$, because a confident conditional answer can coexist with little full-vocabulary probability on either answer token. Each plotted bar averages four traces, one from each input, within a fixed stored-answer group. Four inputs are the independent units. We report descriptive values without significance tests or uncertainty claims.

All attention interventions apply to every layer and head and preserve each token's self-attention. Sentence masks also preserve reads within the same sentence. The production mask excludes the first three chat-template tokens and the unassigned thinking-prefix tokens. The strict variant blocks incoming prompt reads for all reasoning tokens, including the thinking prefix, while keeping only the first three context-free chat-template tokens as shared attention keys. The fully isolated probe prevents every suffix token from reading every prefix token, including all unassigned tokens.

```mermaid
flowchart LR
    P[Prompt tokens] --> R[Stored reasoning tokens]
    P --> Q[Every answer-suffix token]
    R --> Q
    T[Teacher-forced reasoning token identities] --> R
    Q --> L[Next-token logits]
```

The diagram shows why a sentence-attention cut does not remove the semantic content of stored reasoning. Token identities still enter through the residual stream, and the answer suffix can still read the resulting reasoning states. The direct token-mask implementation bypasses sentence-mask expansion but shares the architecture-specific attention projections and SDPA computation. A native eager-attention control checks the unmasked architecture replacement on one prompt.

## Results

![The answer readout changes when the probe loses access to the stored reasoning.](../../images/monitorability_design_audit/readout_controls.png)

*Figure 1. Qwen3-8B readout controls on four admission inputs, each with one stored A trace and one stored B trace. Columns show the three prefix cuts. The top row gives mean probability of the stored answer after renormalizing over A and B. The bottom row gives mean full-vocabulary probability of either answer token on a logarithmic axis. Blue and orange bars summarize the four A and four B traces separately. The explicit token masks affect all layers and heads. The suffix-only control uses the same rotary position indices as the appended suffix. The plot is a descriptive implementation diagnostic with four independent inputs.*

At the end of reasoning, the clean conditional probability of the stored answer is {grouped('clean')}. Dropping the two production reasoning-attention pools plus mapped probe-to-prompt reads gives {grouped('production_no_both_probe')}. Closing the unassigned-token routes while keeping the first three chat-template tokens gives {grouped('strict_no_both_probe_keep_start3')}.

The stored B traces reveal the larger changes. Preventing the probe from reading any reasoning tokens gives {grouped('probe_no_reasoning')}. Allowing the probe to read reasoning but preventing all direct prompt reads gives {grouped('probe_no_prompt')}. One B trace, example 5, changes its preferred answer under the latter intervention; the descriptive mean does not imply that every trace uses the same route.

Full probe isolation gives {grouped('probe_isolated')}. That apparently confident A preference belongs largely to the suffix prior: across the 24 prefix/cut combinations, the isolated conditional probability of A lies between {min(r['conditions']['probe_isolated']['conditional'][0] for r in data['rows']):.4f} and {max(r['conditions']['probe_isolated']['conditional'][0] for r in data['rows']):.4f}. At the end of reasoning, the mean raw answer-token mass falls from {pooled('clean','answer_mass'):.5f} when clean to {pooled('probe_isolated','answer_mass'):.5f} under isolation. The group-specific isolated masses are {grouped('probe_isolated','answer_mass')}. A high normalized answer probability therefore does not establish a high probability of an immediate answer letter. Even the clean prompt-only readout most often prefers the literal token `letter` on three of the four inputs; a continuation such as `letter A` could remain valid. Low immediate A/B probability alone does not establish failure of the final answer format.

The production and direct token implementations agree exactly on conditional A/B probabilities for both tested region masks across all 24 cuts. All-one masks and the public evaluation helper also agree exactly on conditional probabilities with the clean reference. The native eager-attention prompt check has maximum conditional error {parity['native_eager_conditional_max_error']:.6g}, and swapping the future placeholder A token to B has error {parity['placeholder_causality_max_error']:.6g}. These checks support correct causal masking and correct sentence-to-token expansion on this sample. The native eager check covers one prompt and does not establish full-logit equality for every trace.

The isolated probe and suffix-only control differ by at most {parity['isolated_suffix_conditional_max_error']:.5f} in conditional answer probability. Sequence-shape-dependent bfloat16 numerical effects are a plausible explanation for this difference; the run did not verify that explanation independently. The result supports an approximate suffix-prior interpretation rather than numerical identity. The fully isolated-probe mask removes every prefix attention key from every suffix query; no unassigned prefix tokens remain accessible in that condition.

We also reverse the options on the four prompts without replaying any reasoning. After mapping the answers back to admission, the conditional admission probabilities are {reverse}. The semantic answer follows the reversed letter mapping. The fixed A/B ordering does not alone explain the prompt-only admission preferences in these four inputs.

## Takeaways

The tested production masks work: identity, placeholder-causality, and independent token-expansion checks all pass for the conditional answer distribution. The result does not establish that a mask identifies a causal computation of demographic bias.

The unchanged A/admit traces obscure larger effects on B/reject traces. The original A-only analysis therefore needs answer-balanced controls. Existing stored reasoning can carry its conclusion after inter-sentence attention cuts because the token identities and within-sentence processing remain available.

Always report raw answer-token probability beside conditional A/B probability. The fully isolated suffix remains about 84 percent A after normalization while placing less than one percent probability on the two answer tokens in total. An apparently preserved conditional answer under a severe ablation can reflect the suffix prior despite low probability of an immediate answer letter. The readout is a forced next-letter surrogate, not a measurement of a sampled final answer.

These observations favor earlier cuts, explicit answer-route controls, and on-policy regeneration before interpreting sparse masks as faithful behavior circuits. The four-input diagnostic cannot choose a final training objective or establish a monitorability gain.

## Appendix: reproduction and artifacts

The GPU run used job 87007, one GPU, priority 10000, and no custom CPU or memory request. The script initially used the default partition and a 45-minute limit. While pending, we shortened the limit to 20 minutes and expanded this job's eligibility to the currently-up partitions. We did not modify other jobs. The diagnostic forward passes completed in {data['elapsed_seconds']:.1f} seconds after model loading, and the job completed successfully.

- Evaluator: `expts/prompt_bias_circuit_discovery/eval_monitorability_readout_controls.py`.
- Launcher: `claude_scripts/prompt_bias_readout_controls.sbatch`.
- Report generator: `expts/prompt_bias_circuit_discovery/report_monitorability_readout_controls.py`.
- Source dataset: `{data['dataset']}`.
- Detailed results, including raw $p_A$, raw $p_B$, their sum, and conditional probabilities for every condition: `{source.as_posix()}`.
- All per-condition means by cut and stored A/B answer group, with parity maxima: `{source.with_name('summary.json').as_posix()}`.
- Completed job log: `logs/prompt_bias_v2/readout_controls_87007.log`.

```bash
sbatch --priority=10000 claude_scripts/prompt_bias_readout_controls.sbatch
uv run python -m expts.prompt_bias_circuit_discovery.report_monitorability_readout_controls --results {source.as_posix()}
```
'''
    report_path=Path('notes/reports_monitorability/masking_attention_knockouts/readout_controls_0923.md')
    report_path.write_text(report)
    print(f'Wrote {report_path}, {image_path}, and {source.with_name("summary.json")}')
    return summary


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results/prompt_bias_v2/readout_controls_0923/results.json')
    write_report(ap.parse_args().results)
