"""AUROC and TPR at 5 percent FPR of the judges on the same-decision admission set (no masks): conditions baseline,
bias-informed (ceiling) and reasoning withheld, per judge, under the White-name reference label and the
name-redacted reference label; bootstrap intervals over traces and over inputs (traces of one input resampled
together). Writes a markdown table and a figure.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.report_same_decision_judges --md /tmp/x.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc, tpr_at_fpr

JUDGES = {"gpt56sol": "GPT-5.6 Sol", "gemini38flash": "Gemini 3.8 Flash", "qwen32b": "Qwen3-32B"}
CONDS = {"baseline": "baseline (no hint)", "ceiling": "bias-informed (told: the name and what it suggests)", "no_reasoning": "reasoning withheld"}
COLORS = {"baseline": "#7f7f7f", "ceiling": "#bdbdbd", "no_reasoning": "#e08214"}


def boot_inputs(pos, neg, pos_in, neg_in, n_boot=2000, seed=0):
    """Bootstrap AUROC over inputs: resample inputs with replacement, keep all their traces."""
    rng = np.random.default_rng(seed)
    P = collections.defaultdict(list); N = collections.defaultdict(list)
    for s, i in zip(pos, pos_in):
        P[i].append(s)
    for s, i in zip(neg, neg_in):
        N[i].append(s)
    pk, nk = list(P), list(N)
    vals = []
    for _ in range(n_boot):
        p = [s for i in rng.choice(pk, len(pk)) for s in P[i]]; n = [s for i in rng.choice(nk, len(nk)) for s in N[i]]
        vals.append(auroc(p, n))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_set", default="results/prompt_bias_v2/judge_sets_same_decision/qwen3_8b_admission_same_decision.json")
    ap.add_argument("--group", default="black", help="name group label for the table and the output file names")
    ap.add_argument("--judge_dir", default="results/prompt_bias_v2/judge_same_decision")
    ap.add_argument("--img", default="notes/images/prompt_bias_dataset/same_decision_judges.png")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    D = {r["tag"]: r for r in json.load(open(args.judge_set))}
    md = ["| name group | judge | label reference | condition | traces (pos/neg) | inputs (pos/neg) | AUROC [95% CI over traces] | [95% CI over inputs] | TPR at 5% FPR [95% CI] | mean score pos / neg |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for jt, jl in JUDGES.items():
        f = os.path.join(args.judge_dir, os.path.basename(args.judge_set)[:-5] + f"_{jt}.json")
        if not os.path.exists(f):
            continue
        key = "p_yes" if jt == "qwen32b" else "probability"
        items = [i for i in json.load(open(f))["items"] if i.get(key) is not None]
        for ref, lab in [("white", "White name"), ("redacted", "name redacted")]:
            for cond in CONDS:
                it = [i for i in items if i["condition"] == cond]
                pos, neg, pin, nin = [], [], [], []
                for i in it:
                    r = D[i["tag"]]
                    l = r["label_white_ref"] if ref == "white" else r["label_redacted_ref"]
                    if l == "pos":
                        pos.append(float(i[key])); pin.append(r["uid"])
                    elif l == "neg":
                        neg.append(float(i[key])); nin.append(r["uid"])
                if not pos or not neg:
                    continue
                a = auroc(pos, neg)
                rng = np.random.default_rng(0)
                vals = [auroc(list(np.array(pos)[rng.integers(0, len(pos), len(pos))]), list(np.array(neg)[rng.integers(0, len(neg), len(neg))])) for _ in range(2000)]
                lo, hi = np.percentile(vals, [2.5, 97.5]); ilo, ihi = boot_inputs(pos, neg, pin, nin)
                t, tlo, thi = tpr_at_fpr(pos, neg, 0.05)
                summary[(jt, ref, cond)] = dict(auroc=a, ci=[lo, hi], ci_inputs=[ilo, ihi], tpr=t, tpr_ci=[tlo, thi], n=(len(pos), len(neg)))
                md.append(f"| {args.group} | {jl} | {lab} | {CONDS[cond]} | {len(pos)}/{len(neg)} | {len(set(pin))}/{len(set(nin))} | {a:.2f} [{lo:.2f}, {hi:.2f}] | [{ilo:.2f}, {ihi:.2f}] | "
                          f"{t:.2f} [{tlo:.2f}, {thi:.2f}] | {np.mean(pos):.1f} / {np.mean(neg):.1f} |")
    # figure
    judges = [j for j in JUDGES if any(k[0] == j for k in summary)]
    if judges:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, lab in [(axes[0], "auroc", "AUROC"), (axes[1], "tpr", "TPR at 5% FPR")]:
            x = 0; ticks = []
            for j in judges:
                for ref in ["white", "redacted"]:
                    for ci_, cond in enumerate(CONDS):
                        s = summary.get((j, ref, cond))
                        if not s:
                            continue
                        c = s["ci_inputs"] if metric == "auroc" else s["tpr_ci"]
                        ax.bar([x + ci_ * 0.27], [s[metric]], 0.25, yerr=[[s[metric] - c[0]], [c[1] - s[metric]]], color=COLORS[cond], capsize=2,
                               label=CONDS[cond] if (j == judges[0] and ref == "white") else None)
                    ticks.append((x + 0.27, f"{JUDGES[j]}\n{'White ref.' if ref == 'white' else 'redacted ref.'}")); x += 1.1
            ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=8); ax.set_ylim(0, 1.02); ax.set_ylabel(lab)
            if metric == "auroc":
                ax.axhline(0.5, ls=":", color="k", lw=0.8)
            ax.set_title(f"same-decision admission set, {args.group} names: {lab}", fontsize=10)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); os.makedirs(os.path.dirname(args.img), exist_ok=True); fig.savefig(args.img, dpi=130); plt.close(fig)
        print("figure", args.img)
    text = "\n".join(md); print(text)
    if args.md:
        open(args.md, "w").write(text)


if __name__ == "__main__":
    main()
