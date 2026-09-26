"""Figures and tables for the analysis-point sweep on the same-decision
admission set (``eval_analysis_point_sweep.py --probe_conditions`` on
``build_same_decision_sweep_dataset.py``).

Cuts are placed three ways: relative to ``</think>`` (m sentences before the
end of the reasoning), relative to the decision sentence (the last reasoning
sentence that states the answer, found by a regular expression), and as a
fraction of the reasoning length. At every cut and for every ablation the
table gives, per class, the mean P(admit) at the probe, its change from the
unmasked value, how many examples drop by more than 0.1, and the AUROC of
the drop between positives and negatives.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.report_same_decision_sweep --group black --md /tmp/x.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc

CONDS = ["clean", "no_prompt", "probe_no_prompt", "no_prompt_probe", "no_trace", "no_both", "no_both_probe"]
LABEL = {"clean": "no mask",
         "no_prompt": "reasoning's reads of the prompt removed",
         "probe_no_prompt": "probe's reads of the prompt removed",
         "no_prompt_probe": "reasoning's and probe's reads of the prompt removed",
         "no_trace": "reasoning-to-reasoning reads removed",
         "no_both": "reasoning's prompt reads and reasoning-to-reasoning reads removed",
         "no_both_probe": "all three removed"}
SHORT = {"clean": "no mask", "no_prompt": "R->P", "probe_no_prompt": "probe->P", "no_prompt_probe": "R->P + probe->P",
         "no_trace": "R->R", "no_both": "R->P + R->R", "no_both_probe": "R->P + R->R + probe->P"}
COLORS = {"clean": "#6e6e6e", "no_prompt": "#2166ac", "probe_no_prompt": "#1b9e77", "no_prompt_probe": "#2166ac",
          "no_trace": "#d6604d", "no_both": "#1a1a1a", "no_both_probe": "#1a1a1a"}
STYLE = {"clean": "-", "no_prompt": "-", "probe_no_prompt": "-", "no_prompt_probe": "--", "no_trace": "-", "no_both": "-", "no_both_probe": "--"}
DECISION = re.compile(r"\b(answer|go with|choose|pick|lean(?:ing|s)?|decide|conclude|settle)\b", re.I)


def decision_index(texts, letter):
    """Index of the last reasoning sentence that states the decision (a decision verb and the answer letter or 'yes')."""
    pat = re.compile(rf"(\b{letter}\b|\b{letter}\)|\byes\b|\badmit)", re.I)
    for j in range(len(texts) - 1, -1, -1):
        t = texts[j]
        if DECISION.search(t) and pat.search(t):
            return j
    return None


def row_at(cuts, k_target):
    """The cut row with the largest k <= k_target among the sentence cuts (not the </think> row)."""
    cand = [c for c in cuts if not c["at_think"] and c["k"] <= k_target]
    return max(cand, key=lambda c: c["k"]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="black")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--img_dir", default="notes/images/prompt_bias_dataset")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    root = f"results/prompt_bias_v2/same_decision_sweep"
    D = {r["example_id"]: r for r in json.load(open(f"{root}/dataset_{args.tag}_{args.group}.json"))}
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/sweep_{args.tag}_{args.group}/ex*.json"))]
    print(len(R), "examples of", len(D))
    ex = []
    for r in R:
        cuts = r["cuts"]
        think = cuts[-1]
        assert think["at_think"]
        texts = think.get("reasoning_sentence_texts") or []
        d = decision_index(texts, r["target_letter"])
        n_r = r["n_reasoning_sentences"]
        ps = [c["clean"]["p_target"] for c in cuts]
        commit = next((j for j in range(len(cuts)) if all(p >= 0.5 for p in ps[j:])), None)
        meta = D[r["example_id"]]
        ex.append(dict(id=r["example_id"], pos=r["is_positive"], n_r=n_r, n_texts=len(texts), d=d, cuts=cuts, think=think,
                       commit_k=None if commit is None else cuts[commit]["k"], attributable=meta.get("attributable_white"),
                       delta=meta.get("delta_white"), p_black=meta.get("p_black"), p_white=meta.get("p_white")))
    n_pos = sum(e["pos"] for e in ex)
    md = []
    md.append(f"{len(ex)} examples ({n_pos} positives, {len(ex) - n_pos} negatives), one admit trace under the Black name per input. "
              f"Reasoning sentences: median {int(np.median([e['n_r'] for e in ex]))} (range {min(e['n_r'] for e in ex)} to {max(e['n_r'] for e in ex)}). "
              f"Decision sentence found in {sum(e['d'] is not None for e in ex)} of {len(ex)} traces; its position: median "
              f"{np.median([e['d'] / max(1, e['n_texts'] - 1) for e in ex if e['d'] is not None]):.2f} of the reasoning, "
              f"{np.median([e['n_texts'] - 1 - e['d'] for e in ex if e['d'] is not None]):.0f} sentences before the last one (median).")

    # ---- cut definitions ----
    def cut_rows(e):
        out = {}
        cuts, n_r, d = e["cuts"], e["n_r"], e["d"]
        out[("think", 0)] = e["think"]
        for m in [1, 2, 3, 5, 10, 15]:
            out[("think", m)] = row_at(cuts, n_r - m) if n_r - m >= 1 else None
        if d is not None:
            for m in [0, 1, 2, 3, 5, 10]:
                out[("decision", m)] = row_at(cuts, d - m) if d - m >= 1 else None
        for f in [0.25, 0.5, 0.75]:
            out[("fraction", f)] = row_at(cuts, max(1, int(round(f * n_r))))
        out[("prompt", 0)] = cuts[0]
        return out

    rows = {e["id"]: cut_rows(e) for e in ex}
    keys = [("prompt", 0), ("fraction", 0.25), ("fraction", 0.5), ("fraction", 0.75),
            ("decision", 10), ("decision", 5), ("decision", 3), ("decision", 2), ("decision", 1), ("decision", 0),
            ("think", 15), ("think", 10), ("think", 5), ("think", 3), ("think", 2), ("think", 1), ("think", 0)]
    NAME = {("prompt", 0): "prompt only, no reasoning", ("think", 0): "at </think>"}

    def name(key):
        if key in NAME:
            return NAME[key]
        kind, m = key
        if kind == "think":
            return f"{m} sentence{'s' if m > 1 else ''} before </think>"
        if kind == "decision":
            return "just before the decision sentence" if m == 0 else f"{m} sentence{'s' if m > 1 else ''} before the decision sentence"
        return f"{m:.2f} of the reasoning"

    # ---- main table: per cut and condition, per class ----
    md.append("\n### P(admit) at the probe under each ablation, by cut\n")
    md.append("Change = ablated minus unmasked, mean over examples; 'drop > 0.1' counts examples whose P(admit) falls by more than 0.1; "
              "AUROC = how well the size of the drop separates positives from negatives (0.5 = not at all).\n")
    md.append("| cut | class | n | no mask | " + " | ".join(f"{SHORT[c]}: change (drop > 0.1; AUROC)" for c in CONDS[1:]) + " |")
    md.append("|---|---|---|---|" + "---|" * (len(CONDS) - 1))
    summary = {}
    for key in keys:
        for cls, lab in [(True, "positives"), (False, "negatives"), (None, "all")]:
            sel = [e for e in ex if (cls is None or e["pos"] == cls) and rows[e["id"]].get(key) is not None]
            if not sel:
                continue
            clean = np.array([rows[e["id"]][key]["clean"]["p_target"] for e in sel])
            cells = []
            for c in CONDS[1:]:
                have = [e for e in sel if c in rows[e["id"]][key]]
                if not have:
                    cells.append("")
                    continue
                cl = np.array([rows[e["id"]][key]["clean"]["p_target"] for e in have])
                ab = np.array([rows[e["id"]][key][c]["p_target"] for e in have])
                drop = cl - ab
                a = ""
                if cls is None:
                    p = [x for x, e in zip(drop, have) if e["pos"]]; n = [x for x, e in zip(drop, have) if not e["pos"]]
                    a = f"; AUROC {auroc(p, n):.2f}" if p and n else ""
                cells.append(f"{(ab - cl).mean():+.3f} ({int((drop > 0.1).sum())}{a})")
                summary[(key, lab, c)] = dict(n=len(have), clean=float(cl.mean()), ablated=float(ab.mean()), drop_gt=int((drop > 0.1).sum()))
            md.append(f"| {name(key)} | {lab} | {len(sel)} | {clean.mean():.3f} | " + " | ".join(cells) + " |")

    # ---- commitment ----
    ck = [e for e in ex if e["commit_k"] is not None]
    md.append(f"\nCommitment point (first cut after which unmasked P(admit) stays at or above 0.5): median {np.median([e['commit_k'] / max(1, e['n_r']) for e in ck]):.2f} "
              f"of the reasoning; {sum(e['commit_k'] == 0 for e in ck)} of {len(ck)} examples are committed from the prompt alone "
              f"(positives {sum(e['commit_k'] == 0 for e in ck if e['pos'])} of {sum(e['pos'] for e in ck)}, negatives "
              f"{sum(e['commit_k'] == 0 for e in ck if not e['pos'])} of {sum(not e['pos'] for e in ck)}); {len(ex) - len(ck)} never commit before </think>.")
    p0 = [e["cuts"][0]["clean"]["p_target"] for e in ex]
    md.append(f"P(admit) from the prompt alone: positives mean {np.mean([p for p, e in zip(p0, ex) if e['pos']]):.2f}, negatives {np.mean([p for p, e in zip(p0, ex) if not e['pos']]):.2f}; "
              f"at </think>: positives {np.mean([e['think']['clean']['p_target'] for e in ex if e['pos']]):.3f}, negatives {np.mean([e['think']['clean']['p_target'] for e in ex if not e['pos']]):.3f}.")

    # ---- figure 1: trajectories relative to </think> and to the decision sentence ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
    for col, (ref, lab) in enumerate([("think", "sentences before </think>"), ("decision", "sentences before the decision sentence")]):
        for row_, cls in enumerate([True, False]):
            ax = axes[row_][col]
            sel = [e for e in ex if e["pos"] == cls and (ref == "think" or e["d"] is not None)]
            offs = list(range(0, 21))
            for c in CONDS:
                ys = []
                for m in offs:
                    vals = []
                    for e in sel:
                        if ref == "think":
                            r = e["think"] if m == 0 else (row_at(e["cuts"], e["n_r"] - m) if e["n_r"] - m >= 1 else None)
                        else:
                            r = row_at(e["cuts"], e["d"] - m) if e["d"] - m >= 1 else None
                        if r is not None and c in r:
                            vals.append(r[c]["p_target"])
                    ys.append(np.mean(vals) if len(vals) >= 10 else np.nan)
                ax.plot([-m for m in offs], ys, STYLE[c], color=COLORS[c], lw=2 if c in ("clean", "no_trace", "no_prompt") else 1.4, label=LABEL[c])
            ax.axhline(0.5, color="k", ls=":", lw=0.6)
            ax.set_ylim(0.45, 1.02)
            ax.set_title(f"{'positives' if cls else 'negatives'} ({len(sel)}): cut relative to {'</think>' if ref == 'think' else 'the decision sentence'}", fontsize=10)
            if row_ == 1:
                ax.set_xlabel(lab)
            if col == 0:
                ax.set_ylabel("mean P(admit) at the probe")
    axes[0][0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"Same-decision admission set ({args.group} names), Qwen3-8B: what the forced answer depends on, by cut", fontsize=11)
    fig.tight_layout(); os.makedirs(args.img_dir, exist_ok=True)
    f1 = f"{args.img_dir}/same_decision_apsweep_{args.group}_trajectories.png"; fig.savefig(f1, dpi=130); plt.close(fig); print("figure", f1)

    # ---- figure 2: per-example drops at four cuts, positives against negatives ----
    show = [("decision", 5), ("decision", 0), ("think", 1), ("think", 0)]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, key in zip(axes, show):
        for ci, c in enumerate(["no_prompt", "no_prompt_probe", "no_trace", "no_both_probe"]):
            for cls, dx in [(True, -0.17), (False, 0.17)]:
                vals = [rows[e["id"]][key]["clean"]["p_target"] - rows[e["id"]][key][c]["p_target"] for e in ex
                        if e["pos"] == cls and rows[e["id"]].get(key) is not None and c in rows[e["id"]][key]]
                if not vals:
                    continue
                x = ci + dx
                ax.boxplot([vals], positions=[x], widths=0.28, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COLORS[c], alpha=0.45 if cls else 0.15), medianprops=dict(color="k"))
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=6, color=COLORS[c], alpha=0.7 if cls else 0.35, zorder=3)
        ax.axhline(0, color="k", ls=":", lw=0.6)
        ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in ["no_prompt", "no_prompt_probe", "no_trace", "no_both_probe"]], fontsize=8, rotation=20)
        ax.set_title(name(key), fontsize=10)
    axes[0].set_ylabel("P(admit) drop, no mask minus ablated\n(left box: positives, right: negatives)")
    fig.tight_layout()
    f2 = f"{args.img_dir}/same_decision_apsweep_{args.group}_drops.png"; fig.savefig(f2, dpi=130); plt.close(fig); print("figure", f2)

    # ---- figure 3: where the decision sentence and the commitment point lie ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    dd = [e["n_texts"] - 1 - e["d"] for e in ex if e["d"] is not None]
    axes[0].hist(dd, bins=range(0, max(dd) + 2), color="#6e6e6e")
    axes[0].set_xlabel("sentences between the decision sentence and the last reasoning sentence"); axes[0].set_ylabel("examples")
    axes[0].set_title(f"decision sentence position ({len(dd)} traces matched)", fontsize=10)
    for cls, col, lab in [(True, "#d6604d", "positives"), (False, "#2166ac", "negatives")]:
        rp = [e["commit_k"] / max(1, e["n_r"]) for e in ck if e["pos"] == cls]
        axes[1].hist(rp, bins=20, range=(0, 1), color=col, alpha=0.6, label=lab)
    axes[1].set_xlabel("commitment point as a fraction of the reasoning length"); axes[1].legend(fontsize=8)
    axes[1].set_title("where unmasked P(admit) reaches 0.5 and stays", fontsize=10)
    fig.tight_layout()
    f3 = f"{args.img_dir}/same_decision_apsweep_{args.group}_positions.png"; fig.savefig(f3, dpi=130); plt.close(fig); print("figure", f3)


    # ---- figure 4: P(trace's answer) with no mask at every cut, per example ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for cls, col, lab in [(False, "#2166ac", "negatives"), (True, "#d6604d", "positives")]:
        sel = [e for e in ex if e["pos"] == cls]
        for e in sel:
            xs = [c["k"] / max(1, e["n_r"]) for c in e["cuts"]]; ys = [c["clean"]["p_target"] for c in e["cuts"]]
            axes[0].plot(xs, ys, "-", color=col, lw=0.6, alpha=0.25)
            xs2 = [-(e["n_r"] - c["k"]) for c in e["cuts"] if not c["at_think"]] + [0]
            ys2 = [c["clean"]["p_target"] for c in e["cuts"] if not c["at_think"]] + [e["think"]["clean"]["p_target"]]
            axes[1].plot(xs2, ys2, "-", color=col, lw=0.6, alpha=0.25)
        bins = np.linspace(0, 1, 21)
        mean = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            v = [c["clean"]["p_target"] for e in sel for c in e["cuts"] if lo <= c["k"] / max(1, e["n_r"]) < hi or (hi == 1 and c["k"] / max(1, e["n_r"]) == 1)]
            mean.append(np.mean(v) if v else np.nan)
        axes[0].plot((bins[:-1] + bins[1:]) / 2, mean, "-o", color=col, lw=2.2, ms=4, label=f"{lab} ({len(sel)}), mean")
        offs = list(range(0, 31))
        m2 = []
        for m in offs:
            v = [(e["think"] if m == 0 else row_at(e["cuts"], e["n_r"] - m))["clean"]["p_target"] for e in sel if m == 0 or (e["n_r"] - m >= 1 and row_at(e["cuts"], e["n_r"] - m) is not None)]
            m2.append(np.mean(v) if len(v) >= 10 else np.nan)
        axes[1].plot([-m for m in offs], m2, "-o", color=col, lw=2.2, ms=4, label=f"{lab} ({len(sel)}), mean")
    for ax in axes:
        ax.axhline(0.5, color="k", ls=":", lw=0.6); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc="lower left")
    axes[0].set_xlabel("cut as a fraction of the reasoning length (0 = prompt only, 1 = </think>)"); axes[0].set_ylabel("P(admit) at the probe, no mask")
    axes[1].set_xlabel("cut, in sentences before </think>")
    axes[0].set_title("P(trace's answer) along the trace, one line per example", fontsize=10); axes[1].set_title("the same, aligned at </think>", fontsize=10)
    fig.tight_layout()
    f4 = f"{args.img_dir}/same_decision_apsweep_{args.group}_trajectory_per_example.png"; fig.savefig(f4, dpi=130); plt.close(fig); print("figure", f4)

    # ---- figure 5: per-example drops at every cut of the table ----
    show_all = [k for k in keys if k != ("prompt", 0)]
    ncol = 4; nrow = int(np.ceil(len(show_all) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), sharey=True)
    axes = axes.flatten()
    ab_conds = CONDS[1:]
    for ax, key in zip(axes, show_all):
        for ci, c in enumerate(ab_conds):
            for cls, dx in [(True, -0.18), (False, 0.18)]:
                vals = [rows[e["id"]][key]["clean"]["p_target"] - rows[e["id"]][key][c]["p_target"] for e in ex
                        if e["pos"] == cls and rows[e["id"]].get(key) is not None and c in rows[e["id"]][key]]
                if not vals:
                    continue
                x = ci + dx
                ax.boxplot([vals], positions=[x], widths=0.3, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COLORS[c], alpha=0.45 if cls else 0.15), medianprops=dict(color="k"))
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(vals)), vals, s=4, color=COLORS[c], alpha=0.7 if cls else 0.35, zorder=3)
        ax.axhline(0, color="k", ls=":", lw=0.6)
        ax.set_xticks(range(len(ab_conds))); ax.set_xticklabels([SHORT[c] for c in ab_conds], fontsize=7, rotation=25, ha="right")
        n_here = sum(rows[e["id"]].get(key) is not None for e in ex)
        ax.set_title(f"{name(key)} (n = {n_here})", fontsize=9.5)
    for ax in axes[len(show_all):]:
        ax.axis("off")
    for r_ in range(nrow):
        axes[r_ * ncol].set_ylabel("P(admit) drop, no mask minus ablated\n(left box: positives, right: negatives)", fontsize=8)
    fig.tight_layout()
    f5 = f"{args.img_dir}/same_decision_apsweep_{args.group}_drops_all_cuts.png"; fig.savefig(f5, dpi=120); plt.close(fig); print("figure", f5)

    text = "\n".join(md); print(text)
    if args.md:
        open(args.md, "w").write(text)
    json.dump([{k: v for k, v in e.items() if k not in ("cuts", "think")} for e in ex],
              open(f"{root}/sweep_{args.tag}_{args.group}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
