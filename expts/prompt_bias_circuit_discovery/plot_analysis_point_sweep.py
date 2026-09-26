"""Figures and table for the analysis-point sweep (``eval_analysis_point_sweep.py``).

Per example the *commitment point* k* is the smallest cut after which
P(trace's answer) with no mask stays at or above 0.5 for every later cut,
including the exact </think> cut. The figures show (1) P(trace's answer)
under the four conditions against the position of the cut relative to the
commitment point, (2) the three ablations against the unmasked value at the
cut just before the commitment point, at the commitment point and at
</think>, and (3) where the commitment point lies along the reasoning.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.plot_analysis_point_sweep [--tag qwen3_8b]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ["clean", "no_prompt", "no_trace", "no_both"]
LABEL = {"clean": "no mask", "no_prompt": "every prompt read removed", "no_trace": "every reasoning-to-reasoning read removed",
         "no_both": "both removed"}
COLORS = {"clean": "#7f7f7f", "no_prompt": "#2166ac", "no_trace": "#d6604d", "no_both": "#000000"}


def commitment(cuts, thr=0.5):
    ps = [c["clean"]["p_target"] for c in cuts]
    for j in range(len(cuts)):
        if all(p >= thr for p in ps[j:]):
            return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--img_dir", default="notes/images/prompt_bias_masks")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(f"results/prompt_bias_masks/masks_{args.tag}/analysis_point_sweep/ex*.json"))
    R = [json.load(open(f)) for f in files]
    print(len(R), "examples")
    # 1. trajectories relative to the commitment point
    rel = {c: {} for c in CONDS}  # offset -> list
    rows = []
    for r in R:
        cuts = r["cuts"]
        j = commitment(cuts)
        n_r = r["n_reasoning_sentences"]
        rec = {"id": r["example_id"], "case": r["case"], "family": r["family"], "subset": r["subset"], "pos": r["is_positive"], "n_r": n_r,
               "k_star": None if j is None else cuts[j]["k"], "rel_pos": None if j is None else cuts[j]["k"] / max(1, n_r)}
        for name, idx in [("before", None if j is None or j == 0 else j - 1), ("at", j), ("think", len(cuts) - 1)]:
            if idx is None:
                continue
            c = cuts[idx]
            rec[name] = {cond: (c[cond]["p_target"] if cond in c else None) for cond in CONDS}
            rec[name]["k"] = c["k"]
        rows.append(rec)
        if j is None:
            continue
        for q, c in enumerate(cuts):
            off = c["k"] - cuts[j]["k"]
            for cond in CONDS:
                if cond in c:
                    rel[cond].setdefault(off, []).append(c[cond]["p_target"])
    n_commit = sum(r["k_star"] is not None for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    offs = sorted(o for o in rel["clean"] if -15 <= o <= 15 and len(rel["clean"][o]) >= 10)
    for cond in CONDS:
        ax.plot(offs, [np.mean(rel[cond].get(o, [np.nan])) for o in offs], "-o", ms=3, color=COLORS[cond], label=LABEL[cond])
    ax.axvline(0, color="k", ls=":", lw=0.8); ax.axhline(0.5, color="k", ls=":", lw=0.5)
    ax.set_xlabel("cut, in reasoning sentences relative to the commitment point"); ax.set_ylabel("mean P(trace's answer)"); ax.set_ylim(0, 1.02)
    ax.set_title(f"P(trace's answer) around the commitment point ({n_commit} examples)", fontsize=10); ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    for i_, (name, lab) in enumerate([("before", "one sentence before commitment"), ("at", "at commitment"), ("think", "at </think>")]):
        for c_, cond in enumerate(CONDS[1:]):
            vals = [r[name][cond] - r[name]["clean"] for r in rows if name in r and r[name].get(cond) is not None]
            x = i_ + (c_ - 1) * 0.26
            ax.boxplot([vals], positions=[x], widths=0.22, showfliers=False, patch_artist=True, boxprops=dict(facecolor=COLORS[cond], alpha=0.35))
            ax.scatter(x + np.random.default_rng(0).uniform(-0.07, 0.07, len(vals)), vals, s=6, color=COLORS[cond], alpha=0.6, zorder=3,
                       label=LABEL[cond] if i_ == 0 else None)
    ax.axhline(0, color="k", ls=":", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["one sentence before\ncommitment", "at commitment", "at </think>"], fontsize=9)
    ax.set_ylabel("P(trace's answer), ablated minus no mask"); ax.set_title("What the forced answer depends on", fontsize=10); ax.legend(fontsize=7.5, loc="lower left")
    ax = axes[2]
    rp = [r["rel_pos"] for r in rows if r["rel_pos"] is not None]
    ax.hist(rp, bins=20, range=(0, 1), color="#7f7f7f")
    ax.set_xlabel("commitment point as a fraction of the reasoning length"); ax.set_ylabel("examples")
    ax.set_title(f"where the model commits ({len(rows) - n_commit} examples never reach 0.5 and stay)", fontsize=10)
    fig.tight_layout(); os.makedirs(args.img_dir, exist_ok=True)
    out = f"{args.img_dir}/analysis_point_sweep_{args.tag}.png"; fig.savefig(out, dpi=130); plt.close(fig); print("figure", out)
    # table
    md = ["| cut | n | P(trace's answer), no mask (mean) | every prompt read removed (mean, change) | every reasoning-to-reasoning read removed (mean, change) | both removed (mean, change) | examples where removing the prompt reads lowers P by more than 0.1 | ... where removing the reasoning reads does |",
          "|---|---|---|---|---|---|---|---|"]
    for name, lab in [("before", "one sentence before commitment"), ("at", "at commitment"), ("think", "at </think> (the masks' analysis point)")]:
        rr = [r for r in rows if name in r and all(r[name].get(c) is not None for c in CONDS)]
        if not rr:
            continue
        m = {c: np.mean([r[name][c] for r in rr]) for c in CONDS}
        dp = sum(r[name]["clean"] - r[name]["no_prompt"] > 0.1 for r in rr); dt = sum(r[name]["clean"] - r[name]["no_trace"] > 0.1 for r in rr)
        md.append(f"| {lab} | {len(rr)} | {m['clean']:.3f} | {m['no_prompt']:.3f} ({m['no_prompt'] - m['clean']:+.3f}) | {m['no_trace']:.3f} ({m['no_trace'] - m['clean']:+.3f}) | "
                  f"{m['no_both']:.3f} ({m['no_both'] - m['clean']:+.3f}) | {dp} / {len(rr)} | {dt} / {len(rr)} |")
    k0 = [r for r in R]
    p0 = np.mean([r["cuts"][0]["clean"]["p_target"] for r in k0]); p0n = np.mean([r["cuts"][0]["no_prompt"]["p_target"] for r in k0])
    md.append(f"| prompt only, no reasoning (k = 0) | {len(k0)} | {p0:.3f} | {p0n:.3f} ({p0n - p0:+.3f}) | | | | |")
    rp = np.array(rp)
    md.append(f"\nCommitment point: median {np.median(rp):.2f} of the reasoning length (interquartile {np.percentile(rp, 25):.2f} to {np.percentile(rp, 75):.2f}); "
              f"{len(rows) - n_commit} of {len(rows)} examples never reach P(trace's answer) of 0.5 and stay there before </think>.")
    text = "\n".join(md); print(text)
    if args.md:
        open(args.md, "w").write(text)
    json.dump(rows, open(f"results/prompt_bias_masks/analysis/analysis_point_sweep_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
