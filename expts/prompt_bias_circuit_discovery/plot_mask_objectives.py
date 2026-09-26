"""Figure and table for the mask-objective evaluation of the mask-assisted
judge study (``eval_mask_objectives.py``): P(trace's answer) and the KL
from the clean answer distribution under each binarised mask (top 20
percent kept), against random masks of the same size, the unmasked model
and every prompt read removed.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.plot_mask_objectives [--tag qwen3_8b]
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

LABEL = {"clean": "no mask", "p2t_rg": "SNP, raise P(answer)", "p2t_kl": "SNP, keep KL", "ta": "Thought Anchors",
         "random": "random (mean of 5)", "all_removed": "all prompt reads removed"}
ORDER = ["clean", "p2t_rg", "p2t_kl", "ta", "random", "all_removed"]
COLORS = {"clean": "#7f7f7f", "p2t_rg": "#2166ac", "p2t_kl": "#1b7837", "ta": "#d6604d", "random": "#bdbdbd", "all_removed": "#000000"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--img_dir", default="notes/images/prompt_bias_masks")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(f"results/prompt_bias_masks/masks_{args.tag}/objective_eval/ex*.json"))
    R = [json.load(open(f)) for f in files]
    print(len(R), "examples evaluated")
    vals = {c: {"p": [], "kl": []} for c in ORDER}
    per_ex = []
    for r in R:
        vals["clean"]["p"].append(r["clean"]["p_target"]); vals["clean"]["kl"].append(0.0)
        row = dict(id=r["example_id"], case=r["case"], family=r["family"], subset=r["subset"], pos=r["is_positive"], clean=r["clean"]["p_target"])
        for c in ORDER[1:]:
            v = r["conditions"].get(c)
            if v is None:
                continue
            vals[c]["p"].append(v["p_target"]); vals[c]["kl"].append(v["kl"])
            row[c] = (v["p_target"], v["kl"])
        if "random" in r["conditions"]:
            row["random_samples"] = [(s["p_target"], s["kl"]) for s in r["conditions"]["random"]["samples"]]
        per_ex.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, lab, log in [(axes[0], "p", "P(trace's answer) at the probe", False), (axes[1], "kl", "KL from the unmasked answer distribution", True)]:
        data = [vals[c][key] for c in ORDER]
        bp = ax.boxplot(data, positions=range(len(ORDER)), widths=0.55, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for k, c in enumerate(ORDER):
            x = k + rng.uniform(-0.18, 0.18, len(vals[c][key]))
            ax.scatter(x, vals[c][key], s=9, color=COLORS[c], alpha=0.7, zorder=3)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=25, ha="right", fontsize=9)
        ax.set_title(lab, fontsize=10)
        if log:
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel("KL (symlog)")
        else:
            ax.set_ylim(0, 1.02); ax.set_ylabel("probability")
    fig.suptitle(f"Masks of the mask-assisted judge study on their own objective ({len(R)} examples, top 20% of the prompt-to-trace pool kept)", fontsize=10)
    fig.tight_layout()
    os.makedirs(args.img_dir, exist_ok=True)
    out = f"{args.img_dir}/mask_objectives_{args.tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("figure", out)
    # per-example: KL of each method against the random mean, and P(answer) drop
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in ["p2t_rg", "p2t_kl", "ta"]:
        xs = [r["random"][1] for r in per_ex if m in r and "random" in r]; ys = [r[m][1] for r in per_ex if m in r and "random" in r]
        axes[0].scatter(xs, ys, s=12, color=COLORS[m], label=LABEL[m], alpha=0.75)
        xs = [r["clean"] for r in per_ex if m in r]; ys = [r[m][0] for r in per_ex if m in r]
        axes[1].scatter(xs, ys, s=12, color=COLORS[m], label=LABEL[m], alpha=0.75)
    lim = max(max(vals["random"]["kl"] + [1e-3]), max(v for m in ["p2t_rg", "p2t_kl", "ta"] for v in vals[m]["kl"]))
    axes[0].plot([1e-5, lim], [1e-5, lim], "k:", lw=0.8); axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("KL under a random mask of the same size (mean of 5)"); axes[0].set_ylabel("KL under the method's mask"); axes[0].legend(fontsize=8)
    axes[0].set_title("below the diagonal: the mask keeps the distribution better than random", fontsize=9)
    axes[1].plot([0, 1], [0, 1], "k:", lw=0.8); axes[1].set_xlabel("P(trace's answer), no mask"); axes[1].set_ylabel("P(trace's answer) under the mask")
    axes[1].set_xlim(0.3, 1.01); axes[1].set_ylim(0, 1.01); axes[1].set_title("below the diagonal: the mask lowers the answer probability", fontsize=9)
    fig.tight_layout(); out2 = f"{args.img_dir}/mask_objectives_{args.tag}_scatter.png"; fig.savefig(out2, dpi=130); plt.close(fig)
    print("figure", out2)
    # table
    md = ["| condition | n | P(trace's answer): median [IQR] | mean | KL from clean: median [IQR] | mean | examples with KL below the random mean |", "|---|---|---|---|---|---|---|"]
    for c in ORDER:
        p, kl = np.array(vals[c]["p"]), np.array(vals[c]["kl"])
        if not len(p):
            continue
        better = ""
        if c in ("p2t_rg", "p2t_kl", "ta"):
            pairs = [(r[c][1], r["random"][1]) for r in per_ex if c in r and "random" in r]
            better = f"{sum(a < b for a, b in pairs)} / {len(pairs)}"
        md.append(f"| {LABEL[c]} | {len(p)} | {np.median(p):.3f} [{np.percentile(p, 25):.3f}, {np.percentile(p, 75):.3f}] | {p.mean():.3f} | "
                  f"{np.median(kl):.4f} [{np.percentile(kl, 25):.4f}, {np.percentile(kl, 75):.4f}] | {kl.mean():.4f} | {better} |")
    # by class, positives vs negatives, for the learned masks
    md.append("\n| condition | class | P(trace's answer) positives (mean) | negatives (mean) | KL positives (mean) | negatives (mean) |")
    md.append("|---|---|---|---|---|---|")
    for c in ["p2t_rg", "p2t_kl", "ta", "random"]:
        for cls in ["explicit", "prompt_change", "same_prompt"]:
            P = [r for r in per_ex if c in r and r["case"] == cls and not r["subset"] and r["pos"]]
            N = [r for r in per_ex if c in r and r["case"] == cls and not r["subset"] and not r["pos"]]
            if P and N:
                md.append(f"| {LABEL[c]} | {cls} | {np.mean([r[c][0] for r in P]):.3f} | {np.mean([r[c][0] for r in N]):.3f} | "
                          f"{np.mean([r[c][1] for r in P]):.4f} | {np.mean([r[c][1] for r in N]):.4f} |")
    text = "\n".join(md)
    print(text)
    if args.md:
        open(args.md, "w").write(text)
    json.dump(per_ex, open(f"results/prompt_bias_masks/analysis/mask_objectives_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
