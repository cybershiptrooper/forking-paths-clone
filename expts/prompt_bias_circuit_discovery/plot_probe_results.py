"""Figure for ``train_activation_probes.py`` and ``probe_same_decision.py``
outputs: cross-validated g-mean2 (or AUROC) by layer for every position and
training size, and the held-out numbers.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.plot_probe_results --intervention results/prompt_bias_v2/probes/admission_confirmed.json \
    --same_decision results/prompt_bias_v2/probes/same_decision.json --out notes/images/prompt_bias_dataset/activation_probes.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"last_reasoning": "#2166ac", "think": "#d6604d", "mean_reasoning": "#1b9e77", "last_output": "#6e6e6e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--same_decision", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A = json.load(open(args.intervention))
    B = json.load(open(args.same_decision)) if args.same_decision and os.path.exists(args.same_decision) else None
    runs_a = A["runs"]; runs_b = B["runs"] if B else []
    n = len(runs_a) + len(runs_b)
    fig, axes = plt.subplots(1, max(1, n), figsize=(4.6 * max(1, n), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    k = 0
    for run in runs_a:
        ax = axes[k]; k += 1
        for pos, col in COLORS.items():
            rows = [r for r in run["grid"] if r["pos"] == pos]
            layers = sorted({r["layer"] for r in rows})
            ax.plot(layers, [max(r["cv_gmean2"] for r in rows if r["layer"] == l) for l in layers], "-o", ms=4, color=col, label=pos)
        t = run["test"]; d = run.get("degeneracy", {})
        ax.set_title(f"paper-style set, {run['train_size_per_arm']} rollouts/arm ({run['n_train_samples']} train samples)\n"
                     f"held-out g-mean2 {t['gmean2']:.2f}, AUROC {t['auroc']:.2f}; null-input admits flagged {d.get('flagged_fraction', float('nan')):.2f}", fontsize=8.5)
        ax.set_xlabel("layer"); ax.set_ylabel("cross-validated g-mean2 (best C)"); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, ls=":", color="k", lw=0.6)
        if k == 1:
            ax.legend(fontsize=7)
    for run in runs_b:
        ax = axes[k]; k += 1
        for pos, col in COLORS.items():
            rows = [r for r in run["grid"] if r["pos"] == pos]
            layers = sorted({r["layer"] for r in rows})
            ax.plot(layers, [max(r["auroc"] for r in rows if r["layer"] == l) for l in layers], "-o", ms=4, color=col, label=pos)
        b = run["best"]
        ax.set_title(f"same-decision contrast, {run['train_size_per_input']} admits/input ({run['n_pos']} pos, {run['n_neg']} neg)\n"
                     f"best out-of-fold AUROC {b['auroc']:.2f} (inputs {b['auroc_inputs']:.2f}), g-mean2 {b['gmean2']:.2f}", fontsize=8.5)
        ax.set_xlabel("layer"); ax.set_ylabel("out-of-fold AUROC (best C)"); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, ls=":", color="k", lw=0.6)
    fig.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); fig.savefig(args.out, dpi=130); plt.close(fig); print("figure", args.out)


if __name__ == "__main__":
    main()
