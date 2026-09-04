"""Aggregate eval rows into the report figures and tables.

Reads results/external_compression/evals/<instance>/<method>.json and writes

- notes/images/individual_sentence_grading_followups/kl_vs_keep_fraction.png
  (primary: keep-fraction ratio view, binned medians per method)
- notes/images/individual_sentence_grading_followups/our_methods_their_dataset.png
  (their fixed M values, medians over instances)
- notes/images/individual_sentence_grading_followups/kl_by_trace_length_bucket.png
  (per-bucket panels at their fixed M values)
- notes/images/individual_sentence_grading_followups/raw_rows.json
- printed median tables.

Rerunnable at any time; uses whatever evals exist.

Usage:
    uv run python -m expts.external_compression.aggregate
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expts.external_compression.common import (
    BUCKET_NAMES,
    M_VALUES,
    REPO_ROOT,
    RESULTS_DIR,
)

EVALS_DIR = os.path.join(RESULTS_DIR, "evals")
OUT_DIR = os.path.join(
    REPO_ROOT, "notes", "images", "individual_sentence_grading_followups"
)

METHODS = ["colsnp", "colsnp_uniform_l0", "suppress", "ta", "attn_last",
           "attn_next", "oracle"]
LABELS = {
    "colsnp": "Column SNP (matched)",
    "colsnp_uniform_l0": "Column SNP (per-sentence L0 budget)",
    "suppress": "Suppress",
    "ta": "Thought Anchors",
    "attn_last": "Attention (last prefix sentence)",
    "attn_next": "Next-sentence attention",
    "oracle": "Single sentence oracle",
}
# Validated categorical palette (dataviz reference, fixed order by entity).
COLORS = {
    "colsnp": "#2a78d6",
    "colsnp_uniform_l0": "#4a3aa7",
    "suppress": "#eb6834",
    "ta": "#1baf7a",
    "attn_last": "#eda100",
    "attn_next": "#e87ba4",
    "oracle": "#008300",
}
LINESTYLE = {m: ("--" if m == "oracle" else "-") for m in METHODS}
FLOOR = 1e-5  # display floor for log-scale KL


def load_rows():
    rows = []
    if not os.path.isdir(EVALS_DIR):
        return rows
    for inst_dir in sorted(os.listdir(EVALS_DIR)):
        d = os.path.join(EVALS_DIR, inst_dir)
        for fn in sorted(os.listdir(d)):
            method = fn[: -len(".json")]
            if method not in METHODS and method != "reference":
                continue
            with open(os.path.join(d, fn)) as f:
                rec = json.load(f)
            for r in rec["rows"]:
                if "answer_kl" not in r or r.get("row") == "baseline_recompute_drift":
                    continue
                rows.append({
                    "instance_id": rec["instance_id"],
                    "question_id": rec["question_id"],
                    "bucket": rec["bucket"],
                    "num_rankable": rec["num_rankable"],
                    "method": method if method != "reference" else "deletion",
                    "M": r.get("M"),
                    "keep_fraction": r.get("keep_fraction"),
                    "their_m": r.get("their_m"),
                    "answer_kl": r["answer_kl"],
                    "token_kl": r.get("token_kl"),
                })
    return rows


def _median_iqr(vals):
    a = np.array([max(v, FLOOR) for v in vals if v is not None])
    if len(a) == 0:
        return None
    return (
        float(np.median(a)),
        float(np.percentile(a, 25)),
        float(np.percentile(a, 75)),
        len(a),
    )


FRACTION_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def _style_axes(ax):
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")


def _fraction_xaxis(ax):
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ax.set_xticks(FRACTION_TICKS)
    ax.set_xticklabels([f"{t:g}" for t in FRACTION_TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlim(0.008, 0.95)


BINS = np.logspace(math.log10(0.006), math.log10(0.9), 9)
BIN_CENTERS = np.sqrt(BINS[:-1] * BINS[1:])


def fig_fraction(rows, path):
    bins, centers = BINS, BIN_CENTERS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, metric, title in zip(
        axes, ["answer_kl", "token_kl"],
        ["Answer KL vs keep fraction", "Token KL vs keep fraction"],
    ):
        for m in METHODS:
            xs, med, lo, hi = [], [], [], []
            for b0, b1, c in zip(bins[:-1], bins[1:], centers):
                sub = [r[metric] for r in rows
                       if r["method"] == m and r["keep_fraction"] is not None
                       and b0 <= r["keep_fraction"] < b1]
                st = _median_iqr(sub)
                if st and st[3] >= 3:
                    xs.append(c); med.append(st[0]); lo.append(st[1]); hi.append(st[2])
            if not xs:
                continue
            ax.plot(xs, med, LINESTYLE[m], color=COLORS[m], linewidth=2,
                    marker="o", markersize=4.5, label=LABELS[m])
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.12, linewidth=0)
        _style_axes(ax)
        _fraction_xaxis(ax)
        ax.set_xlabel("keep fraction M / (compress region size)")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("KL (median, IQR band)")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_buckets(rows, path):
    """Per-bucket panels with the SAME keep-fraction bins as the pooled
    figure, so curves are directly comparable across trace lengths."""
    # Only draw buckets that actually have instances, so an unrun bucket does
    # not leave an empty column in the figure.
    buckets = [b for b in BUCKET_NAMES
               if any(r["bucket"] == b for r in rows)] or list(BUCKET_NAMES)
    fig, axes = plt.subplots(2, len(buckets), figsize=(4 * len(buckets), 7),
                             facecolor="white", sharey="row", sharex=True,
                             squeeze=False)
    any_label = False
    for col, bucket in enumerate(buckets):
        brows = [r for r in rows if r["bucket"] == bucket]
        n_inst = len({r["instance_id"] for r in brows})
        for rowi, metric in enumerate(["answer_kl", "token_kl"]):
            ax = axes[rowi][col]
            for m in METHODS:
                xs, med = [], []
                for b0, b1, c in zip(BINS[:-1], BINS[1:], BIN_CENTERS):
                    sub = [r[metric] for r in brows
                           if r["method"] == m and r["keep_fraction"] is not None
                           and b0 <= r["keep_fraction"] < b1]
                    st = _median_iqr(sub)
                    if st and st[3] >= 2:
                        xs.append(c)
                        med.append(st[0])
                if xs:
                    ax.plot(xs, med, LINESTYLE[m], color=COLORS[m], linewidth=2,
                            marker="o", markersize=4, label=LABELS[m])
                    any_label = True
            _style_axes(ax)
            _fraction_xaxis(ax)
            if rowi == 0:
                ax.set_title(f"{bucket} rankable ({n_inst} instances)", fontsize=10)
            if rowi == 1:
                ax.set_xlabel("keep fraction")
    axes[0][0].set_ylabel("Answer KL")
    axes[1][0].set_ylabel("Token KL")
    if any_label:
        handles, labels = axes[0][0].get_legend_handles_labels()
        if not handles:
            for col in range(len(buckets)):
                handles, labels = axes[0][col].get_legend_handles_labels()
                if handles:
                    break
        fig.legend(handles, labels, fontsize=8, frameon=False,
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


BUCKET_RAMP = {  # sequential single-hue (blue), light -> dark by trace length
    "lt50": "#9ec4ec", "50-100": "#5f9fe0", "100-200": "#2a78d6",
    "200plus": "#1b4e8f",
}


def fig_convergence(method_name: str, path: str) -> bool:
    """SNP training curves: task loss, L0 hinge, sparsity-minus-target.

    One line per instance, colored by trace-length bucket (light -> dark).
    Reads scores/<instance>/<method>_logs/training_metrics.jsonl.
    """
    import glob as _glob
    scores_dir = os.path.join(RESULTS_DIR, "scores")
    inst_path = os.path.join(REPO_ROOT, "data", "external_compression", "instances.json")
    with open(inst_path) as f:
        bucket_by_id = {r["instance_id"]: r["bucket"] for r in json.load(f)}

    runs = []
    for p in sorted(_glob.glob(
        os.path.join(scores_dir, "*", f"{method_name}_logs", "training_metrics.jsonl")
    )):
        inst = p.split("/scores/")[1].split("/")[0]
        steps = [json.loads(l) for l in open(p)]
        if len(steps) < 3:
            continue
        mask_p = os.path.join(scores_dir, inst, f"{method_name}_mask.json")
        target = None
        if os.path.exists(mask_p):
            target = json.load(open(mask_p))["metadata"].get("target_sparsity")
        runs.append((inst, bucket_by_id.get(inst, "?"), steps, target))
    if not runs:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="white")
    for inst, bucket, steps, target in runs:
        xs = [s["step"] for s in steps]
        c = BUCKET_RAMP.get(bucket, "#888888")
        axes[0].plot(xs, [max(s["task_loss"], FLOOR) for s in steps],
                     color=c, linewidth=1.2, alpha=0.85)
        axes[1].plot(xs, [s["l0_loss"] for s in steps],
                     color=c, linewidth=1.2, alpha=0.85)
        if target is not None:
            axes[2].plot(xs, [s["sparsity"] - target for s in steps],
                         color=c, linewidth=1.2, alpha=0.85)
    n_steps = max(s["step"] for _, _, st, _ in runs for s in st)
    for ax, title, ylab in zip(
        axes,
        ["Task loss (answer-probe KL)", "L0 hinge loss",
         "Achieved sparsity − target"],
        ["KL (log scale)", "ReLU(active − budget)", "sparsity gap"],
    ):
        for frac, lab in [(0.25, "λ ramp start"), (0.75, "λ full")]:
            ax.axvline(frac * n_steps, color="#c9c8c4", linestyle=":",
                       linewidth=1)
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yscale("log")
    axes[2].axhline(0.0, color="#52514e", linewidth=1)
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([], [], color=BUCKET_RAMP[b], linewidth=2,
                        label=f"{b} rankable") for b in BUCKET_NAMES],
        fontsize=8, frameon=False, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"convergence figure ({method_name}): {len(runs)} runs -> {path}")
    return True


def print_tables(rows):
    for metric in ["answer_kl", "token_kl"]:
        print(f"\n=== median {metric} by keep-fraction bin ===")
        header = f"{'frac':>7} {'n':>4} " + " ".join(f"{m:>10}" for m in METHODS)
        print(header)
        for b0, b1, c in zip(BINS[:-1], BINS[1:], BIN_CENTERS):
            cells = []
            n_any = 0
            for m in METHODS:
                sub = [r[metric] for r in rows
                       if r["method"] == m and r["keep_fraction"] is not None
                       and b0 <= r["keep_fraction"] < b1]
                st = _median_iqr(sub)
                n_any = max(n_any, st[3] if st else 0)
                cells.append(f"{st[0]:>10.4f}" if st else f"{'-':>10}")
            if n_any:
                print(f"{c:>7.3f} {n_any:>4} " + " ".join(cells))
    dele = [r for r in rows if r["method"] == "deletion"]
    if dele:
        st = _median_iqr([r["answer_kl"] for r in dele])
        print(f"\ndeletion (M=0) median answer KL: {st[0]:.4f} (n={st[3]})")


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no eval rows found yet")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "raw_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    n_inst = len({r["instance_id"] for r in rows})
    per_bucket = defaultdict(set)
    for r in rows:
        per_bucket[r["bucket"]].add(r["instance_id"])
    print(f"{len(rows)} rows from {n_inst} instances; "
          + ", ".join(f"{b}: {len(v)}" for b, v in sorted(per_bucket.items())))
    fig_fraction(rows, os.path.join(OUT_DIR, "kl_vs_keep_fraction.png"))
    fig_buckets(rows, os.path.join(OUT_DIR, "kl_vs_keep_fraction_by_bucket.png"))
    fig_convergence("colsnp", os.path.join(OUT_DIR, "snp_convergence.png"))
    fig_convergence(
        "colsnp_uniform_l0",
        os.path.join(OUT_DIR, "snp_uniform_l0_convergence.png"),
    )
    print(f"figures written to {OUT_DIR}")
    print_tables(rows)


if __name__ == "__main__":
    main()
