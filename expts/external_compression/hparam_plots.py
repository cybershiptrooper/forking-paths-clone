"""Plots for the column SNP hyperparameter search (preliminary or full).

Produces, under notes/images/individual_sentence_grading_followups/:

- hparam_training_curves.png — the four panels of the spike scan
  (notes/l0_hinge_spike_scan.md): task loss, logged L0 hinge, achieved
  sparsity vs target, and net fraction of gates changing clamped status
  per logged interval; old-schedule run vs a clean and a spiky new config.
- hparam_kl_vs_sparsity.png — answer KL at M=15 of every search config vs
  its achieved sparsity, against the same instance's Suppress / oracle /
  old-schedule column SNP evals (the "any improvement at all?" check).

Also prints the spike classification per config using the scan's
λ-independent "concentrated jump" marker (>=80% of sparsity progress in a
10%-of-training window and absolute jump > 0.15) — the hinge-peak marker
is biased across λ regimes because the logged hinge is λ-scaled.

Usage:  uv run python -m expts.external_compression.hparam_plots
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expts.external_compression.common import REPO_ROOT, RESULTS_DIR

SEARCH_DIR = os.path.join(RESULTS_DIR, "hparam_search")
SCORES_DIR = os.path.join(RESULTS_DIR, "scores")
EVALS_DIR = os.path.join(RESULTS_DIR, "evals")
OUT_DIR = os.path.join(
    REPO_ROOT, "notes", "images", "individual_sentence_grading_followups"
)
INSTANCES = [
    "gpqa_gpqa_diamond_0002_pl50",
    "bigbench_logical_deduction_0000_pl85",
    # Tail instances (original column SNP failed worst here; added for
    # the hc=8 validation round):
    "gpqa_gpqa_diamond_0009_pl50",
    "gpqa_gpqa_diamond_0001_pl100",
]
INIT_COLORS = {  # categorical palette slots, fixed by entity
    "closed": "#2a78d6", "half": "#1baf7a", "open": "#eda100",
    "random": "#e87ba4",
}
LAMBDA_MARKERS = {3: "o", 10: "s", 30: "^"}
FLOOR = 1e-5


def load_metrics(path):
    return [json.loads(l) for l in open(path)]


def concentrated_jump(steps) -> bool:
    """Scan-note marker: >=80% of sparsity progress inside a window of 10%
    of training, with absolute jump > 0.15 (lambda-independent)."""
    sp = [s["sparsity"] for s in steps]
    total = sp[-1] - sp[0]
    if abs(total) < 1e-9:
        return False
    w = max(1, len(sp) // 10)
    best = max(sp[i + w] - sp[i] for i in range(len(sp) - w))
    return best > 0.15 and best / total >= 0.8


def fig_training_curves(path):
    inst = "bigbench_logical_deduction_0000_pl85"
    runs = [
        ("old schedule, init open (original)",
         os.path.join(SCORES_DIR, inst, "colsnp_logs", "training_metrics.jsonl"),
         "#52514e", "-"),
        ("new: lr0.03 λ10 init random",
         os.path.join(SEARCH_DIR, inst, "lr0.03_l10_init-random_hc1",
                      "colsnp_search_logs", "training_metrics.jsonl"),
         "#e87ba4", "-"),
        ("new: lr0.3 λ10 init closed (still spiky)",
         os.path.join(SEARCH_DIR, inst, "lr0.3_l10_init-closed_hc1",
                      "colsnp_search_logs", "training_metrics.jsonl"),
         "#2a78d6", "--"),
    ]
    # target sparsity from any summary of this instance
    summaries = glob.glob(os.path.join(SEARCH_DIR, inst, "*", "summary.json"))
    target = json.load(open(summaries[0]))["target_sparsity"] if summaries else None

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), facecolor="white")
    for label, p, color, ls in runs:
        if not os.path.exists(p):
            continue
        steps = load_metrics(p)
        xs = [s["step"] for s in steps]
        axes[0].plot(xs, [max(s["task_loss"], FLOOR) for s in steps],
                     ls, color=color, linewidth=1.6, label=label)
        axes[1].plot(xs, [s["l0_loss"] for s in steps],
                     ls, color=color, linewidth=1.6)
        axes[2].plot(xs, [s["sparsity"] for s in steps],
                     ls, color=color, linewidth=1.6)
        sp = [s["sparsity"] for s in steps]
        axes[3].plot(xs[1:], [abs(b - a) for a, b in zip(sp, sp[1:])],
                     ls, color=color, linewidth=1.6)
    if target is not None:
        axes[2].axhline(target, color="#c9c8c4", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    titles = [
        "Task loss (answer-probe KL)", "Logged L0 term (λ × hinge)",
        "Achieved sparsity (dashed: target)",
        "Net gate status change / interval",
    ]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(True, which="major", axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def baseline_kl(inst, method, M=15):
    p = os.path.join(EVALS_DIR, inst, f"{method}.json")
    if not os.path.exists(p):
        return None
    for r in json.load(open(p))["rows"]:
        if r.get("M") == M:
            return r["answer_kl"]
    return None


OUTCOME_FREE = ["ta", "attn_last", "attn_next"]
OUTCOME_FREE_LABELS = {
    "ta": "TA", "attn_last": "attention (last prefix sentence)",
    "attn_next": "next-sentence attention",
}


def fig_kl_vs_sparsity(path):
    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.2), facecolor="white",
        gridspec_kw={"wspace": 0.5, "hspace": 0.35},
    )
    for ax, inst in zip(axes.flat, INSTANCES):
        summaries = sorted(glob.glob(
            os.path.join(SEARCH_DIR, inst, "*", "summary.json")))
        target = None
        for p in summaries:
            s = json.load(open(p))
            target = s["target_sparsity"]
            kl = s["evals"].get("answer_kl_M15")
            if kl is None:
                continue
            init = s["tag"].split("init-")[1].split("_")[0]
            lam = float(s["hparams"]["l0_lambda"])
            metrics_p = os.path.join(
                os.path.dirname(p), "colsnp_search_logs",
                "training_metrics.jsonl")
            spiky = (concentrated_jump(load_metrics(metrics_p))
                     if os.path.exists(metrics_p) else False)
            hc8 = s["tag"].endswith("hc8")
            ax.scatter(
                s["achieved_sparsity_final"], max(kl, FLOOR),
                c=INIT_COLORS[init], marker=LAMBDA_MARKERS.get(lam, "o"),
                s=110 if hc8 else 60, alpha=0.9,
                edgecolors="#b91c1c" if spiky else ("#0b0b0b" if hc8 else "white"),
                linewidths=1.6, zorder=3,
            )
        # Reference lines, labeled once at the right edge, outside the axes.
        refs = [
            ("suppress", "Suppress", "#eb6834", "-"),
            ("colsnp", "column SNP\n(original settings)", "#2a78d6", "-"),
            ("oracle", "single sentence oracle\n(uses outcome per candidate)",
             "#008300", ":"),
        ]
        free_vals = {m: baseline_kl(inst, m) for m in OUTCOME_FREE}
        free_vals = {m: v for m, v in free_vals.items() if v is not None}
        if free_vals:
            best_m = min(free_vals, key=free_vals.get)
            refs.append((
                None, f"best outcome-free:\n{OUTCOME_FREE_LABELS[best_m]}",
                "#52514e", "-."))
            best_free = free_vals[best_m]
        import math
        placed = []
        ref_lines = []
        for method, label, color, ls in refs:
            kl = best_free if method is None else baseline_kl(inst, method)
            if kl is not None:
                ref_lines.append((max(kl, FLOOR), label, color, ls))
        for y, label, color, ls in sorted(ref_lines):
            ax.axhline(y, color=color, linewidth=1.4, linestyle=ls, alpha=0.9)
            va = "center"
            if any(abs(math.log10(y) - math.log10(py)) < 0.12 for py in placed):
                va = "bottom"  # nudge the second of two close labels upward
            placed.append(y)
            ax.annotate(
                label, xy=(1.01, y), xycoords=("axes fraction", "data"),
                fontsize=7, color=color, va=va, ha="left",
            )
        if target is not None:
            ax.axvline(target, color="#c9c8c4", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("achieved column sparsity (vertical dashed: target)")
        n_rank = json.load(open(summaries[0]))["num_rankable"] if summaries else "?"
        ax.set_title(
            inst.replace("gpqa_gpqa_diamond_", "gpqa_")
            .replace("bigbench_logical_deduction_", "bigbench_")
            + f"  ({n_rank} rankable)",
            fontsize=10,
        )
        ax.grid(True, which="major", axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("answer KL at M = 15 (log)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", linestyle="",
                      markersize=8, label=f"init {k}")
               for k, c in INIT_COLORS.items()]
    handles += [Line2D([], [], color="#52514e", marker=m, linestyle="",
                       markersize=8, label=f"λ = {l:g}")
                for l, m in LAMBDA_MARKERS.items()]
    handles.append(Line2D([], [], color="white", marker="o", linestyle="",
                          markeredgecolor="#b91c1c", markersize=8,
                          markeredgewidth=1.6,
                          label="concentrated jump (spiky training)"))
    handles.append(Line2D([], [], color="#c3c2b7", marker="o", linestyle="",
                          markeredgecolor="#0b0b0b", markersize=11,
                          markeredgewidth=1.6, label="hc = 8 (large, dark edge)"))
    fig.legend(handles=handles, fontsize=8, frameon=False,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def print_classification():
    n_all = n_spiky = 0
    for inst in INSTANCES:
        for p in sorted(glob.glob(os.path.join(
                SEARCH_DIR, inst, "*", "colsnp_search_logs",
                "training_metrics.jsonl"))):
            n_all += 1
            if concentrated_jump(load_metrics(p)):
                tag = p.split("/")[-3]
                print(f"  concentrated jump: {inst} {tag}")
                n_spiky += 1
    print(f"concentrated-jump marker: {n_spiky}/{n_all} search runs "
          f"(scan note: 27/29 of the original convergence-probe runs)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_training_curves(os.path.join(OUT_DIR, "hparam_training_curves.png"))
    fig_kl_vs_sparsity(os.path.join(OUT_DIR, "hparam_kl_vs_sparsity.png"))
    print_classification()


if __name__ == "__main__":
    main()
