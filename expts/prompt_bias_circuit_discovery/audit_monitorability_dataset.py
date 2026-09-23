"""Read-only audit of the existing monitorability dataset selections.

Reproduce with:
    python -m expts.prompt_bias_circuit_discovery.audit_monitorability_dataset

The audit does not assign individual-rollout causal labels or change existing
selections. Interval checks use the independent-proportions Wilson-Newcombe
interval for the intervention-minus-control rate difference. A 90% interval
inside +/-0.1 is the usual two-one-sided-test equivalence criterion at 5%.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def wilson_interval(p: float, n: int, z: float) -> tuple[float, float]:
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - radius, center + radius


def newcombe_interval(p1: float, p0: float, n1: int, n0: int,
                      confidence: float = 0.9) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    lo1, hi1 = wilson_interval(p1, n1, z)
    lo0, hi0 = wilson_interval(p0, n0, z)
    delta = p1 - p0
    return (delta - math.hypot(p1 - lo1, hi0 - p0),
            delta + math.hypot(hi1 - p1, p0 - lo0))


def input_id(row: dict) -> str:
    return row["uid"].split("_")[2]


def summarize_rates(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    effects = [(r["p1"] - r["p0"]) / r["p1"] for r in rows if r["p1"] > 0]
    return {"n": len(rows), "mean_p0": statistics.mean(r["p0"] for r in rows),
            "mean_p1": statistics.mean(r["p1"] for r in rows),
            "median_attributable_lower_bound": statistics.median(effects) if effects else None,
            "p0_le": {str(c): sum(r["p0"] <= c for r in rows) for c in (0.15, 0.25, 0.35)},
            "gender": dict(collections.Counter(r.get("gender", "unspecified") for r in rows))}


def plot_audit(black: dict, positive: list[dict], null: list[dict],
               summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17,
                         "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, pos, neg, title in (
            (axes[0, 0], black["positives"], black["negatives"], "Admission: selected Black/White contrast"),
            (axes[0, 1], positive, null, "Sarcasm: suggest sincere")):
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=.5, linewidth=1)
        ax.scatter([r["p0"] for r in neg], [r["p1"] for r in neg],
                   color="#777777", alpha=.7, s=38, label=f"Low-effect candidates (n={len(neg)})")
        ax.scatter([r["p0"] for r in pos], [r["p1"] for r in pos],
                   color="#0072B2", marker="^", s=55, label=f"Effect-selected inputs (n={len(pos)})")
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xlabel="Control rate p₀",
               ylabel="Intervention rate p₁", title=title)
        ax.legend(loc="lower right")
        ax.grid(alpha=.15)

    ax = axes[1, 0]
    native = summary["nativeamerican_after_excluding_all_black179_inputs"]
    names = ["Effect-selected", "Low-effect candidates"]
    disjoint = [native[c]["n"] for c in ("positives", "negatives")]
    overlap = [native[c]["n_overlap_black179"] for c in ("positives", "negatives")]
    ax.bar(names, disjoint, color="#0072B2", label="No shared base input")
    ax.bar(names, overlap, bottom=disjoint, color="#E69F00", label="Shares a Black-set input")
    for i, (d, o) in enumerate(zip(disjoint, overlap)):
        ax.text(i, d/2, str(d), ha="center", va="center", color="white", fontsize=17)
        ax.text(i, d+o/2, str(o), ha="center", va="center", fontsize=17)
    ax.set(ylabel="Number of Native American inputs", title="Near-OOD input overlap", ylim=(0, 66))
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    intervals = sorted(summary["null_equivalence"]["intervals"], key=lambda r: r["p1"]-r["p0"])
    # Show every tenth interval and both endpoints, selected without its width.
    selected = sorted(set(range(0, len(intervals), 10)) | {len(intervals)-1})
    for y, idx in enumerate(selected):
        row = intervals[idx]
        delta = row["p1"]-row["p0"]
        ax.errorbar(delta, y, xerr=[[delta-row["lower_90"]], [row["upper_90"]-delta]],
                    fmt="o", color="#777777", capsize=3, markersize=5)
    ax.axvspan(-.1, .1, color="#0072B2", alpha=.1)
    ax.axvline(-.1, color="#0072B2", linestyle="--")
    ax.axvline(.1, color="#0072B2", linestyle="--")
    ax.axvline(0, color="black", linewidth=1, alpha=.5)
    ax.set(yticks=range(len(selected)), yticklabels=[intervals[j]["input"] for j in selected],
           xlabel="Intervention minus control rate (90% interval)", ylabel="Selected low-effect admission inputs",
           title="0/116 establish equivalence within ±0.1")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/prompt_bias_v2"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/prompt_bias_v2/design_audit/summary.json"))
    ap.add_argument("--plot", type=Path,
                    default=Path("notes/images/monitorability_design_audit/dataset_checks.png"))
    args = ap.parse_args()
    paths = {
        "black": args.root / "analysis/admission_same_decision_dataset_selection.json",
        "nativeamerican": args.root / "analysis/admission_same_decision_selection_nativeamerican.json",
        "sarcasm": args.root / "sycophancy/sarcasm_qwen3_8b_switch_rates.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    black, native, sarcasm = (data[k] for k in ("black", "nativeamerican", "sarcasm"))
    black_ids = {input_id(r) for cls in ("positives", "negatives") for r in black[cls]}
    null_intervals = []
    for row in black["negatives"]:
        lo, hi = newcombe_interval(row["p1"], row["p0"], 64, 64)
        null_intervals.append({"input": input_id(row), "p1": row["p1"], "p0": row["p0"],
                               "lower_90": lo, "upper_90": hi,
                               "equivalent_within_0.1": lo > -0.1 and hi < 0.1})
    native_disjoint = {}
    for cls in ("positives", "negatives"):
        remaining = [r for r in native[cls] if input_id(r) not in black_ids]
        native_disjoint[cls] = {
            **summarize_rates(remaining), "n_original": len(native[cls]),
            "n_overlap_black179": len(native[cls]) - len(remaining),
            "inputs": [input_id(r) for r in remaining],
            "redacted_reference_labels": dict(collections.Counter(
                str(r.get("label_redacted_ref")) for r in remaining)),
        }
    sincere = [r for r in sarcasm if r["arm"] == "suggest_sincere"]
    sarcasm_positive = [r for r in sincere if r["testbed_label"] == "sycophantic"]
    sarcasm_null = [r for r in sincere if r["testbed_label"] == "non_sycophantic"]
    p1min, p1max = min(r["p1"] for r in sarcasm_positive), max(r["p1"] for r in sarcasm_positive)
    summary = {
        "source_files": {key: str(path) for key, path in paths.items()},
        "scope": "Existing selected samples only; exploratory, not an untouched test set.",
        "admission": {
            cls: summarize_rates(black[cls]) for cls in ("positives", "negatives")},
        "null_equivalence": {
            "method": "90% Wilson-Newcombe difference interval, independent binomial arms, n=64 each",
            "margin": 0.1, "n_tested": len(null_intervals),
            "n_equivalent": sum(r["equivalent_within_0.1"] for r in null_intervals),
            "intervals": null_intervals,
            "interpretation": "Approximate marginal equivalence is not absence of individual causal effects; no multiplicity correction is applied.",
        },
        "nativeamerican_after_excluding_all_black179_inputs": native_disjoint,
        "sarcasm_suggest_sincere": {
            "strict_positives": summarize_rates(sarcasm_positive),
            "low_effect_candidates": summarize_rates(sarcasm_null),
            "positive_p1_range": [p1min, p1max],
            "n_low_effect_candidates_within_positive_p1_range": sum(
                p1min <= r["p1"] <= p1max for r in sarcasm_null),
            "n_low_effect_candidates_with_some_matching_answer": sum(r["p1"] > 0 for r in sarcasm_null),
            "n_positive_point_effect": sum(r["switch_rate"] > 0 for r in sincere),
            "n_point_effect_above_0.30": sum(r["switch_rate"] > 0.30 for r in sincere),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    plot_audit(black, sarcasm_positive, sarcasm_null, summary, args.plot)
    print(json.dumps({"out": str(args.out), "admission": summary["admission"],
                      "n_nulls_equivalent_at_90pct": summary["null_equivalence"]["n_equivalent"],
                      "nativeamerican_disjoint": {cls: native_disjoint[cls]["n"] for cls in native_disjoint},
                      "sarcasm": summary["sarcasm_suggest_sincere"]}, indent=2))


if __name__ == "__main__":
    main()
