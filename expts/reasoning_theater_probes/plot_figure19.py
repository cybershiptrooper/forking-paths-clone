"""Figure 19-style plot: probe accuracy vs. relative CoT position, multiple probes overlaid.

Usage:
    uv run python -m expts.reasoning_theater_probes.plot_figure19 \
        --runs <label>:<eval_json>[:<history_json>] [<label>:<eval_json>...] \
        --output <out.png> \
        --n_bins 20

Each --runs entry is a colon-separated triple (history_json optional).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"--runs entry must be label:eval[:history], got {spec!r}")
    label, eval_path = parts[0], parts[1]
    history_path = parts[2] if len(parts) == 3 else None
    rows = json.loads(Path(eval_path).read_text())
    history = json.loads(Path(history_path).read_text()) if history_path else None
    return {"label": label, "rows": rows, "history": history}


def bin_accuracy(rows: list[dict], n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_correct = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)
    for r in rows:
        b = min(int(r["frac_t"] * n_bins), n_bins - 1)
        bin_correct[b] += int(r["pred_label"] == r["true_label"])
        bin_count[b] += 1
    bin_acc = bin_correct / np.maximum(bin_count, 1)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    return bin_centers, bin_acc, bin_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more label:eval_json[:history_json] triples",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument(
        "--random_baseline",
        type=float,
        default=0.25,
        help="Horizontal reference line; default 0.25 for 4-way.",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    runs = [load_run(s) for s in args.runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, run in enumerate(runs):
        centers, acc, counts = bin_accuracy(run["rows"], args.n_bins)
        ax.plot(
            centers * 100,
            acc,
            marker="o",
            color=colors[i % len(colors)],
            label=run["label"],
            linewidth=2,
        )

    ax.axhline(
        args.random_baseline,
        linestyle="--",
        color="grey",
        linewidth=0.8,
        label=f"random ({args.random_baseline})",
    )
    ax.set_xlabel("Relative Position (%)")
    ax.set_ylabel("Accuracy (probe predicts model's final answer)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title("Early Decoding Accuracy — probes on Qwen3-8B GPQA-Diamond")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    for run in runs:
        n_rollouts = len({(r["prompt_id"], r["rollout_id"]) for r in run["rows"]})
        if run["history"]:
            best = run["history"]["best_test_acc"]
            print(f"  {run['label']}: rollouts={n_rollouts}  best_test_acc={best:.3f}")
        else:
            print(f"  {run['label']}: rollouts={n_rollouts}")


if __name__ == "__main__":
    main()
