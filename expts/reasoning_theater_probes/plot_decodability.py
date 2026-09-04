"""Plot per-rollout and aggregate answer-decodability curves from token_eval JSON.

Run:
    uv run python -m expts.reasoning_theater_probes.plot_decodability \
        --eval results/reasoning_theater_probes/qwen3_8b/gpqa_filtered/layer18/token_eval_linear_model_answer.json \
        --history results/reasoning_theater_probes/qwen3_8b/gpqa_filtered/layer18/history_linear_model_answer.json \
        --output notes/images/reasoning_theater_probes/decodability_layer18.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="token_eval_*.json from evaluate_probe.py")
    parser.add_argument("--history", required=True, help="history_*.json from train_probe.py")
    parser.add_argument("--output", required=True, help="output .png path")
    parser.add_argument("--n_bins", type=int, default=40)
    args = parser.parse_args()

    rows = json.loads(Path(args.eval).read_text())
    history = json.loads(Path(args.history).read_text())

    by_rollout: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_rollout[(r["prompt_id"], r["rollout_id"])].append(r)
    for v in by_rollout.values():
        v.sort(key=lambda r: r["t"])

    bin_edges = np.linspace(0.0, 1.0, args.n_bins + 1)
    bin_correct = np.zeros(args.n_bins, dtype=float)
    bin_count = np.zeros(args.n_bins, dtype=int)
    for r in rows:
        b = min(int(r["frac_t"] * args.n_bins), args.n_bins - 1)
        bin_correct[b] += int(r["pred_label"] == r["true_label"])
        bin_count[b] += 1
    bin_acc = bin_correct / np.maximum(bin_count, 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for (_pid, _rid), traj in by_rollout.items():
        fracs = [r["frac_t"] for r in traj]
        probs = [r["prob_true_label"] for r in traj]
        ax1.plot(fracs, probs, alpha=0.25, linewidth=0.8)
    ax1.set_xlabel("fraction of CoT consumed (frac_t)")
    ax1.set_ylabel("P(probe predicts model's final answer)")
    ax1.set_title("per-rollout decodability of model's final answer")
    ax1.set_ylim(0, 1.02)
    ax1.axhline(0.25, linestyle="--", color="grey", linewidth=0.7, label="random (4-way)")
    ax1.legend(loc="lower right")

    ax2.bar(bin_centers, bin_acc, width=1.0 / args.n_bins, color="C0", alpha=0.7)
    ax2.axhline(0.25, linestyle="--", color="grey", linewidth=0.7)
    ax2.set_xlabel("fraction of CoT consumed (frac_t)")
    ax2.set_ylabel("probe top-1 accuracy")
    ax2.set_title(
        f"aggregate accuracy across rollouts | best_test_acc (random truncation) = {history['best_test_acc']:.3f}"
    )
    ax2.set_ylim(0, 1.02)

    fig.suptitle(
        f"Reasoning Theater linear probe — layer {history['config']['layer']}, "
        f"label={history['config']['label_type']}, "
        f"{Path(history['config']['data_path']).stem}",
        y=1.02,
    )
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    print(
        f"  rollouts plotted: {len(by_rollout)}  "
        f"total points: {len(rows)}  "
        f"final-bin acc: {bin_acc[-1]:.3f}"
    )


if __name__ == "__main__":
    main()
