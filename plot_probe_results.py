import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(results_path: str) -> dict:
    """Load probe results from a JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_entropy_comparison(
    predictions: list, output_dir: str, question_ids: list = None
):
    """
    Plot true vs predicted entropy for each question.

    Args:
        predictions: List of prediction dictionaries from results file
        output_dir: Directory to save plots
        question_ids: Optional list of specific question IDs to plot. If None, plots all.
    """
    os.makedirs(output_dir, exist_ok=True)

    for pred in predictions:
        q_id = pred["question_id"]

        # Skip if not in the requested question IDs
        if question_ids is not None and q_id not in question_ids:
            continue

        timesteps = pred["t"]
        try:
            true_entropy = pred["true_entropy"]
            pred_entropy = pred["pred_entropy"]
        except KeyError:
            true_entropy = pred["true_label"]
            pred_entropy = pred["pred_label"]

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: True vs Predicted Entropy
        ax1 = axes[0]
        ax1.plot(
            timesteps,
            true_entropy,
            "b-",
            label="True Entropy",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax1.plot(
            timesteps,
            pred_entropy,
            "r--",
            label="Predicted Entropy",
            linewidth=2,
            marker="x",
            markersize=3,
        )
        ax1.set_ylabel("Entropy", fontsize=12)
        ax1.set_title(f"Question {q_id}: True vs Predicted Entropy", fontsize=14)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Difference (True - Predicted)
        ax2 = axes[1]
        difference = np.array(true_entropy) - np.array(pred_entropy)
        ax2.plot(timesteps, difference, "g-", linewidth=2, marker="s", markersize=3)
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Timestep (token position)", fontsize=12)
        ax2.set_ylabel("Difference (True - Pred)", fontsize=12)
        ax2.set_title(f"Question {q_id}: Prediction Error", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Fill regions where prediction is too high (negative diff) vs too low (positive diff)
        ax2.fill_between(
            timesteps,
            difference,
            0,
            where=(difference > 0),
            color="blue",
            alpha=0.2,
            label="Underestimate",
        )
        ax2.fill_between(
            timesteps,
            difference,
            0,
            where=(difference < 0),
            color="red",
            alpha=0.2,
            label="Overestimate",
        )
        ax2.legend(loc="best")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/question_{q_id:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved individual question plots to {output_dir}/")


def plot_aggregate_stats(predictions: list, output_dir: str):
    """
    Plot aggregate statistics across all questions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all errors
    all_errors = []
    all_abs_errors = []
    all_true = []
    all_pred = []

    for pred in predictions:
        true_entropy = np.array(pred["true_entropy"])
        pred_entropy = np.array(pred["pred_entropy"])
        errors = true_entropy - pred_entropy

        all_errors.extend(errors.tolist())
        all_abs_errors.extend(np.abs(errors).tolist())
        all_true.extend(true_entropy.tolist())
        all_pred.extend(pred_entropy.tolist())

    all_errors = np.array(all_errors)
    all_abs_errors = np.array(all_abs_errors)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Scatter plot of true vs predicted
    ax1 = axes[0, 0]
    ax1.scatter(all_true, all_pred, alpha=0.3, s=10)
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax1.set_xlabel("True Entropy", fontsize=12)
    ax1.set_ylabel("Predicted Entropy", fontsize=12)
    ax1.set_title("True vs Predicted Entropy (all timesteps)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error histogram
    ax2 = axes[0, 1]
    ax2.hist(all_errors, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax2.axvline(
        x=np.mean(all_errors),
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(all_errors):.4f}",
    )
    ax2.set_xlabel("Error (True - Predicted)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Error Distribution", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Absolute error histogram
    ax3 = axes[1, 0]
    ax3.hist(all_abs_errors, bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax3.axvline(
        x=np.mean(all_abs_errors),
        color="r",
        linestyle="-",
        linewidth=2,
        label=f"MAE: {np.mean(all_abs_errors):.4f}",
    )
    ax3.axvline(
        x=np.median(all_abs_errors),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_abs_errors):.4f}",
    )
    ax3.set_xlabel("Absolute Error", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Absolute Error Distribution", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error vs True Entropy (to see if error depends on entropy level)
    ax4 = axes[1, 1]
    ax4.scatter(all_true, all_errors, alpha=0.3, s=10)
    ax4.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax4.set_xlabel("True Entropy", fontsize=12)
    ax4.set_ylabel("Error (True - Predicted)", fontsize=12)
    ax4.set_title("Error vs True Entropy", fontsize=14)
    ax4.grid(True, alpha=0.3)

    # Add summary statistics as text
    stats_text = (
        f"Statistics:\n"
        f"  MSE: {np.mean(all_errors**2):.4f}\n"
        f"  RMSE: {np.sqrt(np.mean(all_errors**2)):.4f}\n"
        f"  MAE: {np.mean(all_abs_errors):.4f}\n"
        f"  Mean Error: {np.mean(all_errors):.4f}\n"
        f"  Std Error: {np.std(all_errors):.4f}\n"
        f"  N samples: {len(all_errors)}"
    )
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_dir}/aggregate_stats.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved aggregate statistics plot to {output_dir}/aggregate_stats.png")

    # Print summary to console
    print("\n" + "=" * 50)
    print("PROBE PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  MSE:        {np.mean(all_errors**2):.4f}")
    print(f"  RMSE:       {np.sqrt(np.mean(all_errors**2)):.4f}")
    print(f"  MAE:        {np.mean(all_abs_errors):.4f}")
    print(f"  Mean Error: {np.mean(all_errors):.4f}")
    print(f"  Std Error:  {np.std(all_errors):.4f}")
    print(f"  N samples:  {len(all_errors)}")
    print("=" * 50)


def main(
    results_path: str,
    output_dir: str = None,
    question_ids: list = None,
    plot_individual: bool = True,
    plot_aggregate: bool = True,
):
    """
    Main function to plot probe results.

    Args:
        results_path: Path to the results JSON file
        output_dir: Directory to save plots (defaults to same folder as results)
        question_ids: List of specific question IDs to plot (None = all)
        plot_individual: Whether to plot individual question plots
        plot_aggregate: Whether to plot aggregate statistics
    """
    # Load results
    results = load_results(results_path)
    predictions = results["predictions"]
    hyperparams = results["hyperparameters"]

    print(f"Loaded results from: {results_path}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Number of questions: {len(predictions)}")

    # Set output directory
    if output_dir is None:
        output_dir = str(Path(results_path).parent / "plots")

    # Plot individual questions
    if plot_individual:
        individual_dir = os.path.join(output_dir, "individual")
        plot_entropy_comparison(predictions, individual_dir, question_ids)

    # Plot aggregate statistics
    if plot_aggregate:
        plot_aggregate_stats(predictions, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot probe results from probing experiments"
    )
    parser.add_argument(
        "--results_path",
        "-r",
        type=str,
        required=True,
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save plots (defaults to plots/ in same folder as results)",
    )
    parser.add_argument(
        "--question_ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific question IDs to plot (default: all)",
    )
    parser.add_argument(
        "--no_individual", action="store_true", help="Skip individual question plots"
    )
    parser.add_argument(
        "--no_aggregate", action="store_true", help="Skip aggregate statistics plot"
    )

    args = parser.parse_args()

    main(
        results_path=args.results_path,
        output_dir=args.output_dir,
        question_ids=args.question_ids,
        plot_individual=not args.no_individual,
        plot_aggregate=not args.no_aggregate,
    )
