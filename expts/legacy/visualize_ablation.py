"""
Visualize attention ablation experiment results.

Loads JSON files from attention ablation experiments and creates visualizations
showing how answer changes vary with number of sentences ablated, offset from
convergence, and selection method (top-k vs random).
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import griddata

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, contour plots will use simpler interpolation")


def load_ablation_results(results_dir: str, example_index: int) -> List[Dict]:
    """
    Load all ablation result JSON files for a given example.

    Args:
        results_dir: Directory containing ablation results
        example_index: Example index to load

    Returns:
        List of result dictionaries
    """
    idx_str = str(example_index).zfill(2)
    pattern = os.path.join(results_dir, f"ablation_example_{idx_str}_*.json")
    files = glob.glob(pattern)

    results = []
    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)
            results.append(data)

    return results


def organize_results_by_offset(
    results: List[Dict],
) -> Dict[int, Dict[str, Dict[int, Dict]]]:
    """
    Organize results by offset, then by random_sentences, then by num_sentences.

    Returns:
        Dict[offset, Dict[random_sentences, Dict[num_sentences, result]]]
    """
    organized = defaultdict(lambda: defaultdict(dict))

    for result in results:
        offset = result["offset_from_convergence"]
        is_random = result["random_sentences"]
        num_sentences = result["num_sentences_to_ablate"]
        key = "random" if is_random else "top_k"

        organized[offset][key][num_sentences] = result

    return organized


def calculate_flip_percentage(result: Dict) -> float:
    """Calculate the percentage of answers that flipped."""
    counts = result["answer_changed_counts"]
    total = counts["changed"] + counts["unchanged"]
    if total == 0:
        return 0.0
    return (counts["changed"] / total) * 100.0


def plot_by_timestep(
    organized_results: Dict[int, Dict[str, Dict[int, Dict]]],
    output_dir: str,
    example_index: int,
):
    """
    Create one plot for each timestep (offset).

    For each offset, plot:
    - x axis: number of sentences
    - y axis: probabilities (percent of answers flipped)
    - one line for top_k, one line for random
    """
    offsets = sorted(organized_results.keys())
    idx_str = str(example_index).zfill(2)

    for offset in offsets:
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in ["top_k", "random"]:
            if method not in organized_results[offset]:
                continue

            data = organized_results[offset][method]
            num_sentences_list = sorted(data.keys())
            flip_percentages = [
                calculate_flip_percentage(data[ns]) for ns in num_sentences_list
            ]

            label = "Top-K" if method == "top_k" else "Random"
            ax.plot(
                num_sentences_list,
                flip_percentages,
                marker="o",
                label=label,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Number of Sentences Ablated", fontsize=12)
        ax.set_ylabel("Percent of Answers Flipped (%)", fontsize=12)
        ax.set_title(
            f"Answer Flip Rate vs. Number of Sentences\n(Offset from Convergence: {offset} tokens)",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"ablation_timestep_{idx_str}_offset{offset}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_file}")


def plot_heatmap_topk(
    organized_results: Dict[int, Dict[str, Dict[int, Dict]]],
    output_dir: str,
    example_index: int,
):
    """
    Create a heatmap for top-k only.

    - x axis: offset from convergence (time relative to convergence)
    - y axis: number of sentences ablated
    - pixel strength: percent of answers flipped
    """
    idx_str = str(example_index).zfill(2)

    # Collect all offsets and num_sentences values
    offsets = sorted(organized_results.keys())
    all_num_sentences = set()

    for offset in offsets:
        if "top_k" in organized_results[offset]:
            all_num_sentences.update(organized_results[offset]["top_k"].keys())

    num_sentences_list = sorted(all_num_sentences)

    # Build heatmap data
    heatmap_data = np.zeros((len(num_sentences_list), len(offsets)))

    for i, num_sentences in enumerate(num_sentences_list):
        for j, offset in enumerate(offsets):
            if "top_k" in organized_results[offset]:
                if num_sentences in organized_results[offset]["top_k"]:
                    result = organized_results[offset]["top_k"][num_sentences]
                    heatmap_data[i, j] = calculate_flip_percentage(result)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
        vmax=100,
    )

    # Set ticks and labels
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels(offsets)
    ax.set_yticks(range(len(num_sentences_list)))
    ax.set_yticklabels(num_sentences_list)

    ax.set_xlabel("Offset from Convergence (tokens)", fontsize=12)
    ax.set_ylabel("Number of Sentences Ablated", fontsize=12)
    ax.set_title(
        "Answer Flip Rate Heatmap (Top-K Sentences)\nPercent of Answers Flipped",
        fontsize=14,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percent of Answers Flipped (%)", fontsize=11)

    # Add text annotations for better readability
    for i in range(len(num_sentences_list)):
        for j in range(len(offsets)):
            text = ax.text(
                j,
                i,
                f"{heatmap_data[i, j]:.0f}",
                ha="center",
                va="center",
                color="black" if heatmap_data[i, j] < 50 else "white",
                fontsize=8,
            )

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"ablation_heatmap_topk_{idx_str}.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}")


def find_min_sentences_for_flip_percentage(
    organized_results: Dict[int, Dict[str, Dict[int, Dict]]],
    method: str,
    target_percentages: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each offset and target percentage, find the minimum number of sentences
    needed to flip that percentage of answers.

    Returns:
        Tuple of (offsets_array, percentages_array, min_sentences_array)
        where arrays are flattened for contour plotting
    """
    offsets = sorted(organized_results.keys())

    x_vals = []
    y_vals = []
    z_vals = []

    for offset in offsets:
        if method not in organized_results[offset]:
            continue

        data = organized_results[offset][method]
        num_sentences_list = sorted(data.keys())

        for target in target_percentages:
            min_sentences = None
            for num_sentences in num_sentences_list:
                flip_pct = calculate_flip_percentage(data[num_sentences])
                if flip_pct >= target:
                    min_sentences = num_sentences
                    break

            if min_sentences is not None:
                x_vals.append(offset)
                y_vals.append(target)
                z_vals.append(min_sentences)

    return np.array(x_vals), np.array(y_vals), np.array(z_vals)


def plot_contour_side_by_side(
    organized_results: Dict[int, Dict[str, Dict[int, Dict]]],
    output_dir: str,
    example_index: int,
    target_percentages: List[float] = [10, 20, 30, 40, 50, 60, 70, 80, 90],
):
    """
    Create side-by-side contour plots for top-k and random.

    - x axis: offset from convergence
    - y axis: target percentage of answers to flip
    - z (contour): minimal number of sentences needed to flip that percentage
    """
    idx_str = str(example_index).zfill(2)

    # Calculate data for both methods
    x_topk, y_topk, z_topk = find_min_sentences_for_flip_percentage(
        organized_results, "top_k", target_percentages
    )
    x_random, y_random, z_random = find_min_sentences_for_flip_percentage(
        organized_results, "random", target_percentages
    )

    # Get ranges for grid
    offsets = sorted(organized_results.keys())
    if not offsets:
        print("No offsets found, skipping contour plot")
        return

    x_min, x_max = min(offsets), max(offsets)
    y_min, y_max = min(target_percentages), max(target_percentages)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot top-k contour
    if len(x_topk) > 0:
        # Create grid for interpolation
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate z values
        if len(x_topk) >= 3 and HAS_SCIPY:  # Need at least 3 points for interpolation
            Zi = griddata(
                (x_topk, y_topk),
                z_topk,
                (Xi, Yi),
                method="linear",
                fill_value=np.nan,
            )

            # Create contour plot
            contour = ax1.contourf(Xi, Yi, Zi, levels=15, cmap="viridis", alpha=0.8)
            ax1.contour(Xi, Yi, Zi, levels=15, colors="black", alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=ax1, label="Minimal Sentences Needed")

            # Overlay data points
            scatter = ax1.scatter(
                x_topk,
                y_topk,
                c=z_topk,
                s=50,
                edgecolors="black",
                linewidths=1,
                cmap="viridis",
            )
        else:
            # If not enough points or no scipy, just scatter plot
            scatter = ax1.scatter(
                x_topk,
                y_topk,
                c=z_topk,
                s=100,
                edgecolors="black",
                linewidths=1,
                cmap="viridis",
            )
            plt.colorbar(scatter, ax=ax1, label="Minimal Sentences Needed")

    ax1.set_xlabel("Offset from Convergence (tokens)", fontsize=12)
    ax1.set_ylabel("Target Flip Percentage (%)", fontsize=12)
    ax1.set_title("Top-K Sentences", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot random contour
    if len(x_random) > 0:
        # Create grid for interpolation
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate z values
        if len(x_random) >= 3 and HAS_SCIPY:  # Need at least 3 points for interpolation
            Zi = griddata(
                (x_random, y_random),
                z_random,
                (Xi, Yi),
                method="linear",
                fill_value=np.nan,
            )

            # Create contour plot
            contour = ax2.contourf(Xi, Yi, Zi, levels=15, cmap="viridis", alpha=0.8)
            ax2.contour(Xi, Yi, Zi, levels=15, colors="black", alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=ax2, label="Minimal Sentences Needed")

            # Overlay data points
            scatter = ax2.scatter(
                x_random,
                y_random,
                c=z_random,
                s=50,
                edgecolors="black",
                linewidths=1,
                cmap="viridis",
            )
        else:
            # If not enough points or no scipy, just scatter plot
            scatter = ax2.scatter(
                x_random,
                y_random,
                c=z_random,
                s=100,
                edgecolors="black",
                linewidths=1,
                cmap="viridis",
            )
            plt.colorbar(scatter, ax=ax2, label="Minimal Sentences Needed")

    ax2.set_xlabel("Offset from Convergence (tokens)", fontsize=12)
    ax2.set_ylabel("Target Flip Percentage (%)", fontsize=12)
    ax2.set_title("Random Sentences", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Minimal Sentences Needed to Flip Answers\nby Offset and Target Percentage",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"ablation_contour_{idx_str}.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_file}")


def main(
    example_index: int = 0,
    results_dir: str = "results/attention_ablation",
    output_dir: str = "results/attention_ablation_plots",
):
    """
    Main function to load and visualize ablation results.

    Args:
        example_index: Example index to visualize
        results_dir: Directory containing ablation result JSON files
        output_dir: Directory to save plots
    """
    print(f"Loading ablation results for example {example_index}...")
    results = load_ablation_results(results_dir, example_index)

    if not results:
        print(f"No results found for example {example_index}")
        return

    print(f"Loaded {len(results)} result files")

    # Organize results
    organized_results = organize_results_by_offset(results)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    print("1. Plotting by timestep (offset)...")
    plot_by_timestep(organized_results, output_dir, example_index)

    print("2. Plotting heatmap for top-k...")
    plot_heatmap_topk(organized_results, output_dir, example_index)

    print("3. Plotting contour plots...")
    plot_contour_side_by_side(organized_results, output_dir, example_index)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize attention ablation experiment results"
    )
    parser.add_argument(
        "--example_index",
        "-idx",
        type=int,
        default=0,
        help="Example index to visualize (default: 0)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/attention_ablation",
        help="Directory containing ablation result JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/attention_ablation_plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()
    main(**vars(args))
