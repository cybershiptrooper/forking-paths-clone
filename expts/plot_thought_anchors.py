"""Plot Figure 6 from Thought Anchors (Bogdan et al., arXiv 2506.19143).

Reads a mask JSON that has been annotated by:
  - classify_sentences.py  → function_tags, depends_on
  - thought_anchor_analysis.py → suppression_matrix, receiver_head_attention

Produces a composite figure with:
  1. Attention suppression heatmap  (if suppression_matrix exists)
  2. Receiver head vertical attention line plot  (if receiver_head_attention exists)
  3. Sentence classification strip  (if function_tags exist)

Usage:
    python -m expts.plot_thought_anchors --mask results/circuit_discovery/test_global_2.json
    python -m expts.plot_thought_anchors --mask results/circuit_discovery/test_global_2.json -o fig6.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Consistent colors for each function tag
TAG_COLORS = {
    "problem_setup": "#4e79a7",
    "plan_generation": "#f28e2b",
    "fact_retrieval": "#e15759",
    "active_computation": "#76b7b2",
    "uncertainty_management": "#59a14f",
    "result_consolidation": "#edc948",
    "self_checking": "#b07aa1",
    "final_answer_emission": "#ff9da7",
    "unknown": "#bab0ac",
}


def _truncate(text, max_len=50):
    text = text.strip().replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text


def plot_figure6(data, output_path=None):
    sentences = data["sentences"]
    metadata = data["metadata"]
    num_sents = len(sentences)

    has_suppression = "suppression_matrix" in metadata
    has_receiver = "receiver_head_attention" in metadata
    has_tags = "function_tags" in sentences[0]

    # Decide layout
    num_panels = has_suppression + has_receiver + has_tags
    if num_panels == 0:
        print("No data to plot. Run classify_sentences.py or thought_anchor_analysis.py first.")
        return

    if has_suppression and has_receiver:
        # Full Figure 6 layout: heatmap on top, line plot below, shared x-axis
        fig, axes = plt.subplots(
            2 + has_tags, 1,
            figsize=(max(12, num_sents * 0.5), 6 + 3 * has_receiver + 1.5 * has_tags),
            gridspec_kw={"height_ratios": [3] + ([1.5] if has_receiver else []) + ([0.5] if has_tags else [])},
            sharex=True,
        )
        axes = list(axes) if hasattr(axes, '__len__') else [axes]
    elif has_suppression:
        fig, axes = plt.subplots(
            1 + has_tags, 1,
            figsize=(max(12, num_sents * 0.5), 6 + 1.5 * has_tags),
            gridspec_kw={"height_ratios": [3] + ([0.5] if has_tags else [])},
            sharex=True,
        )
        axes = list(axes) if hasattr(axes, '__len__') else [axes]
    elif has_receiver:
        fig, axes = plt.subplots(
            1 + has_tags, 1,
            figsize=(max(12, num_sents * 0.5), 4 + 1.5 * has_tags),
            gridspec_kw={"height_ratios": [2] + ([0.5] if has_tags else [])},
            sharex=True,
        )
        axes = list(axes) if hasattr(axes, '__len__') else [axes]
    else:
        # Tags only
        fig, ax_tag = plt.subplots(figsize=(max(12, num_sents * 0.5), 2))
        axes = [ax_tag]

    ax_idx = 0

    # ---- Panel 1: Suppression heatmap ----
    if has_suppression:
        ax = axes[ax_idx]; ax_idx += 1
        matrix = np.array(metadata["suppression_matrix"])
        im = ax.imshow(
            matrix.T, aspect="auto", cmap="Reds", origin="lower",
            interpolation="nearest",
        )
        ax.set_ylabel("Affected sentence")
        ax.set_title("Attention suppression: KL divergence")
        fig.colorbar(im, ax=ax, label="KL divergence", shrink=0.8)
        ax.set_xlabel("Suppressed sentence")

    # ---- Panel 2: Receiver head attention ----
    if has_receiver:
        ax = axes[ax_idx]; ax_idx += 1
        vert = metadata["receiver_head_attention"]
        x = np.arange(num_sents)
        ax.bar(x, vert, color="#4e79a7", alpha=0.8, width=0.8)
        ax.set_ylabel("Avg attention\n(receiver heads)")
        config = metadata.get("figure6_config", {})
        top_k = config.get("top_k", "?")
        min_gap = config.get("min_gap", "?")
        ax.set_title(f"Vertical attention via top-{top_k} receiver heads (min_gap={min_gap})")
        ax.set_xlim(-0.5, num_sents - 0.5)

    # ---- Panel 3: Classification strip ----
    if has_tags:
        ax = axes[ax_idx]; ax_idx += 1
        for i, s in enumerate(sentences):
            tags = s.get("function_tags", ["unknown"])
            color = TAG_COLORS.get(tags[0], TAG_COLORS["unknown"])
            ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=0.5)
            # Sentence index label
            ax.text(i + 0.5, 0, str(i), ha="center", va="center", fontsize=7)
        ax.set_xlim(-0.5, num_sents - 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Sentence index")

        # Legend
        used_tags = set()
        for s in sentences:
            for t in s.get("function_tags", []):
                used_tags.add(t)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=TAG_COLORS.get(t, TAG_COLORS["unknown"]))
            for t in sorted(used_tags)
        ]
        ax.legend(
            handles, sorted(used_tags),
            loc="upper center", bbox_to_anchor=(0.5, -0.6),
            ncol=min(len(used_tags), 4), fontsize=8, frameon=False,
        )

    fig.suptitle(
        f"Thought Anchor Analysis — {num_sents} sentences",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Thought Anchors Figure 6.")
    parser.add_argument("--mask", required=True, help="Path to mask JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output image path (e.g. fig6.png)")
    args = parser.parse_args()

    with open(args.mask) as f:
        data = json.load(f)

    plot_figure6(data, output_path=args.output)


if __name__ == "__main__":
    main()
