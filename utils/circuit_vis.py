"""Visualization functions for circuit discovery results.

All collapsing from (num_heads, S, S) to (S, S) happens here during
visualization only — the NodeMask stores full per-head resolution.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from utils.masks import NodeMask
from utils.utils import Sentence


def plot_sentence_heatmap(
    matrix: torch.Tensor,
    title: str = "Attribution Scores",
    sentences: Optional[List[Sentence]] = None,
    sentence_texts: Optional[List[str]] = None,
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a single (S, S) attribution matrix as a heatmap.

    Args:
        matrix: (S, S) tensor of attribution scores.
        title: Plot title.
        sentences: Optional sentence list for tick labels.
        sentence_texts: Optional decoded text per sentence.
        cmap: Colormap name.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    data = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="upper")

    ax.set_xlabel("Key Sentence (attended to)")
    ax.set_ylabel("Query Sentence (attending from)")
    ax.set_title(title)

    if sentences is not None and len(sentences) <= 30:
        if sentence_texts is not None:
            tick_labels = [
                f"S{i}: {t[:20]}..." if len(t) > 20 else f"S{i}: {t}"
                for i, t in enumerate(sentence_texts)
            ]
        else:
            tick_labels = [
                f"S{i} ({s.start}-{s.end})" for i, s in enumerate(sentences)
            ]
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attribution Score")

    plt.tight_layout()
    return fig, ax


def plot_single_head(
    node_mask: NodeMask,
    layer: int,
    head: int,
    output_path: Optional[str] = None,
    cmap: str = "YlOrRd",
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize one head's (S, S) attribution.

    Args:
        node_mask: NodeMask with per-head scores.
        layer: Layer index.
        head: Head index.
        output_path: If provided, save figure to this path.
        cmap: Colormap name.

    Returns:
        (fig, ax) tuple.
    """
    matrix = node_mask.scores[layer][head]
    fig, ax = plot_sentence_heatmap(
        matrix,
        title=f"Layer {layer}, Head {head} - Attribution",
        sentences=node_mask.sentences,
        sentence_texts=node_mask.sentence_texts or None,
        cmap=cmap,
    )
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_node_mask_per_layer(
    node_mask: NodeMask,
    output_dir: str,
    cmap: str = "YlOrRd",
) -> None:
    """Save per-layer (collapsed across heads) and aggregated heatmaps.

    For each layer: mean across heads -> (S, S) heatmap.
    Aggregated: mean across all heads and layers -> single (S, S) heatmap.

    Args:
        node_mask: NodeMask with per-head scores.
        output_dir: Directory to save PNG files.
        cmap: Colormap name.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_layer_means = []

    for layer_idx in sorted(node_mask.scores.keys()):
        scores = node_mask.scores[layer_idx]  # (num_heads, S, S)
        layer_mean = scores.mean(dim=0)  # (S, S)
        all_layer_means.append(layer_mean)

        fig, _ = plot_sentence_heatmap(
            layer_mean,
            title=f"Layer {layer_idx} - Mean Across Heads",
            sentences=node_mask.sentences,
            sentence_texts=node_mask.sentence_texts or None,
            cmap=cmap,
        )
        fig.savefig(
            os.path.join(output_dir, f"attribution_layer_{layer_idx}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Aggregated across all layers
    if all_layer_means:
        agg = torch.stack(all_layer_means).mean(dim=0)  # (S, S)
        fig, _ = plot_sentence_heatmap(
            agg,
            title="Aggregated Attribution (Mean Across Heads & Layers)",
            sentences=node_mask.sentences,
            sentence_texts=node_mask.sentence_texts or None,
            cmap=cmap,
        )
        fig.savefig(
            os.path.join(output_dir, "attribution_aggregated.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_top_heads(
    node_mask: NodeMask,
    output_dir: str,
    top_k: int = 5,
    cmap: str = "YlOrRd",
) -> None:
    """Plot the top-K heads with highest mean attribution.

    Args:
        node_mask: NodeMask with per-head scores.
        output_dir: Directory to save PNG files.
        top_k: Number of top heads to plot.
        cmap: Colormap name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect (mean_score, layer, head) for all heads
    head_scores = []
    for layer_idx, scores in node_mask.scores.items():
        num_heads = scores.shape[0]
        for h in range(num_heads):
            mean_score = scores[h].mean().item()
            head_scores.append((mean_score, layer_idx, h))

    head_scores.sort(reverse=True)

    for rank, (score, layer_idx, head) in enumerate(head_scores[:top_k]):
        fig, _ = plot_single_head(node_mask, layer_idx, head, cmap=cmap)
        fig.suptitle(
            f"Rank {rank + 1}: Layer {layer_idx}, Head {head} (mean={score:.6f})",
            fontsize=12,
        )
        fig.savefig(
            os.path.join(
                output_dir, f"top_head_{rank + 1}_L{layer_idx}_H{head}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
