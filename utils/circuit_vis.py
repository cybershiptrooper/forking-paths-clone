"""Visualization functions for circuit discovery results.

All functions are stateless utilities using matplotlib. Aggregation across
heads/layers happens here, not in the mask classes.
"""

from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils.masks import NodeMask


def _get_sentence_labels(sentences: List[dict], max_chars: int = 20) -> List[str]:
    """Create short labels for sentences."""
    labels = []
    for i, s in enumerate(sentences):
        text = s.get("text", "")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        labels.append(f"S{i}: {text}" if text else f"S{i}")
    return labels


def plot_head_heatmap(
    node_mask: NodeMask,
    layer: int,
    head: int,
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdYlGn",
    title: Optional[str] = None,
):
    """Heatmap of sentence-to-sentence scores for a single (layer, head).

    Args:
        node_mask: NodeMask with attribution scores
        layer: Layer index
        head: Head index
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap name
        title: Plot title (auto-generated if None)
    """
    scores = np.array(node_mask.scores[layer][head])
    labels = _get_sentence_labels(node_mask.sentences)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(scores, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Target Sentence (Key)")
    ax.set_ylabel("Source Sentence (Query)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    if title is None:
        title = f"Layer {layer}, Head {head}"
    ax.set_title(title)

    return ax


def plot_top_heads(
    node_mask: NodeMask,
    layer: int,
    top_k: int = 5,
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
):
    """Subplot grid of top-K heads (ranked by total |attribution|) at a given layer.

    Args:
        node_mask: NodeMask with attribution scores
        layer: Layer index
        top_k: Number of top heads to show
        cmap: Colormap name
        save_path: Path to save figure (shows if None)
    """
    importance = node_mask.get_head_importance(layer)
    top_heads = list(importance.keys())[:top_k]

    ncols = min(top_k, 3)
    nrows = (len(top_heads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, head in enumerate(top_heads):
        plot_head_heatmap(
            node_mask,
            layer,
            head,
            ax=axes[i],
            cmap=cmap,
            title=f"L{layer} H{head} (|attr|={importance[head]:.4f})",
        )

    # Hide unused axes
    for i in range(len(top_heads), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Top {top_k} Heads at Layer {layer}", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_layer_aggregated(
    node_mask: NodeMask,
    layer: int,
    aggregation: str = "mean",
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
):
    """Aggregate across heads within a layer -> single sentence x sentence heatmap.

    Args:
        node_mask: NodeMask with attribution scores
        layer: Layer index
        aggregation: "mean", "max", or "sum"
        ax: Matplotlib axes
        cmap: Colormap
        save_path: Path to save figure
    """
    scores = np.array(node_mask.get_layer_aggregated(layer, aggregation))
    labels = _get_sentence_labels(node_mask.sentences)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(scores, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Target Sentence (Key)")
    ax.set_ylabel("Source Sentence (Query)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Layer {layer} — {aggregation} across heads")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_layer_comparison(
    node_mask: NodeMask,
    layers: Optional[List[int]] = None,
    aggregation: str = "mean",
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
):
    """Side-by-side aggregated heatmaps for multiple layers.

    Args:
        node_mask: NodeMask
        layers: Which layers to plot (default: all)
        aggregation: How to aggregate across heads
        cmap: Colormap
        save_path: Save path
    """
    if layers is None:
        layers = node_mask.layers

    ncols = min(len(layers), 3)
    nrows = (len(layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        plot_layer_aggregated(node_mask, layer, aggregation, ax=axes[i], cmap=cmap)

    for i in range(len(layers), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Layer Comparison ({aggregation} across heads)", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_circuit_graph(
    node_mask: NodeMask,
    threshold: float = 0.1,
    layer: Optional[int] = None,
    aggregation: str = "mean",
    save_path: Optional[str] = None,
):
    """Network graph with sentences as nodes, edges colored by importance.

    Green = above threshold (important), Red = below threshold (not important).

    Args:
        node_mask: NodeMask
        threshold: Edge importance threshold
        layer: Specific layer (None = aggregate across all layers)
        aggregation: How to aggregate
        save_path: Save path
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx required for circuit graph visualization. pip install networkx")
        return None

    if layer is not None:
        scores = np.array(node_mask.get_layer_aggregated(layer, aggregation))
        title = f"Circuit Graph — Layer {layer} (threshold={threshold})"
    else:
        scores = np.array(node_mask.get_all_layers_aggregated(aggregation))
        title = f"Circuit Graph — All Layers (threshold={threshold})"

    num_sents = scores.shape[0]
    labels = _get_sentence_labels(node_mask.sentences, max_chars=15)

    G = nx.DiGraph()
    for i in range(num_sents):
        G.add_node(i, label=labels[i])

    green_edges = []
    red_edges = []
    edge_weights = []

    for i in range(num_sents):
        for j in range(num_sents):
            if i == j:
                continue
            score = abs(scores[i, j])
            if score > threshold:
                green_edges.append((i, j))
            else:
                red_edges.append((i, j))
            edge_weights.append(score)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(num_sents))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=800, ax=ax
    )
    nx.draw_networkx_labels(
        G, pos, {i: labels[i] for i in range(num_sents)}, font_size=7, ax=ax
    )

    # Draw important edges (green)
    if green_edges:
        green_weights = [abs(scores[i, j]) for i, j in green_edges]
        max_w = max(green_weights) if green_weights else 1.0
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=green_edges,
            edge_color="green",
            width=[2.0 * w / max_w + 0.5 for w in green_weights],
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            ax=ax,
        )

    # Draw non-important edges (red, thin)
    if red_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=red_edges,
            edge_color="red",
            width=0.3,
            alpha=0.2,
            arrows=True,
            arrowsize=8,
            ax=ax,
        )

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sparsity_vs_kl(
    thresholds: List[float],
    sparsities: List[float],
    kl_scores: List[float],
    save_path: Optional[str] = None,
):
    """Pareto plot: sparsity vs KL divergence for different thresholds.

    Args:
        thresholds: List of threshold values
        sparsities: Corresponding sparsity values
        kl_scores: Corresponding KL divergence scores
        save_path: Save path
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Sparsity", color=color1)
    ax1.plot(thresholds, sparsities, "o-", color=color1, label="Sparsity")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("KL Divergence", color=color2)
    ax2.plot(thresholds, kl_scores, "s-", color=color2, label="KL Divergence")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Sparsity vs KL Divergence")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
