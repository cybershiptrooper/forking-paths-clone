"""Visualization functions for circuit discovery results.

All functions are stateless utilities using matplotlib. Aggregation across
heads/layers happens here, not in the mask classes.
"""

from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from utils.masks import NodeMask


def _sci_fmt(x, _pos):
    """Format tick value in scientific notation (e.g. 1.0e-8)."""
    if x == 0:
        return "0"
    return f"{x:.1e}"


_sci_formatter = mticker.FuncFormatter(_sci_fmt)


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
    threshold: float = 0.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Greens",
    title: Optional[str] = None,
    use_threshold: bool = False,
):
    """Lower-triangular heatmap of sentence-to-sentence scores for a single (layer, head).

    Args:
        node_mask: NodeMask with attribution scores
        layer: Layer index
        head: Head index
        threshold: Values with score < threshold are set to zero
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap name
        title: Plot title (auto-generated if None)
        use_threshold: Whether to apply thresholding (default False, for cleaner head-level patterns)
    """
    scores = np.array(node_mask.scores[layer][head])
    num_sents = scores.shape[0]
    labels = _get_sentence_labels(node_mask.sentences, max_chars=15)

    # Apply causal (lower-triangular) mask
    causal_mask = np.tril(np.ones_like(scores))
    scores_masked = np.where(causal_mask, scores, np.nan)

    # Apply threshold: dim entries below threshold
    if threshold > 0 and use_threshold:
        scores_masked = np.where(scores_masked >= threshold, scores_masked, 0.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(scores_masked, cmap=cmap, aspect="equal", origin="upper",
                   interpolation="nearest")
    ax.set_xticks(range(num_sents))
    ax.set_yticks(range(num_sents))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key Sentence")
    ax.set_ylabel("Query Sentence")

    # Cell borders for the causal triangle
    for i in range(num_sents):
        for j in range(i + 1):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=0.5, edgecolor="gray", facecolor="none",
            ))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Importance")
    cbar.ax.yaxis.set_major_formatter(_sci_formatter)

    if title is None:
        title = f"Layer {layer}, Head {head}"
    ax.set_title(title)

    return ax


def plot_top_heads(
    node_mask: NodeMask,
    layer: int,
    top_k: int = 5,
    threshold: float = 0.0,
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
):
    """Subplot grid of top-K heads (ranked by total attribution) at a given layer.

    Args:
        node_mask: NodeMask with attribution scores
        layer: Layer index
        top_k: Number of top heads to show
        threshold: Scores below this are zeroed before ranking/display
        cmap: Colormap name
        save_path: Path to save figure (shows if None)
    """
    importance = node_mask.get_head_importance(layer, threshold=threshold)
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
            threshold=threshold,
            ax=axes[i],
            cmap=cmap,
            title=f"L{layer} H{head} (attr={importance[head]:.2e})",
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
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(_sci_formatter)
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
    Thresholding uses signed scores (score > threshold). With default
    negated scores, green edges reduce KL.

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
        title = f"Sentence Graph — Layer {layer} (threshold={threshold})"
    else:
        scores = np.array(node_mask.get_all_layers_aggregated(aggregation))
        title = f"Sentence Graph — All Layers (threshold={threshold})"

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
            score = scores[i, j]
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
        green_weights = [scores[i, j] for i, j in green_edges]
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


def plot_attention_pattern(
    node_mask: NodeMask,
    layer: Optional[int] = None,
    aggregation: str = "mean",
    threshold: float = 0.0,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Greens",
    save_path: Optional[str] = None,
):
    """Lower-triangular attention-pattern heatmap of sentence-to-sentence scores.

    Displays causal dependencies as a triangular matrix (row i attends to
    column j where j <= i), mimicking the classic attention-pattern style.

    Args:
        node_mask: NodeMask with attribution scores
        layer: Specific layer index, or None for all-layers aggregation
        aggregation: How to aggregate across heads ("mean", "max", "sum")
        threshold: Values with score < threshold are dimmed to zero
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap (default "Greens" for the causal-importance look)
        save_path: Path to save figure (shows if None)
    """
    if layer is not None:
        scores = np.array(node_mask.get_layer_aggregated(layer, aggregation))
        title = f"Sentence Pair Importance — Layer {layer} ({aggregation})"
    else:
        scores = np.array(node_mask.get_all_layers_aggregated(aggregation))
        title = f"Sentence Pair Importance — All Layers ({aggregation})"

    num_sents = scores.shape[0]

    # Apply causal (lower-triangular) mask: row i can attend to columns j <= i
    causal_mask = np.tril(np.ones_like(scores))
    scores_masked = np.where(causal_mask, scores, np.nan)

    # Apply threshold: dim entries below threshold
    if threshold > 0:
        scores_masked = np.where(scores_masked >= threshold, scores_masked, 0.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    # Plot heatmap with NaN for upper triangle (renders as white/transparent)
    im = ax.imshow(
        scores_masked,
        cmap=cmap,
        aspect="equal",
        origin="upper",
        interpolation="nearest",
    )

    # Sentence labels
    labels = _get_sentence_labels(node_mask.sentences, max_chars=15)
    ax.set_xticks(range(num_sents))
    ax.set_yticks(range(num_sents))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key Sentence")
    ax.set_ylabel("Query Sentence")

    # Draw cell borders for the causal triangle
    for i in range(num_sents):
        for j in range(i + 1):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=0.5, edgecolor="gray", facecolor="none",
            )
            ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Importance")
    cbar.ax.yaxis.set_major_formatter(_sci_formatter)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig if save_path or ax is None else ax


def plot_threshold_vs_metrics(
    thresholds: List[float],
    sparsities: List[float],
    kl_scores: List[float],
    save_path: Optional[str] = None,
):
    """Plot threshold vs sparsity and KL divergence (dual y-axis).

    Args:
        thresholds: List of threshold values
        sparsities: Corresponding sparsity values
        kl_scores: Corresponding KL divergence scores
        save_path: Save path
    """
    thresholds_arr = np.array(thresholds, dtype=float)
    sparsities_arr = np.array(sparsities, dtype=float)
    kl_arr = np.array(kl_scores, dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Sparsity", color=color1)
    ax1.plot(thresholds_arr, sparsities_arr, "o-", color=color1, label="Sparsity")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.xaxis.set_major_formatter(_sci_formatter)
    # ax1.set_xscale("log")

    ax1b = ax1.twinx()
    color2 = "tab:red"
    ax1b.set_ylabel("KL Divergence", color=color2)
    ax1b.plot(thresholds_arr, kl_arr, "s-", color=color2, label="KL Divergence")
    ax1b.tick_params(axis="y", labelcolor=color2)
    # ax1b.yaxis.set_major_formatter(_sci_formatter)

    fig.suptitle("Threshold vs Sparsity/KL Divergence")
    fig.tight_layout()

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
    thresholds_arr = np.array(thresholds, dtype=float)
    sparsities_arr = np.array(sparsities, dtype=float)
    kl_arr = np.array(kl_scores, dtype=float)

    sort_idx = np.argsort(sparsities_arr)
    sparsities_sorted = sparsities_arr[sort_idx]
    kl_sorted = kl_arr[sort_idx]

    if np.all(thresholds_arr > 0):
        norm = mcolors.LogNorm(vmin=thresholds_arr.min(), vmax=thresholds_arr.max())
    else:
        norm = mcolors.Normalize(vmin=thresholds_arr.min(), vmax=thresholds_arr.max())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sparsities_sorted, kl_sorted, "-", color="gray", alpha=0.4)
    sc = ax.scatter(
        sparsities_arr,
        kl_arr,
        c=thresholds_arr,
        cmap="viridis",
        norm=norm,
        s=40,
        edgecolors="k",
        linewidths=0.3,
    )

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("KL Divergence")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # ax.yaxis.set_major_formatter(_sci_formatter)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Threshold")

    fig.suptitle("Sparsity vs KL Divergence")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_full_circuit(
    node_mask: NodeMask,
    threshold: float = 0.0,
    aggregation: str = "mean",
    save_path: Optional[str] = None,
):
    """Full circuit overview: sentences on x-axis, layers on y-axis.

    Shows per-sentence importance at each layer, giving a bird's-eye view
    of where each sentence matters across the network. Importance is computed
    as the mean attribution score each sentence receives as a key (i.e. how
    much other sentences attend to it), aggregated across heads.

    Green cells indicate above-threshold importance (causal relation),
    red cells indicate below-threshold (no causal relation).
    Thresholding uses signed scores (>= threshold), assuming positive reduces KL.

    Args:
        node_mask: NodeMask with attribution scores
        threshold: Importance threshold for green/red coloring
        aggregation: How to aggregate across heads ("mean", "max", "sum")
        save_path: Path to save figure (shows if None)
    """
    layers = node_mask.layers
    num_sents = len(node_mask.sentences)
    num_layers = len(layers)

    # Build (num_layers, num_sents) importance matrix.
    # For each (layer, sentence), compute how much that sentence is attended
    # to by downstream sentences (column importance under causal mask).
    importance_matrix = np.zeros((num_layers, num_sents))

    for i, layer in enumerate(layers):
        scores = np.array(node_mask.get_layer_aggregated(layer, aggregation))
        # Causal mask: row q attends to column k where k <= q
        causal_mask = np.tril(np.ones_like(scores))
        scores_causal = np.where(causal_mask, scores, 0.0)
        # Exclude self-attention (diagonal)
        np.fill_diagonal(scores_causal, 0.0)
        # Column sum = total importance of each sentence as a key
        col_sum = scores_causal.sum(axis=0)
        # Normalise by number of valid query positions per column
        valid_counts = np.maximum(causal_mask.sum(axis=0) - 1, 1)  # -1 for diagonal
        importance_matrix[i] = col_sum / valid_counts

    labels = _get_sentence_labels(node_mask.sentences, max_chars=15)

    fig_w = max(8, num_sents * 1.0)
    fig_h = max(4, num_layers * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if threshold > 0:
        # Binary green/red coloring based on threshold
        binary = np.where(importance_matrix >= threshold, 1.0, 0.0)
        cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])  # red, green
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(
            binary, cmap=cmap, norm=norm, aspect="auto", origin="lower",
            interpolation="nearest",
        )
        # Overlay the actual score values as text
        for i in range(num_layers):
            for j in range(num_sents):
                val = importance_matrix[i, j]
                ax.text(j, i, f"{val:.1e}", ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
    else:
        im = ax.imshow(
            importance_matrix, cmap="Greens", aspect="auto", origin="lower",
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Importance")
        cbar.ax.yaxis.set_major_formatter(_sci_formatter)

    # Cell grid lines
    for i in range(num_layers):
        for j in range(num_sents):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=0.5, edgecolor="gray", facecolor="none",
            ))

    ax.set_xticks(range(num_sents))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=8)
    ax.set_xlabel("Sentence")
    ax.set_ylabel("Layer")
    ax.set_title("Circuit Overview — Per-Sentence Importance by Layer")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_per_token_objective(
    per_token_data: List[dict],
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Line plot of per-token KL divergence across continuation branches.

    X-axis: token index in continuation, Y-axis: KL divergence.
    One line per branch, one subplot per threshold.

    Args:
        per_token_data: List of threshold evaluation dicts, each containing
            "threshold", "per_token_kl" (list of lists, one per branch).
        threshold: If given, plot only this threshold. Otherwise plot all.
        save_path: Path to save figure (shows if None).
    """
    # Filter to entries that have per-token data
    entries = [d for d in per_token_data if "per_token_kl" in d]
    if not entries:
        return None

    if threshold is not None:
        entries = [d for d in entries if d["threshold"] == threshold]
        if not entries:
            return None

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes.flatten()

    cmap = plt.get_cmap("tab10")

    for idx, entry in enumerate(entries):
        ax = axes[idx]
        branches = entry["per_token_kl"]
        for b_idx, branch_kl in enumerate(branches):
            ax.plot(
                range(len(branch_kl)),
                branch_kl,
                linewidth=0.8,
                alpha=0.7,
                color=cmap(b_idx % 10),
                label=f"Branch {b_idx}",
            )
        ax.set_xlabel("Continuation Token Index")
        ax.set_ylabel("KL Divergence")
        ax.set_title(
            f"threshold={entry['threshold']:.1e} "
            f"(sparsity={entry.get('sparsity', 0):.1%})"
        )
        ax.yaxis.set_major_formatter(_sci_formatter)
        if len(branches) <= 10:
            ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Per-Token Objective (KL) Across Branches", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_per_sentence_objective(
    per_token_data: List[dict],
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Line plot of per-sentence mean KL divergence across continuation branches.

    X-axis: sentence index in continuation, Y-axis: mean KL divergence.
    One "x-" line per branch, one subplot per threshold.

    Args:
        per_token_data: List of threshold evaluation dicts, each containing
            "threshold", "per_sentence_kl" (list of lists of dicts with
            "text" and "mean_kl", one list per branch).
        threshold: If given, plot only this threshold. Otherwise plot all.
        save_path: Path to save figure (shows if None).
    """
    entries = [d for d in per_token_data if "per_sentence_kl" in d]
    if not entries:
        return None

    if threshold is not None:
        entries = [d for d in entries if d["threshold"] == threshold]
        if not entries:
            return None

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes.flatten()

    cmap = plt.get_cmap("tab10")

    for idx, entry in enumerate(entries):
        ax = axes[idx]
        branches = entry["per_sentence_kl"]  # list of list-of-dicts

        for b_idx, branch_sents in enumerate(branches):
            kl_vals = [s["mean_kl"] for s in branch_sents]
            ax.plot(
                range(len(kl_vals)),
                kl_vals,
                "x-",
                linewidth=0.8,
                alpha=0.7,
                color=cmap(b_idx % 10),
                label=f"Branch {b_idx}",
            )

        ax.set_xlabel("Continuation Sentence Index")
        ax.set_ylabel("Mean KL Divergence")
        ax.set_title(
            f"threshold={entry['threshold']:.1e} "
            f"(sparsity={entry.get('sparsity', 0):.1%})"
        )
        ax.yaxis.set_major_formatter(_sci_formatter)
        if len(branches) <= 10:
            ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Per-Sentence Objective (KL) Across Branches", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sparsity_vs_kl_with_random(
    thresholds: list[float],
    sparsities: list[float],
    kl_scores: list[float],
    random_kl_scores: list[float] | list[list[float]] | None,
    save_path: str,
):
    thresholds_arr = np.array(thresholds, dtype=float)
    sparsities_arr = np.array(sparsities, dtype=float)
    kl_arr = np.array(kl_scores, dtype=float)

    sort_idx = np.argsort(sparsities_arr)
    sparsities_sorted = sparsities_arr[sort_idx]
    kl_sorted = kl_arr[sort_idx]

    if np.all(thresholds_arr > 0):
        norm = mcolors.LogNorm(vmin=thresholds_arr.min(), vmax=thresholds_arr.max())
    else:
        norm = mcolors.Normalize(vmin=thresholds_arr.min(), vmax=thresholds_arr.max())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sparsities_sorted, kl_sorted, "-", color="gray", alpha=0.4)
    sc = ax.scatter(
        sparsities_arr,
        kl_arr,
        c=thresholds_arr,
        cmap="viridis",
        norm=norm,
        s=40,
        edgecolors="k",
        linewidths=0.3,
    )

    if random_kl_scores is not None:
        random_arr = np.array(random_kl_scores, dtype=float)
        if random_arr.ndim == 2:
            # Multiple random samples: shape (num_thresholds, K)
            random_mean = random_arr.mean(axis=1)
            random_std = random_arr.std(axis=1)
            random_mean_sorted = random_mean[sort_idx]
            random_std_sorted = random_std[sort_idx]
            ax.plot(
                sparsities_sorted,
                random_mean_sorted,
                "--",
                color="tab:orange",
                label="Random baseline (mean)",
            )
            ax.fill_between(
                sparsities_sorted,
                random_mean_sorted - random_std_sorted,
                random_mean_sorted + random_std_sorted,
                alpha=0.25,
                color="tab:orange",
                label="Random baseline (\u00b11\u03c3)",
            )
            ax.scatter(
                sparsities_arr,
                random_mean,
                marker="x",
                color="tab:orange",
                s=35,
                linewidths=0.8,
            )
        else:
            # Single random baseline (backward compat)
            random_sorted = random_arr[sort_idx]
            ax.plot(
                sparsities_sorted,
                random_sorted,
                "--",
                color="tab:orange",
                label="Random baseline",
            )
            ax.scatter(
                sparsities_arr,
                random_arr,
                marker="x",
                color="tab:orange",
                s=35,
                linewidths=0.8,
            )
        ax.legend(loc="best", frameon=False)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("KL Divergence")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Threshold")

    fig.suptitle("Sparsity vs KL Divergence")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
