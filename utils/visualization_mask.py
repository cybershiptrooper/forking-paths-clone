from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from utils.masks import EdgewiseMask


def plot_head_mask_matrix(
    mask: EdgewiseMask,
    layer: int,
    head: int,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    layer_idx = mask.layers.index(layer)
    data = np.array(mask.mask_values)[layer_idx][head]
    if threshold is not None:
        data = (data >= threshold).astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    ax.set_title(f"Layer {layer} Head {head} Mask")
    ax.set_xlabel("Key Sentence Chunk")
    ax.set_ylabel("Query Sentence Chunk")
    ax.set_xticks(range(data.shape[0]))
    ax.set_yticks(range(data.shape[0]))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mask Value")
    plt.tight_layout()
    return fig, ax


def plot_layer_circuit(
    mask: EdgewiseMask,
    layer: int,
    aggregation: str = "mean",
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    layer_idx = mask.layers.index(layer)
    layer_data = np.array(mask.mask_values)[layer_idx]
    if aggregation == "mean":
        data = layer_data.mean(axis=0)
    elif aggregation == "max":
        data = layer_data.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    if threshold is not None:
        data = (data >= threshold).astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    ax.set_title(f"Layer {layer} Circuit ({aggregation})")
    ax.set_xlabel("Key Sentence Chunk")
    ax.set_ylabel("Query Sentence Chunk")
    ax.set_xticks(range(data.shape[0]))
    ax.set_yticks(range(data.shape[0]))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mask Value")
    plt.tight_layout()
    return fig, ax
