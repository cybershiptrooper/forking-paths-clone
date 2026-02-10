"""Visualize circuit discovery results (NodeMask).

Loads a NodeMask JSON file and generates per-layer heatmaps (collapsed across
heads), an aggregated view, and top-K individual head plots.

Usage:
    uv run python expts/visualize_circuit.py results/circuit_discovery/node_mask_*.json
    uv run python expts/visualize_circuit.py results/circuit_discovery/node_mask_*.json --output_dir plots/
"""

import argparse

from utils.circuit_vis import plot_node_mask_per_layer, plot_top_heads
from utils.masks import NodeMask


def main(
    mask_path: str,
    output_dir: str = "results/circuit_discovery/plots",
    cmap: str = "YlOrRd",
    top_k: int = 5,
):
    print(f"Loading NodeMask from: {mask_path}")
    node_mask = NodeMask.from_json(mask_path)

    print(f"  Algorithm: {node_mask.metadata.get('algorithm', 'unknown')}")
    print(f"  Layers: {sorted(node_mask.scores.keys())}")
    print(f"  Sentences: {len(node_mask.sentences)}")
    if node_mask.sentence_texts:
        for i, text in enumerate(node_mask.sentence_texts):
            print(f"    S{i}: {repr(text[:60])}")

    # Print top edges per layer
    for layer_idx in sorted(node_mask.scores.keys()):
        scores = node_mask.scores[layer_idx]  # (num_heads, S, S)
        num_heads, S, _ = scores.shape

        # Find top-5 edges across all heads
        flat = scores.reshape(-1)
        top_k_vals, top_flat_idx = flat.topk(min(5, flat.numel()))

        print(f"\n  Layer {layer_idx} top edges:")
        for val, flat_i in zip(top_k_vals, top_flat_idx):
            idx = flat_i.item()
            h = idx // (S * S)
            remainder = idx % (S * S)
            q_sent = remainder // S
            k_sent = remainder % S
            print(f"    H{h}: S{q_sent} -> S{k_sent}: {val.item():.6f}")

    print(f"\nGenerating plots in: {output_dir}")
    plot_node_mask_per_layer(node_mask, output_dir, cmap=cmap)
    print("  Saved per-layer and aggregated heatmaps")

    plot_top_heads(node_mask, output_dir, top_k=top_k, cmap=cmap)
    print(f"  Saved top-{top_k} head plots")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize circuit discovery results"
    )
    parser.add_argument("mask_path", type=str, help="Path to NodeMask JSON file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/circuit_discovery/plots",
    )
    parser.add_argument("--cmap", type=str, default="YlOrRd")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))
