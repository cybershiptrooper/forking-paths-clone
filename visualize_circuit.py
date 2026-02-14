"""Visualize learned circuit masks from circuit discovery.

Loads a saved NodeMask JSON and generates multi-level visualizations:
- Per-head heatmaps (top K heads at each layer)
- Per-layer aggregated heatmaps
- Layer comparison (side-by-side)
- Circuit graph (sentences as nodes, edges by importance)
- Sparsity vs KL plot (x = sparsity, y = KL; if threshold evaluation data is available)
"""

import os
import argparse

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.masks import NodeMask, load_mask
from utils.circuit_vis import (
    plot_top_heads,
    plot_layer_aggregated,
    plot_layer_comparison,
    plot_circuit_graph,
    plot_attention_pattern,
    plot_threshold_vs_metrics,
    plot_full_circuit,
    plot_per_token_objective,
    plot_per_sentence_objective,
    plot_sparsity_vs_kl_with_random,
)

def main(
    mask_path: str,
    layers: list[int] = None,
    threshold: float = 0.1,
    top_k_heads: int = 5,
    output_dir: str = None,
    per_head: bool = False,
    per_layer: bool = False,
):
    # Load mask
    print(f"Loading mask from {mask_path}...")
    mask = load_mask(mask_path)
    if not isinstance(mask, NodeMask):
        raise ValueError(f"Expected NodeMask, got {type(mask).__name__}")

    if layers is None:
        layers = mask.layers

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(mask_path),
            os.path.splitext(os.path.basename(mask_path))[0] + "_viz",
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Mask: {mask.algorithm}, layers={mask.layers}")
    print(f"Sentences: {len(mask.sentences)}")
    print(f"Saving visualizations to {output_dir}/")

    # 1. Top heads per layer (opt-in)
    if per_head:
        print("\nGenerating top-heads heatmaps...")
        for layer in layers:
            save_path = os.path.join(output_dir, f"top_heads_layer_{layer}.png")
            plot_top_heads(mask, layer, top_k=top_k_heads, threshold=threshold, save_path=save_path)
            print(f"  Saved: {save_path}")

    # 2. Layer-aggregated heatmaps (opt-in)
    if per_layer:
        print("\nGenerating layer-aggregated heatmaps...")
        for layer in layers:
            save_path = os.path.join(output_dir, f"layer_aggregated_{layer}.png")
            plot_layer_aggregated(mask, layer, aggregation="mean", save_path=save_path)
            print(f"  Saved: {save_path}")

    # 3. Layer comparison
    print("\nGenerating layer comparison...")
    save_path = os.path.join(output_dir, "layer_comparison.png")
    plot_layer_comparison(mask, layers=layers, save_path=save_path)
    print(f"  Saved: {save_path}")

    # 4. Attention pattern (causal triangle) heatmaps
    # Per-layer attention patterns (opt-in)
    if per_layer:
        print("\nGenerating per-layer attention pattern heatmaps...")
        for layer in layers:
            save_path = os.path.join(output_dir, f"attn_pattern_layer_{layer}.png")
            plot_attention_pattern(mask, layer=layer, threshold=threshold, save_path=save_path)
            print(f"  Saved: {save_path}")

    # All-layers aggregated attention pattern
    save_path = os.path.join(output_dir, "attn_pattern_all_layers.png")
    plot_attention_pattern(mask, layer=None, threshold=threshold, save_path=save_path)
    print(f"  Saved: {save_path}")

    # 5. Circuit graphs
    print("\nGenerating circuit graphs...")
    # Aggregated across all layers
    save_path = os.path.join(output_dir, f"circuit_all_layers_t{threshold}.png")
    plot_circuit_graph(mask, threshold=threshold, layer=None, save_path=save_path)
    print(f"  Saved: {save_path}")

    # Per-layer circuit graphs (opt-in)
    if per_layer:
        for layer in layers:
            save_path = os.path.join(
                output_dir, f"circuit_layer_{layer}_t{threshold}.png"
            )
            plot_circuit_graph(mask, threshold=threshold, layer=layer, save_path=save_path)
            print(f"  Saved: {save_path}")

    # 6. Full circuit overview (sentences x layers)
    print("\nGenerating full circuit overview...")
    save_path = os.path.join(output_dir, f"full_circuit_t{threshold}.png")
    plot_full_circuit(mask, threshold=threshold, save_path=save_path)
    print(f"  Saved: {save_path}")

    # 7. Sparsity vs KL (if threshold evaluation data exists)
    threshold_eval = mask.metadata.get("threshold_evaluation", [])
    if threshold_eval:
        print("\nGenerating threshold vs sparsity/KL plot...")
        thresholds = [r["threshold"] for r in threshold_eval]
        sparsities = [r["sparsity"] for r in threshold_eval]
        kl_scores = [r["kl_divergence"] for r in threshold_eval]
        save_path = os.path.join(output_dir, "threshold_vs_metrics.png")
        plot_threshold_vs_metrics(thresholds, sparsities, kl_scores, save_path=save_path)

        print("\nGenerating sparsity vs KL plot...")
        save_path = os.path.join(output_dir, "sparsity_vs_kl.png")
        random_kl = [r.get("random_kl_divergence") for r in threshold_eval]
        if any(v is None for v in random_kl):
            random_kl = None
        plot_sparsity_vs_kl_with_random(
            thresholds, sparsities, kl_scores, random_kl, save_path=save_path
        )
        print(f"  Saved: {save_path}")
    else:
        print("\nNo threshold evaluation data found in mask metadata, skipping sparsity-KL plot.")

    # 8. Per-token objective (KL) across branches
    has_per_token = any("per_token_kl" in r for r in threshold_eval)
    if has_per_token:
        print("\nGenerating per-token objective plot...")
        save_path = os.path.join(output_dir, "per_token_kl.png")
        plot_per_token_objective(threshold_eval, save_path=save_path)
        print(f"  Saved: {save_path}")
    else:
        print("\nNo per-token KL data found in mask metadata, skipping per-token plot.")

    # 9. Per-sentence objective (KL) across branches
    has_per_sent = any("per_sentence_kl" in r for r in threshold_eval)
    if has_per_sent:
        print("\nGenerating per-sentence objective plot...")
        save_path = os.path.join(output_dir, "per_sentence_kl.png")
        plot_per_sentence_objective(threshold_eval, save_path=save_path)
        print(f"  Saved: {save_path}")
    else:
        print("\nNo per-sentence KL data found in mask metadata, skipping per-sentence plot.")

    print(f"\nDone! All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize learned circuit masks"
    )
    parser.add_argument(
        "--mask_path", required=True, help="Path to NodeMask JSON file"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to visualize (default: all in mask)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Edge importance threshold for circuit graphs",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=5,
        help="Number of top heads to show per layer",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for plots (default: alongside mask file)",
    )
    parser.add_argument(
        "--per_head",
        action="store_true",
        help="Generate per-head heatmaps (top K heads per layer)",
    )
    parser.add_argument(
        "--per_layer",
        action="store_true",
        help="Generate per-layer plots (aggregated heatmaps, attention patterns, circuit graphs)",
    )
    args = parser.parse_args()
    main(**vars(args))
