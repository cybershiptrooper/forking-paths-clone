import argparse
import os

from utils.masks import EdgewiseMask
from utils.visualization_mask import plot_head_mask_matrix, plot_layer_circuit


def main(mask_json: str, layer: int, head: int, output_dir: str, threshold: float = None):
    mask = EdgewiseMask.from_json(mask_json)
    os.makedirs(output_dir, exist_ok=True)

    fig, _ = plot_head_mask_matrix(mask, layer=layer, head=head, threshold=threshold)
    head_path = os.path.join(output_dir, f"mask_layer{layer}_head{head}.png")
    fig.savefig(head_path, dpi=150, bbox_inches="tight")
    print(f"Saved head mask: {head_path}")

    fig, _ = plot_layer_circuit(mask, layer=layer, aggregation="mean", threshold=threshold)
    layer_path = os.path.join(output_dir, f"mask_layer{layer}_circuit.png")
    fig.savefig(layer_path, dpi=150, bbox_inches="tight")
    print(f"Saved layer circuit: {layer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize learned EAP mask")
    parser.add_argument("--mask_json", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="results/mask_visuals")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    main(**vars(args))
