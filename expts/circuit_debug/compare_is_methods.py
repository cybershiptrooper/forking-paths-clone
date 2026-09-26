"""Compare IS methods on an existing mask's saved log_weights.

Usage:
    uv run python -m expts.circuit_debug.compare_is_methods \\
        results/circuit_discovery/.../mask.json [--chain_lengths T1,T2,...]

By default reads chain lengths from the cached completions (via cache_key in
metadata). Override with --chain_lengths for quick what-if comparisons.
"""
import argparse
import json
from pathlib import Path

import torch

from utils.importance_sampling import importance_weights
from utils.completion_cache import load_from_cache, DEFAULT_CACHE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_path")
    parser.add_argument(
        "--chain_lengths", type=str, default=None,
        help="Comma-separated override for chain lengths.",
    )
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    data = json.loads(Path(args.mask_path).read_text())
    meta = data["metadata"]

    if args.chain_lengths:
        lengths = torch.tensor(
            [int(x) for x in args.chain_lengths.split(",")], dtype=torch.long,
        )
    else:
        cached = load_from_cache(meta["cache_key"], args.cache_dir)
        if cached is None:
            raise SystemExit(
                f"Cache for key {meta['cache_key']!r} not found in {args.cache_dir}"
            )
        lengths = torch.tensor(
            [len(b["token_ids"]) for b in cached["branches"]], dtype=torch.long,
        )

    print(
        f"Chains: {len(lengths)}  lengths: "
        f"min={lengths.min().item()} max={lengths.max().item()} "
        f"mean={lengths.float().mean().item():.1f}"
    )
    print()

    for entry in meta["threshold_evaluation"]:
        lw = entry.get("log_weights")
        if lw is None:
            continue
        lw = torch.tensor(lw, dtype=torch.float32)
        # Fake log_p_target = lw, log_p_proposal = 0: importance_weights uses
        # the difference (log_p_target - log_p_proposal), so this recovers the
        # same weights without needing the original two tensors.
        zeros = torch.zeros_like(lw)
        w_snis = importance_weights(lw, zeros, method="snis")
        w_geo = importance_weights(
            lw, zeros, method="geometric_mean", chain_lengths=lengths,
        )
        print(
            f"sparsity={entry['sparsity']:.2%}  "
            f"SNIS:     top1={w_snis.max().item():.3f}  "
            f"N_eff={1 / (w_snis**2).sum().item():.1f}"
        )
        print(
            f"                        GeoMean:  top1={w_geo.max().item():.3f}  "
            f"N_eff={1 / (w_geo**2).sum().item():.1f}"
        )


if __name__ == "__main__":
    main()
