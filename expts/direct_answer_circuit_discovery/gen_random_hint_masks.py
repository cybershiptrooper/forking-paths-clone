"""Random-mask baselines for the hint-removal experiment.

For each selected prompt, takes one learned mask JSON as a schema
template, replaces the scores with seeded uniform-random values,
sets score_readout to raw_score, and saves N copies.  Evaluating these
with eval_log_alpha at the same top-k sparsities gives the matched-
sparsity random baseline.

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.gen_random_hint_masks \
        --mask_dir results/snp_sweep/e2_hint_removal/masks \
        --output_dir results/snp_sweep/e2_hint_random/masks \
        --n_seeds 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # one template per prompt: prefer the tsp50 mask, else any
    by_prompt = {}
    for path in sorted(glob.glob(os.path.join(args.mask_dir, "*.json"))):
        with open(path) as f:
            m = json.load(f)
        pi = m["metadata"]["prompt_index"]
        if pi not in by_prompt or "tsp50" in path:
            by_prompt[pi] = (path, m)

    os.makedirs(args.output_dir, exist_ok=True)
    n = 0
    for pi, (path, m) in sorted(by_prompt.items()):
        scores = m["scores"]
        S = len(scores)
        for si in range(args.n_seeds):
            rng = random.Random(args.seed + 1000 * si + pi)
            m2 = json.loads(json.dumps(m))  # deep copy
            m2["scores"] = [[rng.random() for _ in range(S)] for _ in range(S)]
            m2["metadata"]["score_readout"] = "raw_score"
            m2["metadata"]["random_baseline_seed"] = args.seed + 1000 * si + pi
            m2["metadata"]["random_baseline_template"] = os.path.basename(path)
            out = os.path.join(
                args.output_dir, f"e2_hint_random_p{pi:02d}_seed{si}.json")
            with open(out, "w") as f:
                json.dump(m2, f)
            n += 1
    print(f"Wrote {n} random masks -> {args.output_dir}")


if __name__ == "__main__":
    main()
