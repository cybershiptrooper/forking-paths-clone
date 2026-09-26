"""Generate one sweep YAML per (bank, target sparsity).

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.gen_sweep_configs \
        --manifest results/cot_termination/banks/manifest.json \
        --sweep_name e1_termination \
        --target_sparsities 0.2 0.5
"""

import argparse
import json
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sweep_name", default="e1_termination")
    ap.add_argument("--target_sparsities", nargs="+", type=float,
                    default=[0.2, 0.5])
    ap.add_argument("--num_training_steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    out_dir = (f"expts/cot_termination_circuit_discovery/configs/sweeps/"
               f"{args.sweep_name}")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for m in manifest:
        stem = os.path.splitext(os.path.basename(m["bank_path"]))[0]
        for tsp in args.target_sparsities:
            name = f"{args.sweep_name}_{stem}_tsp{int(round(tsp * 100))}"
            cfg = {
                "base_config":
                    "expts/cot_termination_circuit_discovery/configs/base.yaml",
                "data_path": m["data_path"],
                "prompt_index": m["prompt_index"],
                "analysis_sentence_step": m["analysis_sentence_step"],
                "answer_bank_path": m["bank_path"],
                "target_sparsity": tsp,
                "output_dir": f"results/snp_sweep/{args.sweep_name}/masks",
                "file_name": name,
            }
            if args.num_training_steps is not None:
                cfg["num_training_steps"] = args.num_training_steps
            with open(os.path.join(out_dir, f"{name}.yaml"), "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n += 1
    print(f"Wrote {n} configs to {out_dir}")


if __name__ == "__main__":
    main()
