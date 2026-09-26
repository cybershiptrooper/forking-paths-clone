"""Emit DCM+PID v3 full-grid configs (the 8 early-analysis-point banks not
used in the stability-fix pilot) for one chosen fix variant, holding its
hyperparameters constant across prompts.

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.gen_dcm_v3_full_configs \
        --variant sgdnorm --learning_rate 0.005
"""

import argparse
import json
import os

MANIFEST = "results/cot_termination/early_2200/banks/manifest.json"
BOUNDARY_DIR = "results/cot_termination/early_2200/boundary_data"
OUT_DIR = "expts/cot_termination_circuit_discovery/configs/sweeps/dcm_pid_v3_full"
PILOT_STEMS = {"aqua_p008_s78", "gpqa_p038_s164"}

TEMPLATE = """base_config: expts/cot_termination_circuit_discovery/configs/base.yaml
data_path: {data_path}
prompt_index: {prompt_index}
analysis_sentence_step: {sentence_step}
answer_bank_path: {bank_path}
objective: boundary_stop_prob
masking_algorithm: nodewise_dcm_pid_boundary_hazard_batched
boundary_data_path: {boundary_path}
target_sparsity: 0.25
num_training_steps: 600
log_every: 10
snapshot_sparsities: "0.10,0.25"
pid_max_target_sparsity: 0.25
gradient_checkpointing: true
candidate_batch_size: 2
learning_rate: {lr}
dcm_lr_init: {lr_init}
dcm_lr_warmup_frac: 0.5
dcm_polarization: 0.1
pid_snapshot_hold_steps: 50
output_dir: results/snp_sweep/dcm_pid_v3_full/masks
file_name: {file_name}
{variant_lines}"""

VARIANT_LINES = {
    "sgdnorm": "dcm_task_optimizer: sgd_norm\n",
    "clip": "dcm_max_flips_per_step: 10\ndcm_flip_cap_ramp_mult: 2.0\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANT_LINES), required=True)
    ap.add_argument("--learning_rate", type=float, required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for rec in json.load(open(MANIFEST)):
        stem = os.path.basename(rec["bank_path"]).removesuffix(".json")
        if stem in PILOT_STEMS:
            continue
        boundary_path = f"{BOUNDARY_DIR}/{stem}.json"
        if not os.path.exists(boundary_path):
            print(f"SKIP {stem}: no boundary data at {boundary_path}")
            continue
        file_name = f"dcm3_{args.variant}_{stem}_stop_prob"
        cfg = TEMPLATE.format(
            data_path=rec["data_path"],
            prompt_index=rec["prompt_index"],
            sentence_step=rec["analysis_sentence_step"],
            bank_path=rec["bank_path"],
            boundary_path=boundary_path,
            lr=args.learning_rate,
            lr_init=args.learning_rate / 10,
            file_name=file_name,
            variant_lines=VARIANT_LINES[args.variant],
        )
        path = os.path.join(OUT_DIR, f"{file_name}.yaml")
        with open(path, "w") as f:
            f.write(cfg)
        n += 1
        print(f"wrote {path}")
    print(f"{n} configs")


if __name__ == "__main__":
    main()
