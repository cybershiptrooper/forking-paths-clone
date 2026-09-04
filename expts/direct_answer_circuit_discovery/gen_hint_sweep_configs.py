"""Generate sweep configs for the hint-removal experiment.

Writes, per selected prompt:
- one SNP config per target sparsity (objective answer_probe_reward_gap,
  target letter = the control arm's modal answer, hint frozen in the
  prompt) under configs/sweeps/e2_hint_removal/
- one flat Thought Anchors config under configs/sweeps/e2_hint_ta/

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.gen_hint_sweep_configs \
        --selection results/hint_removal/selection.json \
        --data_path data/collection/qwen3_8b/hinted_gpqa_aqua.json
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

CANONICAL = dict(
    mode="learn",
    masking_algorithm="nodewise_subnetwork_probing_sdpa",
    objective="answer_probe_reward_gap",
    model_name="Qwen/Qwen3-8B",
    probe_suffix=" </think> I think the answer is",
    mask_mode="prefix",
    mask_granularity="pair",
    pair_aggregation="mean",
    sentence_gap=0,
    sentence_chunk=1,
    layers_to_analyse="all",
    freeze_prompt_sentences=True,
    renormalize_masked_attention=True,
    gradient_checkpointing=True,
    sparsity_loss_mode="target_size_relu",
    optimizer="hybrid",
    save_log_alpha=True,
    l0_lambda=100,
    learning_rate=0.1,
    log_alpha_init=2.0,
    num_training_steps=1000,
    log_every=20,
    seed=42,
    device="cuda",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--sweep_name", default="e2_hint_removal")
    ap.add_argument("--target_sparsities", nargs="+", type=float,
                    default=[0.05, 0.2, 0.5])
    args = ap.parse_args()

    with open(args.selection) as f:
        selected = json.load(f)["selected"]

    snp_dir = f"expts/direct_answer_circuit_discovery/configs/sweeps/{args.sweep_name}"
    ta_dir = f"expts/direct_answer_circuit_discovery/configs/sweeps/e2_hint_ta"
    os.makedirs(snp_dir, exist_ok=True)
    os.makedirs(ta_dir, exist_ok=True)

    n = 0
    for sel in selected:
        pi = sel["prompt_index"]
        letters = [" " + l for l in sel["all_letters"]]
        for tsp in args.target_sparsities:
            name = f"{args.sweep_name}_p{pi:02d}_tsp{int(round(tsp * 100)):02d}"
            cfg = dict(
                CANONICAL,
                data_path=args.data_path,
                prompt_index=pi,
                analysis_sentence_step=sel["analysis_sentence_step"],
                sentences_after_prefix=5,
                answer_letters=letters,
                target_letter=sel["target_letter"],
                target_sparsity=tsp,
                output_dir=f"results/snp_sweep/{args.sweep_name}/masks",
                file_name=name,
            )
            with open(os.path.join(snp_dir, f"{name}.yaml"), "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n += 1
        ta_cfg = dict(
            model_name="Qwen/Qwen3-8B",
            data_path=args.data_path,
            prompt_index=pi,
            analysis_sentence_step=sel["analysis_sentence_step"],
            sentences_after_prefix=5,
            sentence_gap=0,
            sentence_chunk=1,
            mask_mode="prefix",
            device="cuda",
            seed=42,
            output_dir="results/snp_sweep/e2_hint_ta/masks",
            file_name=f"e2_hint_ta_p{pi:02d}",
        )
        with open(os.path.join(ta_dir, f"e2_hint_ta_p{pi:02d}.yaml"), "w") as f:
            yaml.dump(ta_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {n} SNP configs -> {snp_dir}")
    print(f"Wrote {len(selected)} TA configs -> {ta_dir}")


if __name__ == "__main__":
    main()
