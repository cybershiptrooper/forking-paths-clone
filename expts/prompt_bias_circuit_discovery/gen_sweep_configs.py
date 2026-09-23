"""Write the training configs, baselines and manifests for one prompt set.

Input: a selection JSON, a list of dicts with keys
    tag, data_path, prompt_index, analysis_sentence_step, sentences_after_prefix,
    all_letters (list of bare letters), target_letter (the answer without
    the cue), biased_letter (the answer the stored trace gives).

Per prompt it writes
- SNP configs with the reward-gap objective toward target_letter
  (the mask that flips the answer back), one per --flip_sparsities;
- SNP configs with the answer-KL objective (the mask that keeps the biased
  answer distribution), one per --kl_sparsities;
- one Thought Anchors config;
- one line each in the edge-ablation and random-baseline manifests.
All masks are learned over the prompt-to-trace pool only
(``learnable_region: prompt_to_trace``), sentence_gap 1, the canonical
L2 sparsity loss with lambda 1000 and 4 Hard-Concrete samples per step.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.gen_sweep_configs \
        --selection results/prompt_bias/hint_selection.json --sweep_name p2t_hint
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

CANONICAL = dict(
    mode="learn",
    masking_algorithm="nodewise_subnetwork_probing_region",
    model_name="Qwen/Qwen3-8B",
    probe_suffix=" </think> I think the answer is",
    mask_mode="prefix",
    mask_granularity="pair",
    pair_aggregation="mean",
    sentence_gap=1,
    sentence_chunk=1,
    layers_to_analyse="all",
    learnable_region="prompt_to_trace",
    renormalize_masked_attention=True,
    gradient_checkpointing=True,
    sparsity_loss_mode="target_size_l2",
    optimizer="hybrid",
    save_log_alpha=True,
    l0_lambda=1000.0,
    learning_rate=0.1,
    log_alpha_init=2.0,
    num_training_steps=1000,
    num_hc_samples_per_step=4,
    batch_chunk_size=4,
    log_every=20,
    seed=42,
    device="cuda",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--sweep_name", required=True)
    ap.add_argument("--flip_sparsities", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    ap.add_argument("--kl_sparsities", nargs="+", type=float,
                    default=[0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--all_sparsities", default="0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.95")
    args = ap.parse_args()

    with open(args.selection) as f:
        selected = json.load(f)
    root = "expts/prompt_bias_circuit_discovery/configs/sweeps"
    snp_dir = f"{root}/{args.sweep_name}"
    ta_dir = f"{root}/{args.sweep_name}_ta"
    os.makedirs(snp_dir, exist_ok=True)
    os.makedirs(ta_dir, exist_ok=True)
    res = f"results/prompt_bias/{args.sweep_name}"
    os.makedirs(res, exist_ok=True)
    edge_manifest, random_manifest = [], []
    n = 0
    for sel in selected:
        tag = sel["tag"]
        letters = [" " + l for l in sel["all_letters"]]
        common = dict(
            data_path=sel["data_path"], prompt_index=sel["prompt_index"],
            analysis_sentence_step=sel["analysis_sentence_step"],
            sentences_after_prefix=sel.get("sentences_after_prefix", 5),
            answer_letters=letters,
        )
        for obj, short, sps in [("answer_probe_reward_gap", "rg", args.flip_sparsities),
                                ("answer_probe_kl", "kl", args.kl_sparsities)]:
            for tsp in sps:
                name = f"{args.sweep_name}_{tag}_{short}_tsp{int(round(tsp * 100)):02d}"
                cfg = dict(CANONICAL, objective=obj, **common,
                           target_letter=sel["target_letter"], target_sparsity=tsp,
                           output_dir=f"{res}/masks", file_name=name)
                with open(os.path.join(snp_dir, f"{name}.yaml"), "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                n += 1
        ta_cfg = dict(
            model_name="Qwen/Qwen3-8B", **{k: v for k, v in common.items() if k != "answer_letters"},
            sentence_gap=1, sentence_chunk=1, mask_mode="prefix",
            learnable_region="prompt_to_trace", device="cuda", seed=42,
            output_dir=f"{res}/masks_ta", file_name=f"{args.sweep_name}_{tag}",
        )
        with open(os.path.join(ta_dir, f"{args.sweep_name}_{tag}_ta.yaml"), "w") as f:
            yaml.dump(ta_cfg, f, default_flow_style=False, sort_keys=False)
        edge_manifest.append(dict(
            data_path=sel["data_path"], prompt_index=sel["prompt_index"],
            analysis_sentence_step=sel["analysis_sentence_step"],
            sentences_after_prefix=sel.get("sentences_after_prefix", 5), sentence_gap=1,
            target_letter=sel["target_letter"], answer_letters=",".join(sel["all_letters"]),
            output=f"{res}/edges/{args.sweep_name}_{tag}.edges.json",
        ))
        random_manifest.append(dict(
            data_path=sel["data_path"], prompt_index=sel["prompt_index"],
            analysis_sentence_step=sel["analysis_sentence_step"],
            sentences_after_prefix=sel.get("sentences_after_prefix", 5), sentence_gap=1,
            answer_letters=",".join(sel["all_letters"]), sparsities=args.all_sparsities,
            output=f"{res}/random/{args.sweep_name}_{tag}.random_eval.json",
        ))
    with open(f"{res}/edges_manifest.jsonl", "w") as f:
        f.write("\n".join(json.dumps(x) for x in edge_manifest) + "\n")
    with open(f"{res}/random_manifest.jsonl", "w") as f:
        f.write("\n".join(json.dumps(x) for x in random_manifest) + "\n")
    with open(f"{res}/selection.json", "w") as f:
        json.dump(selected, f, indent=1)
    print(f"Wrote {n} SNP configs -> {snp_dir}; {len(selected)} TA configs -> {ta_dir}")
    print(f"Manifests -> {res}/edges_manifest.jsonl, {res}/random_manifest.jsonl")


if __name__ == "__main__":
    main()
