"""Training configs, Thought Anchors configs and the edge-ablation manifest
for every example of the mask-assisted judge dataset.

Per example (one prompt, one rollout, its analysis point and prompt chunk
spans from the record):

- ``p2t_rg``: subnetwork probing over the prompt-to-trace pool, reward-gap
  objective towards the trace's own answer (``mask_target_letter``), 80
  percent of the pool removed;
- ``p2t_kl``: the same pool and budget with the answer-KL objective (keep
  the answer distribution);
- ``trace_rg``: subnetwork probing over reasoning-to-reasoning cells only
  (prompt frozen), reward gap towards the trace's answer, 80 percent removed;
- ``ta``: Thought Anchors (attention suppression) scores over the
  prompt-to-trace pool (binarised at 80 percent when summarised);
- ``attribution``: direct column and single-cell ablations of the
  prompt-to-trace pool (``eval_prompt_edges.py``), the attribution table of
  the earlier judge report.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.gen_mask_configs \
        --dataset results/prompt_bias_masks/dataset_qwen3_8b.json --name masks_qwen3_8b
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from expts.prompt_bias_circuit_discovery.gen_sweep_configs import CANONICAL

METHODS_SNP = {
    "p2t_rg": dict(learnable_region="prompt_to_trace", objective="answer_probe_reward_gap"),
    "p2t_kl": dict(learnable_region="prompt_to_trace", objective="answer_probe_kl"),
    # reasoning-only mask (prompt frozen): kept as a to-do, written only with --with_trace_rg
    "trace_rg": dict(freeze_prompt_sentences=True, objective="answer_probe_reward_gap"),
}
# the prefix is the whole reasoning up to (not including) </think>; the probe forces
# "</think>\n\n**Final answer (" and reads the letter logits at the next token
PROBE_SUFFIX = "</think>\n\n**Final answer ("


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--target_sparsity", type=float, default=0.8)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--only", nargs="*", default=None, help="example ids to write (default all)")
    ap.add_argument("--with_trace_rg", action="store_true")
    args = ap.parse_args()
    D = json.load(open(args.dataset))
    root = f"expts/prompt_bias_circuit_discovery/configs/{args.name}"
    res = f"results/prompt_bias_masks/{args.name}"
    for d in [f"{root}/snp", f"{root}/ta", f"{res}/masks", f"{res}/masks_ta", f"{res}/edges"]:
        os.makedirs(d, exist_ok=True)
    only = set(int(x) for x in args.only) if args.only else None
    edges = []
    n_snp = n_ta = 0
    for rec in D:
        i = rec["example_id"]
        if only is not None and i not in only:
            continue
        letters = list(rec["all_letters"])  # no leading space: the suffix ends in "("
        common = dict(model_name=args.model_name, data_path=args.dataset, prompt_index=i,
                      analysis_timestep=rec["analysis_timestep"], sentences_after_prefix=0, answer_letters=letters)
        for m, spec in METHODS_SNP.items():
            if m == "trace_rg" and not args.with_trace_rg:
                continue
            name = f"ex{i:03d}_{m}"
            cfg = dict(CANONICAL, **common, probe_suffix=PROBE_SUFFIX, target_sparsity=args.target_sparsity,
                       output_dir=f"{res}/masks", file_name=name, **spec)
            if spec["objective"] == "answer_probe_reward_gap":
                cfg["target_letter"] = rec["mask_target_letter"]
            with open(f"{root}/snp/{name}.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n_snp += 1
        ta = dict(model_name=args.model_name, **{k: v for k, v in common.items() if k not in ("answer_letters", "model_name")},
                  sentence_gap=1, sentence_chunk=1, mask_mode="prefix", learnable_region="prompt_to_trace", device="cuda", seed=42,
                  output_dir=f"{res}/masks_ta", file_name=f"ex{i:03d}")
        with open(f"{root}/ta/ex{i:03d}_ta.yaml", "w") as f:
            yaml.dump(ta, f, default_flow_style=False, sort_keys=False)
        n_ta += 1
        edges.append(dict(data_path=args.dataset, prompt_index=i, analysis_timestep=rec["analysis_timestep"], sentence_gap=1,
                          target_letter=rec["mask_target_letter"], answer_letters=",".join(rec["all_letters"]), probe_suffix=PROBE_SUFFIX,
                          output=f"{res}/edges/ex{i:03d}.edges.json"))
    with open(f"{res}/edges_manifest.jsonl", "w") as f:
        f.write("\n".join(json.dumps(x) for x in edges) + "\n")
    print(f"{n_snp} SNP configs -> {root}/snp; {n_ta} TA configs -> {root}/ta; {len(edges)} edge tasks -> {res}/edges_manifest.jsonl")


if __name__ == "__main__":
    main()
