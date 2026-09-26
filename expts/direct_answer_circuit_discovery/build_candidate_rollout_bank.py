"""Clean rollout banks for the answer-distribution task on open-ended answers (MATH).

The open-ended counterpart of stage A of ``eval_onpolicy_kl.py``. From the
stored trace cut at the analysis point (no stored continuation sentences),
sample three independent sets of ``K`` continuations from the unmasked model
(temperature 0.7, at most 200 tokens): set B is the training bank of the
answer-distribution masks, sets A and C are held out. For every continuation
record, over the candidate answer bank built at the same analysis point
(``build_answer_bank.py``):

- the teacher-forced log-probability of every candidate path after
  ``prefix + continuation + probe suffix``, unmasked and with every eligible
  edge removed (frozen prompt sentences, the given sentence gap);
- the resulting cluster distributions and their KL (the per-continuation
  ``kl_max``).

Seeds follow ``eval_onpolicy_kl.py`` (set A = seed, set B = seed + 1000) and
set C uses seed + 2000.

Usage:
  uv run python -m expts.direct_answer_circuit_discovery.build_candidate_rollout_bank \
      --data_path data/collection/qwen3_8b/math_merged_filtered.json --prompt_index 3 \
      --analysis_sentence_step 50 --answer_bank_path results/math_paper/answer_banks/s50/p03.json \
      --output final_results/kl_frozen/qwen3_8b/math/clean_rollouts_k32/math_idx003_s50.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from expts.direct_answer_circuit_discovery.candidate_rollouts import bank_paths, path_logprobs, cluster_probs, kl
from expts.direct_answer_circuit_discovery.eval_log_alpha import _all_zero_mask, _binary_to_per_layer_masks
from expts.direct_answer_circuit_discovery.eval_masked_rollouts import _install_mask_on_model, _clear_mask_from_model
from expts.direct_answer_circuit_discovery.eval_onpolicy_kl import (
    _get_model, _prefix_and_sentences, _sample_continuations, _filters, _token_to_sent, _write,
)
from utils.utils import clear_cuda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--prompt_index", type=int, required=True)
    ap.add_argument("--analysis_sentence_step", type=int, required=True)
    ap.add_argument("--answer_bank_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_rollouts", type=int, default=32)
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--rows_per_forward", type=int, default=8)
    ap.add_argument("--sentence_gap", type=int, default=1)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.answer_bank_path) as f:
        bank = json.load(f)
    if (bank["data_path"], bank["prompt_index"], bank["analysis_sentence_step"]) != (
            args.data_path, args.prompt_index, args.analysis_sentence_step):
        raise ValueError(f"bank {args.answer_bank_path} is for another prompt or analysis point")
    suffix, paths, cluster_ids = bank_paths(bank)
    cids = torch.tensor(cluster_ids)
    nc = int(bank["num_clusters"])

    ctx = {}
    model, tokenizer = _get_model(args.model_name, ctx)
    model.eval()
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    prefix_ids, sentences, correct_answer, n_prompt = _prefix_and_sentences(
        tokenizer, args.data_path, args.prompt_index, args.analysis_sentence_step, 0)
    num_sents = len(sentences)
    combined, num_frozen = _filters({"num_frozen_prompt_sentences": n_prompt}, num_sents,
                                    args.sentence_gap, args.mask_mode, True, n_prompt, device)
    length = prefix_ids.shape[-1] + args.max_new_tokens + len(suffix) + max(len(p) for p in paths) + 5
    t2s = _token_to_sent(sentences, length, device)
    zero = _binary_to_per_layer_masks(_all_zero_mask(num_sents, len(layers), num_heads, device), layers, num_heads)
    suffix_t = torch.tensor([suffix], dtype=torch.long, device=device)
    pre = prefix_ids.to(device)

    sets = {}
    for i, name in enumerate(args.sets.split(",")):
        seed = args.seed + 1000 * i
        toks = _sample_continuations(model, tokenizer, prefix_ids, args.n_rollouts, args.max_new_tokens,
                                     args.temperature, seed, args.gen_batch)
        rows = []
        for t in toks:
            context = torch.cat([pre, torch.tensor([t], dtype=torch.long, device=device), suffix_t], dim=-1)
            lp_c = path_logprobs(model, context, paths, args.rows_per_forward)
            _install_mask_on_model(model, layers, zero, t2s, combined, True)
            lp_z = path_logprobs(model, context, paths, args.rows_per_forward)
            _clear_mask_from_model(model, layers)
            pc, pz = cluster_probs(lp_c, cids, nc), cluster_probs(lp_z, cids, nc)
            rows.append({"tokens": t, "n_tokens": len(t), "text": tokenizer.decode(t, skip_special_tokens=False),
                         "clean_path_logprobs": lp_c.tolist(), "all_zero_path_logprobs": lp_z.tolist(),
                         "clean_cluster_probs": pc.tolist(), "all_zero_cluster_probs": pz.tolist(),
                         "kl_max": float(kl(pc, pz))})
        clear_cuda()
        sets[name] = {"seed": seed, "rollouts": rows}
        mean_pc = torch.tensor([r["clean_cluster_probs"] for r in rows]).mean(0)
        print(f"  set {name}: mean clean cluster probs {[round(x, 3) for x in mean_pc.tolist()]}, "
              f"median kl_max {float(torch.tensor([r['kl_max'] for r in rows]).median()):.4f}")

    _write(args.output, {
        "kind": "candidate_rollout_bank",
        "model_name": args.model_name, "data_path": args.data_path, "prompt_index": args.prompt_index,
        "analysis_sentence_step": args.analysis_sentence_step, "sentence_gap": args.sentence_gap,
        "mask_mode": args.mask_mode, "num_frozen_prompt_sentences": num_frozen, "num_sentences": num_sents,
        "prefix_len": int(prefix_ids.shape[-1]), "probe_suffix": bank["probe_suffix"],
        "probe_suffix_token_ids": suffix, "answer_bank_path": args.answer_bank_path,
        "candidate_paths": paths, "cluster_ids": cluster_ids, "num_clusters": nc,
        "target_cluster": int(bank["target_cluster"]), "gold_answer": bank["gold_answer_normalized"],
        "candidate_answers": [{"answer_text": c["answer_text"], "cluster_id": c["cluster_id"],
                               "is_correct": c["is_correct"]} for c in bank["candidates"]],
        "correct_answer": correct_answer, "n_rollouts": args.n_rollouts, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "gen_batch": args.gen_batch, "sets": sets,
    })
    print("wrote", args.output)


if __name__ == "__main__":
    main()
