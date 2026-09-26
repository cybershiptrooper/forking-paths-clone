"""Masked-rollout evaluation for candidate-bank masks (open-ended answers).

Standalone counterpart of ``eval_masked_rollouts.py`` for masks trained
with ``candidate_*`` objectives — kept separate so the letter-probe
script stays untouched while other sweeps run.

Generation procedure is identical to the letter-probe script: install the
mask, drop the stored k continuation sentences, generate a fresh
continuation from the analysis prefix (token-capped), then read out the
answer.  Two readouts replace P(A/B/C/D):

1. Fixed-corpus candidate distribution: teacher-force every bank
   candidate after (context + probe suffix); softmax over sequence
   log-probs -> cluster probabilities -> p_gold, reward gap, and KL
   against the clean fixed-continuation reference.
2. Greedy boxed answer: continue (context + probe suffix + " \\boxed{")
   greedily until the brace closes; the extracted string is graded
   inline when it matches a bank candidate (reusing the bank's verdict)
   and marked ``needs_judge`` otherwise (graded offline — compute nodes
   may lack network access for the judge API).

Contexts evaluated: fixed-k clean, fixed-k masked, n masked rollouts,
n clean rollouts.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.direct_answer_circuit_discovery.eval_log_alpha import (
    _scores_to_log_alpha,
    _build_binary_mask,
    _binary_to_per_layer_masks,
)
from expts.direct_answer_circuit_discovery.eval_candidate_bank import (
    _candidate_logprobs,
    _metrics_from_logprobs,
)
from expts.direct_answer_circuit_discovery.eval_masked_rollouts import (
    _install_mask_on_model,
    _clear_mask_from_model,
    _generate_continuations,
)
from expts.direct_answer_circuit_discovery.build_answer_bank import (
    extract_answer_from_generation,
    normalize_answer,
)


def _greedy_boxed_answer(model, tokenizer, context_ids, suffix_boxed_ids,
                         device, max_new_tokens=64):
    """Greedily generate the boxed answer after context + suffix + ' \\boxed{'."""
    full = torch.cat(
        [context_ids, torch.tensor([suffix_boxed_ids], device=device)], dim=-1,
    )
    out = model.generate(
        input_ids=full,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, full.shape[-1]:], skip_special_tokens=True)
    return extract_answer_from_generation(text), text


def _readout(model, tokenizer, context_ids, continuations, lp_clean_ref,
             answer_ids, num_clusters, target_cluster, counts,
             suffix_boxed_ids, bank_verdicts, device):
    """Both readouts for one context (mask state, if any, already installed)."""
    lp = _candidate_logprobs(model, context_ids, continuations, device)
    metrics = _metrics_from_logprobs(
        lp, lp_clean_ref, answer_ids, num_clusters, target_cluster, counts,
    )
    ans, raw = _greedy_boxed_answer(
        model, tokenizer, context_ids, suffix_boxed_ids, device,
    )
    if ans is None:
        graded = "unparsed"
    elif ans in bank_verdicts:
        graded = "correct" if bank_verdicts[ans] else "wrong"
    else:
        graded = "needs_judge"
    metrics["greedy_answer"] = ans
    metrics["greedy_answer_grade"] = graded
    metrics["greedy_raw_text"] = raw[:200]
    clear_cuda()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_index", type=int, required=True)
    parser.add_argument("--analysis_sentence_step", type=int, required=True)
    parser.add_argument("--sentences_after_prefix", type=int, default=5)
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--mask_mode", default="prefix")
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    set_seed(args.seed)

    nm = NodeMask.from_json(args.mask_path)
    bank_path = nm.metadata.get("answer_bank_path")
    if not bank_path:
        raise ValueError(
            f"{args.mask_path} has no answer_bank_path metadata — use "
            "eval_masked_rollouts.py for letter-probe masks."
        )
    with open(bank_path) as f:
        bank = json.load(f)
    from expts.direct_answer_circuit_discovery.answer_bank_utils import (
        flatten_bank_candidates,
    )
    candidates_meta = bank["candidates"]
    _token_lists, _cluster_ids, _counts = flatten_bank_candidates(bank)
    answer_ids = torch.tensor(_cluster_ids)
    counts = torch.tensor(_counts, dtype=torch.float)
    num_clusters = int(bank["num_clusters"])
    target_cluster = int(bank["target_cluster"])
    bank_verdicts = {c["answer_text"]: c["is_correct"] for c in candidates_meta}

    score_readout = nm.score_readout
    granularity = nm.metadata.get("mask_granularity") or nm.granularity or "pair"
    if granularity != "pair":
        raise NotImplementedError("candidate rollout eval supports pair masks")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prefix_no_k, sentences, _p, _c, _f, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None,
        data_path=args.data_path, prompt_index=args.prompt_index,
        base_answer_type="stored", analysis_timestep=None,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=0, min_sentence_length=10, sentence_chunk=1,
    )
    prefix_with_k, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer, prompt=None,
        data_path=args.data_path, prompt_index=args.prompt_index,
        base_answer_type="stored", analysis_timestep=None,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len_no_k = prefix_no_k.shape[-1]

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    sdpa_handles = install_clean_sdpa_forward(model)

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen_prompt = int(nm.metadata.get("num_frozen_prompt_sentences", 0) or 0)
    prompt_filter = (
        build_prompt_filter(num_frozen_prompt, num_sents, device=device)
        if num_frozen_prompt else None
    )
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter
    )
    valid_filter = ~combined_filter.bool()

    log_alpha = _scores_to_log_alpha(nm.scores, score_readout).to(device)
    binary = _build_binary_mask(log_alpha, "top_k", args.target_sparsity, valid_filter)
    binary_masks = _binary_to_per_layer_masks(binary, layers, num_heads)

    continuations = [
        torch.tensor([ids], device=device) for ids in _token_lists
    ]
    max_cont = max(c.shape[-1] for c in continuations)
    suffix_boxed_ids = tokenizer.encode(
        bank["probe_suffix"] + " \\boxed{", add_special_tokens=False,
    )

    max_total_len = (
        prefix_len_no_k + args.max_new_tokens
        + max(max_cont, len(suffix_boxed_ids) + 64) + 5
    )
    token_to_sent = torch.full(
        (max_total_len,), -1, dtype=torch.long, device=device,
    )
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx

    prefix_no_k_dev = prefix_no_k.to(device)
    prefix_with_k_dev = prefix_with_k.to(device)

    def run_readout(context_ids):
        return _readout(
            model, tokenizer, context_ids, continuations, lp_clean_ref,
            answer_ids, num_clusters, target_cluster, counts,
            suffix_boxed_ids, bank_verdicts, device,
        )

    # ===== Clean baseline (fixed k, no mask) — also the KL reference =====
    print("=== Clean baseline (no mask, fixed k) ===")
    lp_clean_ref = _candidate_logprobs(model, prefix_with_k_dev, continuations, device)
    clean_fixed = run_readout(prefix_with_k_dev)
    print(f"  clean p_gold={clean_fixed['p_gold']:.4f} "
          f"greedy={clean_fixed['greedy_answer']!r} ({clean_fixed['greedy_answer_grade']})")

    # ===== Masked baseline (fixed k) =====
    print("=== Masked baseline (fixed k) ===")
    _install_mask_on_model(model, layers, binary_masks, token_to_sent, combined_filter, True)
    masked_fixed = run_readout(prefix_with_k_dev)
    _clear_mask_from_model(model, layers)
    print(f"  masked p_gold={masked_fixed['p_gold']:.4f} "
          f"greedy={masked_fixed['greedy_answer']!r} ({masked_fixed['greedy_answer_grade']})")

    # ===== Masked rollouts =====
    print(f"=== Masked rollouts ({args.n_rollouts}, temp={args.temperature}) ===")
    _install_mask_on_model(model, layers, binary_masks, token_to_sent, combined_filter, True)
    masked_rollouts = []
    for r in range(args.n_rollouts):
        set_seed(args.seed + r)
        cont = _generate_continuations(
            model, tokenizer, prefix_no_k_dev, 1, args.max_new_tokens,
            args.temperature, device,
        )[0]
        ctx = torch.cat(
            [prefix_no_k_dev, cont["tokens"].to(device).unsqueeze(0)], dim=-1,
        )
        m = run_readout(ctx)
        m["rollout_idx"] = r
        m["n_tokens"] = cont["n_tokens"]
        m["rollout_text"] = cont["text"][:500]
        masked_rollouts.append(m)
        print(f"  rollout {r}: p_gold={m['p_gold']:.4f} "
              f"greedy={m['greedy_answer']!r} ({m['greedy_answer_grade']})")
    _clear_mask_from_model(model, layers)

    # ===== Clean rollouts =====
    print(f"=== Clean rollouts ({args.n_rollouts}) ===")
    clean_rollouts = []
    for r in range(args.n_rollouts):
        set_seed(args.seed + r)
        cont = _generate_continuations(
            model, tokenizer, prefix_no_k_dev, 1, args.max_new_tokens,
            args.temperature, device,
        )[0]
        ctx = torch.cat(
            [prefix_no_k_dev, cont["tokens"].to(device).unsqueeze(0)], dim=-1,
        )
        m = run_readout(ctx)
        m["rollout_idx"] = r
        m["n_tokens"] = cont["n_tokens"]
        m["rollout_text"] = cont["text"][:500]
        clean_rollouts.append(m)
        print(f"  clean rollout {r}: p_gold={m['p_gold']:.4f} "
              f"greedy={m['greedy_answer']!r} ({m['greedy_answer_grade']})")

    remove_handles(sdpa_handles)

    out = {
        "eval_kind": "candidate_bank_rollout",
        "prompt_index": args.prompt_index,
        "mask_path": args.mask_path,
        "answer_bank_path": bank_path,
        "gold_answer": bank.get("gold_answer_normalized"),
        "granularity": granularity,
        "target_sparsity": args.target_sparsity,
        "n_rollouts": args.n_rollouts,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "num_frozen_prompt_sentences": num_frozen_prompt,
        "clean_fixed": clean_fixed,
        "masked_fixed": masked_fixed,
        "masked_rollouts": masked_rollouts,
        "clean_rollouts": clean_rollouts,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
