"""Evaluate learned masks under freshly sampled rollouts.

Instead of using the fixed k continuation sentences from the dataset,
this script:
  1. Loads a mask and installs it on the model's attention.
  2. Generates N new continuations from the masked prefix via
     model.generate() (mask active during generation).
  3. Appends the probe suffix to each generated continuation.
  4. Runs a forward pass (mask still active) to get P(A/B/C/D).
  5. Compares against the fixed-continuation baseline.

This tests whether the mask preserves the model's output distribution
when the model is allowed to generate its own reasoning under the mask,
rather than being fed the original (unmasked) reasoning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import (
    install_clean_sdpa_forward,
    remove_handles,
)
from utils.utils import set_seed, clear_cuda, get_attention_module
from expts.direct_answer_circuit_discovery.probe import (
    answer_probs_from_logits,
    build_answer_probe,
    DEFAULT_SUFFIX,
    DEFAULT_ANSWER_LETTERS,
)
from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix,
    load_model_eager,
)
from expts.direct_answer_circuit_discovery.eval_log_alpha import (
    _scores_to_log_alpha,
    _build_binary_mask,
    _binary_to_per_layer_masks,
    _kl,
)


def _install_mask_on_model(
    model, layers, binary_masks, token_to_sent, combined_filter,
    renormalize,
):
    """Set mask attributes on each layer for SDPA forward to pick up."""
    for l in layers:
        attn = get_attention_module(model, l)
        attn._circuit_mask = binary_masks[l]
        attn._token_to_sent = token_to_sent
        attn._gap_filter = combined_filter
        attn._renormalize_masked_attn = renormalize


def _clear_mask_from_model(model, layers):
    """Remove mask attributes so clean forwards are unaffected."""
    for l in layers:
        attn = get_attention_module(model, l)
        for attr in ("_circuit_mask", "_token_to_sent", "_gap_filter",
                      "_renormalize_masked_attn"):
            if hasattr(attn, attr):
                delattr(attn, attr)


def _generate_continuations(
    model, tokenizer, prefix_ids, n_rollouts, max_new_tokens,
    temperature, device,
):
    """Generate N continuations from the prefix (hooks must already be installed)."""
    input_ids = prefix_ids.to(device)
    prefix_len = input_ids.shape[-1]

    outputs = model.generate(
        input_ids=input_ids.expand(n_rollouts, -1),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    continuations = []
    for i in range(n_rollouts):
        new_tokens = outputs[i, prefix_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        continuations.append({
            "tokens": new_tokens.cpu(),
            "text": text,
            "n_tokens": len(new_tokens),
        })
    return continuations


def _eval_with_continuation(
    model, full_input, probe, prefix_len, device,
):
    """Run forward pass and extract answer probs."""
    with torch.no_grad():
        logits = model(full_input.to(device)).logits
    p = answer_probs_from_logits(logits, probe, prefix_len).cpu()
    del logits
    return p


def build_parser():
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
    parser.add_argument(
        "--force_freeze_prompt", action="store_true",
        help="Freeze all prompt-sentence edges at 1.0 during evaluation, "
        "even if the mask was not trained with freeze_prompt_sentences.",
    )
    parser.add_argument(
        "--answer_letters", type=str, default=None,
        help="Comma-separated probe letters override (e.g. ' A, B, C, D, E'). "
        "Needed for masks without probe metadata (thought anchors) on "
        "datasets whose choice count differs from the 4-letter default.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_rollout_eval(args)


def run_rollout_eval(args, ctx=None):
    """Run one masked-rollout evaluation.

    ``ctx`` is an optional dict used to reuse a loaded model across calls
    (batch mode); it caches the model, tokenizer and installed SDPA
    forwards under the model name. When it is None the model is loaded
    and the SDPA patches removed within this call, exactly as the CLI
    entry point has always behaved.
    """
    set_seed(args.seed)

    # ----- Load mask -----
    nm = NodeMask.from_json(args.mask_path)
    score_readout = nm.score_readout
    granularity = nm.metadata.get("mask_granularity") or nm.granularity or "pair"

    # ----- Build prefix WITHOUT extra k sentences (we'll generate them) -----
    cached = ctx.setdefault(args.model_name, {}) if ctx is not None else {}
    if "tokenizer" in cached:
        tokenizer = cached["tokenizer"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        cached["tokenizer"] = tokenizer
    prefix_ids_no_k, sentences_to_be_masked, prompt, correct_answer, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer,
        prompt=None,
        data_path=args.data_path,
        prompt_index=args.prompt_index,
        base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=0,
        min_sentence_length=10,
        sentence_chunk=1,
    )

    # Also build prefix WITH k sentences (for the fixed-continuation baseline)
    prefix_ids_with_k, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer,
        prompt=None,
        data_path=args.data_path,
        prompt_index=args.prompt_index,
        base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        min_sentence_length=10,
        sentence_chunk=1,
    )

    sentences = sentences_to_be_masked
    num_sents = len(sentences)
    prefix_len_no_k = prefix_ids_no_k.shape[-1]
    prefix_len_with_k = prefix_ids_with_k.shape[-1]

    # The original k continuation tokens
    original_k_tokens = prefix_ids_with_k[0, prefix_len_no_k:]
    original_k_text = tokenizer.decode(original_k_tokens, skip_special_tokens=False)

    # ----- Build log_alpha and binary mask -----
    if granularity == "pair":
        log_alpha = _scores_to_log_alpha(nm.scores, score_readout)
    elif granularity == "layer":
        num_layers_mask = len(nm.scores)
        model_tmp, _ = load_model_eager(args.model_name, device="cpu")
        num_heads = model_tmp.config.num_attention_heads
        del model_tmp
        layer_tensors = []
        for l in range(num_layers_mask):
            la_2d = _scores_to_log_alpha(nm.scores[l], score_readout)
            layer_tensors.append(la_2d.unsqueeze(0).expand(num_heads, -1, -1))
        log_alpha = torch.stack(layer_tensors)
    elif granularity == "head":
        model_tmp, _ = load_model_eager(args.model_name, device="cpu")
        num_heads = model_tmp.config.num_attention_heads
        num_layers_mask = model_tmp.config.num_hidden_layers
        del model_tmp
        layer_tensors = []
        for l in range(num_layers_mask):
            head_tensors = [_scores_to_log_alpha(nm.scores[l][h], score_readout) for h in range(num_heads)]
            layer_tensors.append(torch.stack(head_tensors))
        log_alpha = torch.stack(layer_tensors)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")

    print(f"  Granularity: {granularity}, log_alpha shape: {tuple(log_alpha.shape)}")

    # ----- Load model -----
    if "model" in cached:
        model = cached["model"]
        sdpa_handles = cached["sdpa_handles"]
        print(f"  Reusing loaded {args.model_name} "
              f"({len(sdpa_handles)} SDPA-patched layers)")
    else:
        model, _ = load_model_eager(args.model_name, device="cuda")
        sdpa_handles = install_clean_sdpa_forward(model)
        print(f"  SDPA patched {len(sdpa_handles)} layers")
        cached["model"] = model
        cached["sdpa_handles"] = sdpa_handles
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads

    # ----- Filters -----
    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen_prompt = int(nm.metadata.get("num_frozen_prompt_sentences", 0) or 0)
    if args.force_freeze_prompt and not num_frozen_prompt:
        num_frozen_prompt = num_prompt_sentences
        print(f"  --force_freeze_prompt: freezing {num_frozen_prompt} prompt sentences")
    prompt_filter = (
        build_prompt_filter(num_frozen_prompt, num_sents, device=device)
        if num_frozen_prompt else None
    )
    if num_frozen_prompt:
        print(f"  Frozen prompt sentences: {num_frozen_prompt}")
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter, prompt_filter)
    valid_filter = ~combined_filter.bool()

    # ----- Probe -----
    suffix = nm.metadata.get("probe_suffix", DEFAULT_SUFFIX)
    if args.answer_letters:
        answer_letters = [s for s in args.answer_letters.split(",")]
    else:
        answer_letters = nm.metadata.get("answer_letters") or list(DEFAULT_ANSWER_LETTERS)
    probe = build_answer_probe(tokenizer, suffix=suffix, answer_letters=answer_letters)
    target_letter = nm.metadata.get("target_letter") or correct_answer
    stripped = [l.strip() for l in probe.answer_letters]
    target_idx = stripped.index(target_letter.strip()) if target_letter and target_letter.strip() in stripped else None

    # ----- Build binary mask -----
    log_alpha_dev = log_alpha.to(device)
    binary = _build_binary_mask(log_alpha_dev, "top_k", args.target_sparsity, valid_filter)
    per_layer = granularity in ("layer", "head")
    binary_masks = _binary_to_per_layer_masks(binary, layers, num_heads)

    # ----- token_to_sent: allocate enough for prefix + generation + suffix -----
    max_total_len = prefix_len_no_k + args.max_new_tokens + probe.suffix_len + 5
    token_to_sent = torch.full((max_total_len,), -1, dtype=torch.long, device=device)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx

    # Also build token_to_sent for the fixed-continuation (with k) eval
    max_total_len_k = prefix_len_with_k + probe.suffix_len + 5
    token_to_sent_k = torch.full((max_total_len_k,), -1, dtype=torch.long, device=device)
    for idx, sent in enumerate(sentences):
        token_to_sent_k[sent.start : sent.end + 1] = idx

    # ========== CLEAN BASELINE (no mask) ==========
    print("\n=== Clean baseline (no mask, fixed k sentences) ===")
    continuation = probe.make_continuation(device)
    full_clean = torch.cat([prefix_ids_with_k.to(device), continuation], dim=-1)
    clean_p = _eval_with_continuation(model, full_clean, probe, prefix_len_with_k, device)
    print(f"  Clean P(answer): {clean_p.tolist()}")
    if target_idx is not None:
        print(f"  Clean P(correct={target_letter}): {clean_p[target_idx].item():.4f}")
    clear_cuda()

    # ========== MASKED BASELINE (fixed k sentences) ==========
    print("\n=== Masked baseline (mask on, fixed k sentences) ===")
    _install_mask_on_model(model, layers, binary_masks, token_to_sent_k, combined_filter, True)
    masked_fixed_p = _eval_with_continuation(model, full_clean, probe, prefix_len_with_k, device)
    print(f"  Masked-fixed P(answer): {masked_fixed_p.tolist()}")
    if target_idx is not None:
        print(f"  Masked-fixed P(correct): {masked_fixed_p[target_idx].item():.4f}")
    kl_fixed = _kl(clean_p, masked_fixed_p)
    print(f"  KL(clean || masked-fixed): {kl_fixed:.6f}")
    _clear_mask_from_model(model, layers)
    clear_cuda()

    # ========== MASKED ROLLOUTS (generate k sentences, then probe) ==========
    print(f"\n=== Masked rollouts ({args.n_rollouts} rollouts, temp={args.temperature}) ===")
    _install_mask_on_model(model, layers, binary_masks, token_to_sent, combined_filter, True)

    rollout_results = []
    for r in range(args.n_rollouts):
        set_seed(args.seed + r)
        conts = _generate_continuations(
            model, tokenizer, prefix_ids_no_k, 1, args.max_new_tokens,
            args.temperature, device,
        )
        cont = conts[0]

        # Build full input: prefix (no k) + generated continuation + probe suffix
        gen_tokens = cont["tokens"].to(device).unsqueeze(0)
        full_rollout = torch.cat([
            prefix_ids_no_k.to(device), gen_tokens, continuation
        ], dim=-1)

        # token_to_sent already covers this length (allocated with max_total_len)
        rollout_p = _eval_with_continuation(model, full_rollout, probe,
                                            prefix_len_no_k + cont["n_tokens"], device)
        kl_rollout = _kl(clean_p, rollout_p)

        result = {
            "rollout_idx": r,
            "n_tokens": cont["n_tokens"],
            "text": cont["text"][:500],
            "answer_probs": rollout_p.tolist(),
            "p_correct": float(rollout_p[target_idx].item()) if target_idx is not None else None,
            "kl_vs_clean": kl_rollout,
        }
        rollout_results.append(result)
        print(f"  Rollout {r}: P(correct)={result['p_correct']:.4f}, KL={kl_rollout:.6f}, "
              f"tokens={cont['n_tokens']}")

    _clear_mask_from_model(model, layers)
    clear_cuda()

    # ========== CLEAN ROLLOUTS (no mask, for comparison) ==========
    print(f"\n=== Clean rollouts (no mask, {args.n_rollouts} rollouts) ===")
    clean_rollout_results = []
    for r in range(args.n_rollouts):
        set_seed(args.seed + r)
        conts = _generate_continuations(
            model, tokenizer, prefix_ids_no_k, 1, args.max_new_tokens,
            args.temperature, device,
        )
        cont = conts[0]
        gen_tokens = cont["tokens"].to(device).unsqueeze(0)
        full_rollout = torch.cat([
            prefix_ids_no_k.to(device), gen_tokens, continuation
        ], dim=-1)
        rollout_p = _eval_with_continuation(model, full_rollout, probe,
                                            prefix_len_no_k + cont["n_tokens"], device)
        kl_rollout = _kl(clean_p, rollout_p)

        result = {
            "rollout_idx": r,
            "n_tokens": cont["n_tokens"],
            "text": cont["text"][:500],
            "answer_probs": rollout_p.tolist(),
            "p_correct": float(rollout_p[target_idx].item()) if target_idx is not None else None,
            "kl_vs_clean": kl_rollout,
        }
        clean_rollout_results.append(result)
        print(f"  Clean rollout {r}: P(correct)={result['p_correct']:.4f}, KL={kl_rollout:.6f}, "
              f"tokens={cont['n_tokens']}")

    # ========== Summary ==========
    import numpy as np
    masked_pcorrects = [r["p_correct"] for r in rollout_results if r["p_correct"] is not None]
    masked_kls = [r["kl_vs_clean"] for r in rollout_results]
    clean_pcorrects = [r["p_correct"] for r in clean_rollout_results if r["p_correct"] is not None]
    clean_kls = [r["kl_vs_clean"] for r in clean_rollout_results]

    print(f"\n{'='*60}")
    print(f"SUMMARY (prompt {args.prompt_index}, sparsity {args.target_sparsity})")
    print(f"{'='*60}")
    print(f"  Clean P(correct) [fixed k]:      {clean_p[target_idx].item():.4f}")
    print(f"  Masked P(correct) [fixed k]:     {masked_fixed_p[target_idx].item():.4f}")
    print(f"  Masked P(correct) [rollouts]:    {np.median(masked_pcorrects):.4f} "
          f"(std={np.std(masked_pcorrects):.4f})")
    print(f"  Clean P(correct) [rollouts]:     {np.median(clean_pcorrects):.4f} "
          f"(std={np.std(clean_pcorrects):.4f})")
    print(f"  KL [masked fixed]:               {kl_fixed:.6f}")
    print(f"  KL [masked rollouts] median:     {np.median(masked_kls):.6f}")
    print(f"  KL [clean rollouts] median:      {np.median(clean_kls):.6f}")
    print(f"  Original k text (first 200):     {original_k_text[:200]}")

    # Save
    out = {
        "prompt_index": args.prompt_index,
        "mask_path": args.mask_path,
        "granularity": granularity,
        "target_sparsity": args.target_sparsity,
        "n_rollouts": args.n_rollouts,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "target_letter": target_letter,
        "clean_answer_probs": clean_p.tolist(),
        "masked_fixed_answer_probs": masked_fixed_p.tolist(),
        "kl_masked_fixed": kl_fixed,
        "masked_rollouts": rollout_results,
        "clean_rollouts": clean_rollout_results,
        "original_k_text": original_k_text[:1000],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")

    _clear_mask_from_model(model, layers)
    if ctx is None:
        remove_handles(sdpa_handles)
    clear_cuda()


if __name__ == "__main__":
    main()
