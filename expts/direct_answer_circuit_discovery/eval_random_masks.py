"""Random-mask baseline: evaluate uniformly sampled binary masks.

For one prompt, at each requested target sparsity, samples N random
binary masks over the LEARNABLE pool only — the cells that survive the
gap / mode / causal filters and, when the prompt is frozen
(``--force_freeze_prompt``), the prompt filter — and evaluates each with
the same masked forward used by ``eval_log_alpha.py``. Frozen /
filtered cells are pinned at full attention by the hook layer exactly
as in every other evaluation, so a "random mask at sparsity s" ablates
``round(s * n_valid)`` uniformly chosen learnable edges and nothing
else. Reports per-sample KL and the mean over samples, raw and
normalized by the same all-learnable-edges-ablated ceiling.

Sampling is deterministic: the RNG is seeded from
(seed, prompt_index, target sparsity, sample index), so reruns
reproduce the same masks.

Usage (one process = one prompt, all sparsities):
  uv run python -m expts.direct_answer_circuit_discovery.eval_random_masks \\
      --model_name Qwen/Qwen3-8B \\
      --data_path data/collection/qwen3_8b/gpqa_filtered.json \\
      --prompt_index 27 --analysis_sentence_step 50 \\
      --sentences_after_prefix 5 --sentence_gap 1 \\
      --force_freeze_prompt \\
      --sparsities 0.01,0.05,0.2,...,0.99 --n_samples 3 \\
      --output out.random_eval.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from utils.masks import (
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.probe import (
    answer_probs_from_logits, build_answer_probe, DEFAULT_SUFFIX,
    DEFAULT_ANSWER_LETTERS,
)
from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.direct_answer_circuit_discovery.eval_log_alpha import (
    _evaluate_mask, _kl, _all_zero_mask,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--prompt_index", type=int, required=True)
    p.add_argument("--analysis_sentence_step", type=int, required=True)
    p.add_argument("--sentences_after_prefix", type=int, default=5)
    p.add_argument("--sentence_gap", type=int, default=1)
    p.add_argument("--mask_mode", default="prefix")
    p.add_argument("--sparsities", required=True,
                   help="Comma-separated target sparsities.")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--force_freeze_prompt", action="store_true",
                   help="Freeze all prompt-sentence edges at 1.0 and exclude "
                   "them from the samplable pool (the frozen treatment). "
                   "Omit for the maskable treatment: prompt edges are part "
                   "of the pool and can be ablated by the random mask.")
    p.add_argument("--answer_letters", type=str, default=None,
                   help="Comma-separated probe letters override "
                   "(e.g. ' A, B, C, D, E' for AQuA).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    set_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    (prefix_ids, sentences, _, correct_answer, _,
     num_prompt_sentences) = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=args.data_path,
        prompt_index=args.prompt_index, base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    if args.answer_letters:
        answer_letters = [s for s in args.answer_letters.split(",")]
    else:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)
    probe = build_answer_probe(tokenizer, suffix=DEFAULT_SUFFIX,
                               answer_letters=answer_letters)

    model, _ = load_model_eager(args.model_name, device=args.device)
    device = next(model.parameters()).device
    full_input = torch.cat(
        [prefix_ids.to(device), probe.make_continuation(device)], dim=-1)
    full_len = full_input.shape[-1]
    sdpa_handles = install_clean_sdpa_forward(model)

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode,
                                    device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen = num_prompt_sentences if args.force_freeze_prompt else 0
    prompt_filter = (build_prompt_filter(num_frozen, num_sents, device=device)
                     if num_frozen else None)
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter)
    valid = ~combined_filter.bool()
    valid_idx = valid.flatten().nonzero(as_tuple=True)[0].cpu().numpy()
    n_valid = len(valid_idx)
    print(f"  learnable pool: {n_valid} cells "
          f"({'frozen' if num_frozen else 'maskable'} prompt, "
          f"{num_frozen} frozen prompt sentences)")

    token_to_sent = torch.full((full_len,), -1, dtype=torch.long, device=device)
    for i, sent in enumerate(sentences):
        token_to_sent[sent.start:sent.end + 1] = i

    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads

    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
    del clean_logits
    clear_cuda()

    p_zero = _evaluate_mask(
        model, layers, num_heads, full_input, prefix_len, probe,
        _all_zero_mask(num_sents, len(layers), num_heads, device),
        token_to_sent, combined_filter, device, True, "sdpa")
    kl_max = _kl(clean_p, p_zero)
    print(f"  kl_max = {kl_max:.4f}")

    rows = []
    for s in [float(x) for x in args.sparsities.split(",") if x.strip()]:
        # matched-size keep set: identical accounting to the top-K evaluator
        n_keep = max(0, int(round((1.0 - s) * n_valid)))
        kls = []
        for i in range(args.n_samples):
            rng = np.random.default_rng(
                (args.seed, args.prompt_index, int(round(s * 1000)), i))
            keep = rng.choice(valid_idx, size=n_keep, replace=False)
            binary = torch.zeros(num_sents * num_sents, device=device)
            binary[torch.as_tensor(keep, device=device)] = 1.0
            binary = binary.view(num_sents, num_sents)
            p_masked = _evaluate_mask(
                model, layers, num_heads, full_input, prefix_len, probe,
                binary, token_to_sent, combined_filter, device, True, "sdpa")
            kls.append(_kl(clean_p, p_masked))
        mean_kl = float(np.mean(kls))
        rows.append({
            "target_sparsity": s,
            "n_keep": n_keep,
            "sample_kls": kls,
            "mean_kl": mean_kl,
            "mean_kl_normalized": mean_kl / kl_max if kl_max > 0 else None,
            "median_kl": float(np.median(kls)),
        })
        print(f"  sp={s}: mean KL {mean_kl:.5f} "
              f"(norm {rows[-1]['mean_kl_normalized']})")

    remove_handles(sdpa_handles)
    out = {
        "baseline": "random_mask_mean_of_samples",
        "model_name": args.model_name,
        "data_path": args.data_path,
        "prompt_index": args.prompt_index,
        "analysis_sentence_step": args.analysis_sentence_step,
        "sentences_after_prefix": args.sentences_after_prefix,
        "sentence_gap": args.sentence_gap,
        "mask_mode": args.mask_mode,
        "num_frozen_prompt_sentences": num_frozen,
        "n_valid": n_valid,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "answer_letters": probe.answer_letters,
        "correct_answer": correct_answer,
        "clean_answer_probs": clean_p.tolist(),
        "kl_max": kl_max,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
