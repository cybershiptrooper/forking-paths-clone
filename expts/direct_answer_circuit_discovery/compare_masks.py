"""Compare NodeMasks by their answer-token KL when installed.

Loads N saved NodeMask files, builds the answer probe, and for each
mask:

1. Threshold the score matrix at a target sparsity (or accept the mask
   as-is if it's already binary, e.g. SNP).
2. Install attention hooks via :func:`utils.circuit_eval.install_mask_hooks`.
3. Run a forward pass on ``prefix + suffix + placeholder``.
4. Read the answer-token logits, softmax-renormalise over A/B/C/D, and
   compute ``KL(P_clean || P_masked)`` and per-letter probabilities.

Usage::

    uv run python -m expts.direct_answer_circuit_discovery.compare_masks \\
        --masks results/.../thought_anchors.json \\
                results/.../suppress_on_answer.json \\
                results/.../snp.json \\
        --model_name Qwen/Qwen3-8B \\
        --data_path data/.../gpqa_filtered.json \\
        --prompt_index 32 \\
        --analysis_sentence_step 50 \\
        --sparsities 0.0 0.5 0.7 0.9
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda
from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
)
from utils.circuit_eval import (
    build_token_to_sent_map,
    build_binary_masks,
    install_mask_hooks,
    remove_handles,
)

from expts.direct_answer_circuit_discovery.probe import (
    DEFAULT_ANSWER_LETTERS,
    DEFAULT_SUFFIX,
    build_answer_probe,
)
from expts.direct_answer_circuit_discovery.learn import _build_prefix


def _threshold_for_sparsity(
    node_mask: NodeMask, target_sparsity: float, gap_filter: torch.Tensor,
) -> float:
    """Pick the threshold whose sparsity is closest to *target_sparsity*."""
    g = node_mask.granularity
    flat: List[float] = []
    if g == "pair":
        for i, row in enumerate(node_mask.scores):
            for j, val in enumerate(row):
                if not bool(gap_filter[i, j]):
                    flat.append(float(val))
    elif g == "layer":
        for layer_scores in node_mask.scores.values():
            for i, row in enumerate(layer_scores):
                for j, val in enumerate(row):
                    if not bool(gap_filter[i, j]):
                        flat.append(float(val))
    else:  # head
        for layer_scores in node_mask.scores.values():
            for head_scores in layer_scores.values():
                for i, row in enumerate(head_scores):
                    for j, val in enumerate(row):
                        if not bool(gap_filter[i, j]):
                            flat.append(float(val))
    flat.sort()
    if not flat:
        return 0.0
    if target_sparsity <= 0:
        return min(flat) - 1e-9  # keep all
    if target_sparsity >= 1:
        return max(flat) + 1e-9  # drop all
    idx = int(target_sparsity * len(flat))
    idx = max(0, min(len(flat) - 1, idx))
    return flat[idx]


def _kl_on_answer(
    model,
    full_input: torch.Tensor,
    answer_pos: int,
    answer_token_ids: torch.Tensor,
    clean_lp_renorm: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(full_input).logits
    row = logits[0, answer_pos].float()
    masked_lp = F.log_softmax(row[answer_token_ids], dim=-1)
    p_clean = clean_lp_renorm.exp()
    kl = (p_clean * (clean_lp_renorm - masked_lp)).sum().item()
    return {"kl": kl, "p_masked": masked_lp.exp().cpu().tolist()}


def main(
    *,
    masks: List[str],
    model_name: str,
    data_path: Optional[str] = None,
    prompt_index: Optional[int] = None,
    prompt: Optional[str] = None,
    base_answer_type: str = "stored",
    analysis_timestep: Optional[int] = None,
    analysis_sentence_step: Optional[int] = None,
    probe_suffix: str = DEFAULT_SUFFIX,
    answer_letters: Optional[List[str]] = None,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    min_sentence_length: int = 10,
    sparsities: Optional[List[float]] = None,
    seed: int = 42,
    device: str = "cuda",
    output_path: Optional[str] = None,
):
    if answer_letters is None:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)
    if sparsities is None:
        sparsities = [0.0, 0.3, 0.5, 0.7, 0.9]

    set_seed(seed)

    print("=" * 80)
    print("Step 1: Build prefix + probe")
    print("=" * 80)
    tok = AutoTokenizer.from_pretrained(model_name)
    prefix_ids, sentences, prompt, correct_answer, _, _ = _build_prefix(
        tokenizer=tok,
        prompt=prompt,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type=base_answer_type,
        analysis_timestep=analysis_timestep,
        analysis_sentence_step=analysis_sentence_step,
        min_sentence_length=min_sentence_length,
        sentence_chunk=sentence_chunk,
    )
    prefix_len = prefix_ids.shape[-1]
    num_sents = len(sentences)
    probe = build_answer_probe(tok, suffix=probe_suffix, answer_letters=answer_letters)
    print(f"  prefix_len={prefix_len}, sentences={num_sents}")

    print("=" * 80)
    print("Step 2: Load model")
    print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)
    cont = probe.make_continuation(target_device)
    full_input = torch.cat([input_ids, cont], dim=-1)
    answer_pos = probe.answer_logit_position(prefix_len)
    ans_ids = probe.answer_token_ids.to(target_device)

    # Clean baseline (no hooks)
    print("=" * 80)
    print("Step 3: Clean baseline P(answer)")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_lp = F.log_softmax(
        clean_logits[0, answer_pos, ans_ids].float(), dim=-1,
    ).detach()
    clean_p = clean_lp.exp().cpu().tolist()
    del clean_logits
    torch.cuda.empty_cache()
    print(f"  clean P: {dict(zip(probe.answer_letters, clean_p))}")

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=target_device)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix", device=target_device)
    causal_filter = build_causal_filter(num_sents, device=target_device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
    combined_filter_cpu = combined_filter.cpu()
    token_to_sent = build_token_to_sent_map(
        sentences, full_input.shape[-1], target_device,
    )

    print("=" * 80)
    print("Step 4: Evaluate each mask at each sparsity")
    print("=" * 80)
    rows = []
    for mask_path in masks:
        node_mask = NodeMask.from_json(mask_path)
        algo = node_mask.algorithm
        print(f"\n  mask: {mask_path}")
        print(f"    algorithm: {algo}, granularity: {node_mask.granularity}")
        if set(node_mask.layers) != set(layers):
            missing = sorted(set(layers) - set(node_mask.layers))
            extra = sorted(set(node_mask.layers) - set(layers))
            raise ValueError(
                f"Mask {mask_path!r} was trained on layers "
                f"{sorted(node_mask.layers)} but eval applies hooks to "
                f"{layers}. Eval always installs hooks on all model layers, "
                f"so a subset-of-layers mask would be evaluated against an "
                f"untouched model on the non-target layers — silently "
                f"different from the training-time non-target ablation. "
                f"Train with layers_to_analyse='all' or extend this eval to "
                f"replicate the training non-target treatment. "
                f"Missing from mask: {missing}; extra in mask: {extra}."
            )
        for sp in sparsities:
            thresh = _threshold_for_sparsity(node_mask, sp, combined_filter_cpu)
            binary_masks = build_binary_masks(
                node_mask, thresh, layers, num_heads, num_sents,
                combined_filter, target_device,
            )
            handles = install_mask_hooks(
                model, layers, binary_masks, token_to_sent,
                combined_filter, renormalize=True,
            )
            try:
                out = _kl_on_answer(model, full_input, answer_pos, ans_ids, clean_lp)
            finally:
                remove_handles(handles)
                del binary_masks
                torch.cuda.empty_cache()

            actual_sp = node_mask.sparsity(thresh, gap_filter=combined_filter_cpu)
            row = {
                "mask_path": mask_path,
                "algorithm": algo,
                "target_sparsity": sp,
                "actual_sparsity": actual_sp,
                "threshold": thresh,
                "probe_kl": out["kl"],
                "p_masked": out["p_masked"],
            }
            rows.append(row)
            p_str = " ".join(f"{l}={p:.3f}" for l, p in zip(probe.answer_letters, out["p_masked"]))
            print(f"    sparsity_target={sp:.2f} actual={actual_sp:.3f} | KL={out['kl']:.4f} | {p_str}")

    print("=" * 80)
    print("Summary table")
    print("=" * 80)
    # Pretty print
    by_mask = {}
    for r in rows:
        by_mask.setdefault(r["mask_path"], []).append(r)
    print(f"{'mask':<70} | {'sparsity':>8} | {'KL':>7}")
    for mp, xs in by_mask.items():
        for r in xs:
            short = os.path.basename(mp)
            print(f"{short:<70} | {r['actual_sparsity']:>8.3f} | {r['probe_kl']:>7.4f}")

    out_payload = {
        "model_name": model_name,
        "prompt_index": prompt_index,
        "data_path": data_path,
        "prefix_len": prefix_len,
        "num_sentences": num_sents,
        "answer_letters": probe.answer_letters,
        "answer_token_ids": probe.answer_token_ids.tolist(),
        "answer_probs_clean": clean_p,
        "rows": rows,
    }
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nSaved comparison to {output_path}")

    del model
    clear_cuda()
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", nargs="+", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--base_answer_type", default="stored")
    parser.add_argument("--analysis_timestep", type=int, default=None)
    parser.add_argument("--analysis_sentence_step", type=int, default=None)
    parser.add_argument("--probe_suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--answer_letters", default=None,
        help="Comma-separated, e.g. ' A, B, C, D'.")
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument("--sparsities", nargs="+", type=float,
        default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    answer_letters = None
    if args.answer_letters is not None:
        answer_letters = [(" " + p.strip()) for p in args.answer_letters.split(",") if p.strip()]
    main(
        masks=args.masks,
        model_name=args.model_name,
        data_path=args.data_path,
        prompt_index=args.prompt_index,
        prompt=args.prompt,
        base_answer_type=args.base_answer_type,
        analysis_timestep=args.analysis_timestep,
        analysis_sentence_step=args.analysis_sentence_step,
        probe_suffix=args.probe_suffix,
        answer_letters=answer_letters,
        sentence_gap=args.sentence_gap,
        sentence_chunk=args.sentence_chunk,
        sparsities=args.sparsities,
        seed=args.seed,
        device=args.device,
        output_path=args.output_path,
    )
