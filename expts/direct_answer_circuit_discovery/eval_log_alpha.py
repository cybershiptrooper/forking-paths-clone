"""Evaluate SNP masks saved as ``log_alpha`` (or HC mean) under three
threshold modes plus an all-zero baseline that gives the per-prompt
``KL_max(p)`` ceiling.

Threshold modes:
- ``m_gt_0``        : keep edge iff ``log_alpha > β·log(−γ/ζ) ≈ −1.60``
                     (equivalent to "HC clamp didn't drive m to 0").
- ``m_gt_0.5``      : keep edge iff ``log_alpha > 0`` (canonical Cao midpoint).
- ``top_k@<sparsity>``: keep top-(1−sparsity)·N_valid edges by score, where
                       N_valid counts only entries that pass the
                       gap+mode+causal filter.

Outputs a JSON with one row per (mode, sparsity-target) plus the
``kl_max`` row and a ``clean`` row, suitable for the dataset-wide
paired comparison against ``thought_anchors``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

import torch

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import (
    install_mask_hooks,
    install_sdpa_mask_hooks,
    install_clean_sdpa_forward,
    remove_handles,
)
from utils.utils import set_seed, clear_cuda, get_attention_module
from expts.direct_answer_circuit_discovery.probe import (
    answer_probs_from_logits, build_answer_probe, DEFAULT_SUFFIX,
    DEFAULT_ANSWER_LETTERS,
)
from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)


_HC_BETA = 2.0 / 3.0
_HC_GAMMA = -0.1
_HC_ZETA = 1.1
M_GT_0_LOG_ALPHA_THRESHOLD = _HC_BETA * math.log(-_HC_GAMMA / _HC_ZETA)  # ≈ -1.6
M_GT_HALF_LOG_ALPHA_THRESHOLD = 0.0


def _scores_to_log_alpha(
    scores, score_readout: str, hc_beta: float = _HC_BETA,
) -> torch.Tensor:
    """Return scores in a form suitable for top-K ranking and HC threshold modes.

    Three score formats supported:

    - ``"log_alpha"``: SNP-saved raw log_alpha values. Returned unchanged.
    - ``"hard_concrete_mean"``: SNP-saved HC mean ``m ∈ [0,1]``. Inverted
      back to log_alpha via the deterministic HC readout
      ``m = clamp(σ(la/β)·(ζ−γ) + γ, 0, 1)``. Edges with ``m == 0`` get
      ``la = -1e6`` (HC clamp killed them — never alive at any threshold).
    - ``"raw_score"``: per-pair importance score (e.g. raw KL from
      thought_anchors / suppress / activation patching). Returned
      unchanged for top-K ranking. The HC threshold modes (``m>0``,
      ``m>0.5``) are not meaningful for raw scores and will fall back
      to a degenerate cutoff (see ``_build_binary_mask``).

    Auto-detection. If ``score_readout`` is missing/None and the scores
    fall outside [0, 1], we treat them as ``raw_score`` so that legacy
    TA / suppress masks (which save raw KL with ``score_readout=None``)
    rank correctly under top-K. Without this, cells with KL>1 all clamp
    to the same la value, breaking the top-K ranking among
    high-importance cells.
    """
    arr = torch.tensor(scores, dtype=torch.float32)  # may be (S,S) or nested
    if arr.dim() == 0:
        arr = arr.unsqueeze(0)
    if score_readout == "log_alpha":
        return arr
    if score_readout == "raw_score":
        return arr
    # Auto-detect: if scores are clearly not HC means (out of [0, 1]),
    # treat as raw_score for top-K.
    if score_readout in (None, "None"):
        if float(arr.min().item()) < 0.0 or float(arr.max().item()) > 1.0:
            return arr
    # HC mean → log_alpha. Inversion uses the training-time β from mask
    # metadata (hard_concrete_beta) so hc_beta_anneal runs invert correctly.
    m = arr.clamp(0.0, 1.0)
    s = (m - _HC_GAMMA) / (_HC_ZETA - _HC_GAMMA)
    s = s.clamp(1e-6, 1 - 1e-6)
    la = hc_beta * (torch.log(s) - torch.log1p(-s))
    # Edges with m == 0 had log_alpha ≤ β·log(−γ/ζ); mark very negative so
    # they are never picked for top-K and always pruned by m>0 / m>0.5.
    la = torch.where(m == 0.0, torch.full_like(la, -1e6), la)
    # Edges saturated at exactly m == 1.0 all invert to the same clamped
    # log_alpha, so ranking among them is arbitrary — warn when top-K would
    # have to break such ties (finding 4 in
    # notes/reports_diagnostics/snp_review_findings.md). Retrain with
    # save_log_alpha: true to preserve the ranking.
    n_saturated = int((m == 1.0).sum().item())
    if n_saturated > 0:
        print(
            f"  WARNING: {n_saturated}/{m.numel()} hard_concrete_mean scores "
            f"are saturated at exactly 1.0; top-K ranking among them is "
            f"arbitrary (legacy readout — retrain with save_log_alpha: true)."
        )
    return la


def _build_binary_mask(
    log_alpha: torch.Tensor,
    mode: str,
    target_sparsity: Optional[float],
    valid_filter: torch.Tensor,
    m_gt_0_threshold: float = M_GT_0_LOG_ALPHA_THRESHOLD,
) -> torch.Tensor:
    """Return a binary mask in {0, 1} with the same shape as *log_alpha*.

    Accepts (S, S) for pair granularity or (L, H, S, S) for
    layer/head granularity.

    valid_filter is (S, S) with 0 on filtered-out cells. For multi-dim
    log_alpha it is broadcast over the leading (L, H) dimensions. We
    never prune filtered-out cells (they get value 1) — they are
    handled separately at hook time.
    """
    if mode == "m_gt_0":
        keep = (log_alpha > m_gt_0_threshold).float()
    elif mode == "m_gt_0.5":
        keep = (log_alpha > M_GT_HALF_LOG_ALPHA_THRESHOLD).float()
    elif mode == "top_k":
        assert target_sparsity is not None
        valid = valid_filter.bool()
        if log_alpha.dim() > 2:
            valid = valid.unsqueeze(0).unsqueeze(0).expand_as(log_alpha)
        n_valid = int(valid.sum().item())
        n_keep = max(0, int(round((1.0 - target_sparsity) * n_valid)))
        flat_la = log_alpha.flatten()
        flat_valid = valid.flatten()
        keep_flat = torch.zeros_like(flat_la)
        if n_keep > 0:
            scores_for_rank = torch.where(
                flat_valid, flat_la, torch.full_like(flat_la, -math.inf),
            )
            _, top_idx = torch.topk(scores_for_rank, n_keep)
            keep_flat[top_idx] = 1.0
        keep = keep_flat.view_as(log_alpha)
    else:
        raise ValueError(f"Unknown threshold mode: {mode!r}")
    return keep


def _all_zero_mask(num_sents, num_layers, num_heads, device, per_layer=False):
    """All-zero mask. If *per_layer*, returns (L, H, S, S); else (S, S)."""
    if per_layer:
        return torch.zeros(num_layers, num_heads, num_sents, num_sents, device=device)
    return torch.zeros(num_sents, num_sents, device=device)


def _binary_to_per_layer_masks(binary, layers, num_heads):
    """Convert binary mask to ``{layer: (H, S, S)}`` dict for hooks.

    *binary* can be:
    - (S, S): broadcast to all layers and heads.
    - (L, H, S, S): slice per layer.
    """
    if binary.dim() == 2:
        expanded = binary.unsqueeze(0).expand(num_heads, -1, -1).contiguous()
        return {l: expanded for l in layers}
    return {l: binary[i].contiguous() for i, l in enumerate(layers)}


def _evaluate_mask(
    model, layers, num_heads, full_input, prefix_len, probe,
    binary, token_to_sent, combined_filter, device,
    renormalize_masked_attention, backend,
):
    """Install per-layer mask hooks and run one forward pass.

    *binary* is either (S, S) — broadcast to all layers/heads — or
    (L, H, S, S) for layer/head granularity.
    """
    binary_masks = _binary_to_per_layer_masks(binary, layers, num_heads)
    if backend == "sdpa":
        # Forward is already SDPA-patched. Just attach mask state and run.
        for l in layers:
            attn = get_attention_module(model, l)
            attn._circuit_mask = binary_masks[l]
            attn._token_to_sent = token_to_sent
            attn._gap_filter = combined_filter
            attn._renormalize_masked_attn = renormalize_masked_attention
        try:
            with torch.no_grad():
                logits = model(full_input).logits
            masked_p = answer_probs_from_logits(logits, probe, prefix_len).cpu()
        finally:
            # Clear so the next clean / masked forward starts fresh.
            for l in layers:
                attn = get_attention_module(model, l)
                for attr in ("_circuit_mask", "_token_to_sent", "_gap_filter"):
                    if hasattr(attn, attr):
                        delattr(attn, attr)
        del logits
        clear_cuda()
        return masked_p

    # Eager backend (legacy path).
    handles = install_mask_hooks(
        model, layers, binary_masks, token_to_sent, combined_filter,
        renormalize=renormalize_masked_attention,
    )
    try:
        with torch.no_grad():
            logits = model(full_input).logits
        masked_p = answer_probs_from_logits(logits, probe, prefix_len).cpu()
    finally:
        for h in handles:
            h.remove()
    del logits
    clear_cuda()
    return masked_p


def _kl(p_clean: torch.Tensor, p_masked: torch.Tensor) -> float:
    eps = 1e-12
    pc = p_clean.clamp_min(eps)
    pm = p_masked.clamp_min(eps)
    return float((pc * (pc.log() - pm.log())).sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_index", type=int, required=True)
    parser.add_argument("--analysis_sentence_step", type=int, required=True)
    parser.add_argument("--sentences_after_prefix", type=int, default=0)
    parser.add_argument("--sentence_gap", type=int, default=0)
    parser.add_argument("--mask_mode", default="prefix")
    parser.add_argument(
        "--top_k_sparsities", type=str, default="0.3,0.5,0.7,0.9",
        help="Comma-separated target sparsities for top-K mode.",
    )
    parser.add_argument(
        "--no_renormalize_masked_attention",
        dest="renormalize_masked_attention",
        action="store_false",
    )
    parser.add_argument(
        "--answer_letters", type=str, default=None,
        help="Comma-separated probe letters override (e.g. ' A, B, C, D, E'). "
        "Needed for masks without probe metadata (thought anchors) on "
        "datasets whose choice count differs from the 4-letter default.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda",
        help="Device map for model loading. Use 'auto' for multi-GPU.",
    )
    parser.add_argument(
        "--backend", choices=["eager", "sdpa"], default="sdpa",
        help="Attention backend used for BOTH the clean reference forward "
        "and the masked forward. SDPA matches the backend SNP and the "
        "attribution algorithms train under; eager matches the old "
        "thought_anchors / suppress score-generation backend.",
    )
    parser.add_argument(
        "--force_freeze_prompt", action="store_true",
        help="Freeze all prompt-sentence edges at 1.0 during evaluation, "
        "even if the mask was not trained with freeze_prompt_sentences. "
        "Uses the prompt boundary derived from the data.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    # ----- Load mask -----
    nm = NodeMask.from_json(args.mask_path)

    # Masks trained with a candidate-set objective (open-ended answers)
    # record an answer_bank_path; they are evaluated on the candidate
    # bank instead of the letter probe. Legacy masks are unaffected.
    if nm.metadata.get("answer_bank_path"):
        from expts.direct_answer_circuit_discovery.eval_candidate_bank import (
            evaluate_candidate_mask,
        )
        evaluate_candidate_mask(args, nm)
        return

    score_readout = nm.metadata.get("score_readout", "hard_concrete_mean")
    granularity = nm.metadata.get("mask_granularity") or nm.granularity or "pair"
    # Training-time Hard-Concrete temperature (differs from 2/3 for
    # hc_beta_anneal runs); drives both the HC-mean inversion and the
    # m>0 threshold so eval matches the trained distribution.
    hc_beta = float(nm.metadata.get("hard_concrete_beta") or _HC_BETA)
    m_gt_0_threshold = hc_beta * math.log(-_HC_GAMMA / _HC_ZETA)

    # ----- Build prefix + probe -----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # sentences_to_be_masked has only the first analysis_sentence_step
    # sentences; the k extra context sentences are in prefix_ids but
    # unmapped (mask = 1.0 via sentinel padding).
    (
        prefix_ids,
        sentences_to_be_masked,
        prompt,
        correct_answer,
        _,
        num_prompt_sentences,
    ) = _build_prefix(
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
    prefix_len = prefix_ids.shape[-1]

    # Pull probe spec from mask metadata where available so the eval matches
    # the training-time probe exactly. Falls back to defaults for legacy masks.
    suffix = nm.metadata.get("probe_suffix", DEFAULT_SUFFIX)
    if args.answer_letters:
        answer_letters = [s for s in args.answer_letters.split(",")]
    else:
        answer_letters = nm.metadata.get("answer_letters") or list(DEFAULT_ANSWER_LETTERS)
    probe = build_answer_probe(tokenizer, suffix=suffix, answer_letters=answer_letters)
    target_letter = nm.metadata.get("target_letter") or correct_answer
    target_answer_id = None
    if target_letter is not None:
        stripped = [l.strip() for l in probe.answer_letters]
        if target_letter.strip() in stripped:
            target_answer_id = stripped.index(target_letter.strip())

    # ----- Load model -----
    device = getattr(args, "device", "cuda") or "cuda"
    model, _ = load_model_eager(args.model_name, device=device)
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)
    continuation = probe.make_continuation(target_device)
    full_input = torch.cat([input_ids, continuation], dim=-1)
    full_len = full_input.shape[-1]

    # If --backend sdpa, monkey-patch every attention layer to the SDPA
    # forward *before* the clean forward, so clean and masked logits
    # both go through SDPA. The SDPA forward is a no-op (returns None
    # for the additive mask) when ``_circuit_mask`` is not set on the
    # module — so the unmasked clean forward is unaffected.
    sdpa_clean_handles = None
    if args.backend == "sdpa":
        sdpa_clean_handles = install_clean_sdpa_forward(model)
        print(f"  Backend: sdpa (patched {len(sdpa_clean_handles)} layers)")
    else:
        print(f"  Backend: eager")

    # ----- Filters -----
    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=target_device)
    mode_filter = build_mode_filter(
        num_sents, num_sents, args.mask_mode, device=target_device,
    )
    causal_filter = build_causal_filter(num_sents, device=target_device)
    # Masks trained with freeze_prompt_sentences record the number of
    # frozen prompt sentences in metadata. The same entries must be
    # frozen at eval so the top-K pool matches the learnable pool.
    num_frozen_prompt = int(nm.metadata.get("num_frozen_prompt_sentences", 0) or 0)
    if args.force_freeze_prompt and not num_frozen_prompt:
        num_frozen_prompt = num_prompt_sentences
        print(f"  --force_freeze_prompt: freezing {num_frozen_prompt} prompt sentences")
    if num_frozen_prompt:
        if num_frozen_prompt != num_prompt_sentences:
            raise ValueError(
                f"Mask metadata says {num_frozen_prompt} frozen prompt "
                f"sentences but _build_prefix found {num_prompt_sentences}. "
                "Sentence splitting is inconsistent between training and eval."
            )
        print(f"  Frozen prompt sentences: {num_frozen_prompt}")
    prompt_filter = (
        build_prompt_filter(num_frozen_prompt, num_sents, device=target_device)
        if num_frozen_prompt
        else None
    )
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter
    )

    # token_to_sent
    token_to_sent = torch.full((full_len,), -1, dtype=torch.long, device=target_device)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx

    # Respect a layer-subset mask instead of silently masking every layer
    # (finding 5 in notes/reports_diagnostics/snp_review_findings.md).
    # Pair masks are applied only to their trained layers; layer/head
    # masks trained on a subset are rejected loudly (the score-loading
    # loop below indexes scores by position, which only matches the
    # all-layers layout).
    all_layers = list(range(model.config.num_hidden_layers))
    mask_layers = sorted(int(l) for l in (nm.layers or all_layers))
    if granularity == "pair":
        layers = mask_layers
        if layers != all_layers:
            print(f"  Layer-subset pair mask: hooks on {len(layers)} layers only")
    else:
        if mask_layers != all_layers:
            raise ValueError(
                f"{granularity}-granularity mask trained on a layer subset "
                f"({len(mask_layers)}/{len(all_layers)} layers) is not "
                "supported by this evaluator."
            )
        layers = all_layers
    num_heads = model.config.num_attention_heads
    num_layers = len(layers)

    # ----- Build log_alpha tensor -----
    per_layer = granularity in ("layer", "head")
    if granularity == "pair":
        log_alpha = _scores_to_log_alpha(nm.scores, score_readout, hc_beta=hc_beta)
        if log_alpha.shape != (num_sents, num_sents):
            raise ValueError(
                f"mask shape {tuple(log_alpha.shape)} != ({num_sents}, {num_sents})"
            )
    elif granularity == "layer":
        layer_tensors = []
        for l in range(num_layers):
            la_2d = _scores_to_log_alpha(nm.scores[l], score_readout, hc_beta=hc_beta)
            layer_tensors.append(la_2d.unsqueeze(0).expand(num_heads, -1, -1))
        log_alpha = torch.stack(layer_tensors)  # (L, H, S, S)
    elif granularity == "head":
        layer_tensors = []
        for l in range(num_layers):
            head_tensors = []
            for h in range(num_heads):
                la_2d = _scores_to_log_alpha(nm.scores[l][h], score_readout, hc_beta=hc_beta)
                head_tensors.append(la_2d)
            layer_tensors.append(torch.stack(head_tensors))  # (H, S, S)
        log_alpha = torch.stack(layer_tensors)  # (L, H, S, S)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")
    print(f"  Granularity: {granularity}, log_alpha shape: {tuple(log_alpha.shape)}")

    # ----- Clean P(answer) -----
    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
    del clean_logits
    clear_cuda()

    log_alpha_dev = log_alpha.to(target_device)
    # combined_filter == True means *frozen at 1.0* (gap/mode/causal exclude).
    # Learnable / valid edges are the complement.
    valid_filter = ~combined_filter.bool()

    rows = [
        {
            "row": "clean",
            "mode": "clean",
            "sparsity": 0.0,
            "kl": 0.0,
            "answer_probs": clean_p.tolist(),
        }
    ]

    # ----- KL_max via all-zero mask -----
    p_zero = _evaluate_mask(
        model, layers, num_heads, full_input, prefix_len, probe,
        _all_zero_mask(num_sents, num_layers, num_heads, target_device, per_layer=per_layer),
        token_to_sent, combined_filter, target_device,
        args.renormalize_masked_attention, args.backend,
    )
    kl_max = _kl(clean_p, p_zero)
    rows.append({
        "row": "kl_max",
        "mode": "all_zero",
        "sparsity": 1.0,
        "kl": kl_max,
        "answer_probs": p_zero.tolist(),
    })

    # ----- Three threshold modes -----
    sparsities = [float(s) for s in args.top_k_sparsities.split(",") if s.strip()]
    eval_specs = []
    eval_specs.append(("m_gt_0", None))
    eval_specs.append(("m_gt_0.5", None))
    for s in sparsities:
        eval_specs.append(("top_k", s))

    for mode, sp in eval_specs:
        binary = _build_binary_mask(
            log_alpha_dev, mode, sp, valid_filter,
            m_gt_0_threshold=m_gt_0_threshold,
        )
        with torch.no_grad():
            valid = valid_filter.bool()
            if binary.dim() > 2:
                valid_expanded = valid.unsqueeze(0).unsqueeze(0).expand_as(binary)
            else:
                valid_expanded = valid
            n_valid = int(valid_expanded.sum().item())
            n_pruned = int(((binary == 0) & valid_expanded).sum().item())
            sp_actual = n_pruned / n_valid if n_valid > 0 else 0.0
        p_masked = _evaluate_mask(
            model, layers, num_heads, full_input, prefix_len, probe,
            binary, token_to_sent, combined_filter, target_device,
            args.renormalize_masked_attention, args.backend,
        )
        kl = _kl(clean_p, p_masked)
        kl_norm = kl / kl_max if kl_max > 0 else float("nan")
        p_target = (
            float(p_masked[target_answer_id].item())
            if target_answer_id is not None else None
        )
        rows.append({
            "row": "mask_eval",
            "mode": mode,
            "target_sparsity": sp,
            "sparsity": sp_actual,
            "kl": kl,
            "kl_normalized": kl_norm,
            "answer_probs": p_masked.tolist(),
            "p_target": p_target,
        })

    # Cleanup SDPA-patched layers if we installed them.
    if sdpa_clean_handles is not None:
        remove_handles(sdpa_clean_handles)

    out = {
        "mask_path": args.mask_path,
        "model_name": args.model_name,
        "data_path": args.data_path,
        "prompt_index": args.prompt_index,
        "analysis_sentence_step": args.analysis_sentence_step,
        "sentences_after_prefix": args.sentences_after_prefix,
        "sentence_gap": args.sentence_gap,
        "mask_mode": args.mask_mode,
        "num_frozen_prompt_sentences": num_frozen_prompt,
        "mask_granularity": granularity,
        "score_readout_input": score_readout,
        "target_letter": target_letter,
        "answer_letters": probe.answer_letters,
        "clean_answer_probs": clean_p.tolist(),
        "kl_max": kl_max,
        "backend": args.backend,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved evaluation to {args.output}")
    print(f"  kl_max = {kl_max:.4f}")
    for r in rows:
        if r["row"] == "mask_eval":
            print(
                f"  {r['mode']:>10s} target_sp={r.get('target_sparsity')} "
                f"sp={r['sparsity']:.3f} kl={r['kl']:.4f} "
                f"kl_norm={r['kl_normalized']:.4f}"
            )


if __name__ == "__main__":
    main()
