"""Evaluate masks trained with candidate-set objectives (open-ended answers).

Dispatched from ``eval_log_alpha.py`` when the mask metadata records an
``answer_bank_path`` — i.e. the mask was trained with one of the
``candidate_*`` objectives on an open-ended dataset (MATH).

For each threshold mode the mask is installed and every candidate
continuation (probe suffix + candidate answer tokens, from the frozen
answer bank) is teacher-forced; the resulting per-candidate sequence
log-probabilities give:

- ``p_gold``           — softmax-normalized probability of the gold cluster
                         over the candidate set (the fixed-corpus P(gold)
                         readout);
- ``reward_gap``       — p_gold − max wrong-cluster probability;
- ``logprob_margin``   — logsumexp(gold members) − max wrong-cluster
                         logsumexp (no softmax);
- ``snis_reward_gap``  — the self-normalized importance-sampling variant,
                         weights ∝ count · exp(logprob_masked − logprob_clean),
                         sampled candidates only;
- ``candidate_kl``     — KL(clean ‖ masked) between the two candidate
                         distributions.

All are computed from the same (N,) log-probability vector, so the eval
JSON supports matched-target evaluation for any of the three training
objectives plus cross-objective diagnostics. Output schema mirrors
``eval_log_alpha.py``: a ``rows`` list with ``clean``, ``kl_max`` (all
learnable edges ablated; ceiling for ``candidate_kl`` normalization), and
one ``mask_eval`` row per (mode, sparsity).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

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
    install_clean_sdpa_forward,
    remove_handles,
)
from utils.utils import set_seed, clear_cuda, get_attention_module


def _candidate_logprobs(model, prefix_ids, continuations, device):
    """Teacher-forced sequence log-prob of each continuation after the prefix.

    Hooks (if any) must already be installed on the model. Returns (N,)
    tensor on CPU.
    """
    lps = []
    prefix_len = prefix_ids.shape[-1]
    with torch.no_grad():
        for cont in continuations:
            full = torch.cat([prefix_ids, cont], dim=-1)
            logits = model(full).logits.float()
            rows = logits[0, prefix_len - 1 : -1, :]
            logprobs = F.log_softmax(rows, dim=-1)
            targets = full[0, prefix_len:]
            lps.append(logprobs[torch.arange(len(targets)), targets].sum().cpu())
            del logits
    clear_cuda()
    return torch.stack(lps)


def _masked_candidate_logprobs(
    model, layers, binary_masks, token_to_sent, combined_filter,
    renormalize, backend, prefix_ids, continuations, device,
):
    """Install mask state, teacher-force all candidates, clean up."""
    if backend == "sdpa":
        for l in layers:
            attn = get_attention_module(model, l)
            attn._circuit_mask = binary_masks[l]
            attn._token_to_sent = token_to_sent
            attn._gap_filter = combined_filter
            attn._renormalize_masked_attn = renormalize
        try:
            return _candidate_logprobs(model, prefix_ids, continuations, device)
        finally:
            for l in layers:
                attn = get_attention_module(model, l)
                for attr in ("_circuit_mask", "_token_to_sent", "_gap_filter"):
                    if hasattr(attn, attr):
                        delattr(attn, attr)
    handles = install_mask_hooks(
        model, layers, binary_masks, token_to_sent, combined_filter,
        renormalize=renormalize,
    )
    try:
        return _candidate_logprobs(model, prefix_ids, continuations, device)
    finally:
        for h in handles:
            h.remove()


def _cluster_probs(logprobs, answer_ids, num_clusters):
    """Softmax over candidates, summed into clusters. Returns (C,) tensor."""
    p = torch.softmax(logprobs, dim=0)
    return torch.zeros(num_clusters).index_add(0, answer_ids, p)


def _metrics_from_logprobs(
    lp_masked, lp_clean, answer_ids, num_clusters, target_cluster, counts,
):
    """All candidate-set metrics from one (N,) masked log-prob vector."""
    p_cluster = _cluster_probs(lp_masked, answer_ids, num_clusters)
    other = torch.ones(num_clusters, dtype=torch.bool)
    other[target_cluster] = False

    p_gold = float(p_cluster[target_cluster].item())
    p_best_other = float(p_cluster[other].max().item()) if other.any() else 0.0
    reward_gap = p_gold - p_best_other

    cluster_lse = torch.stack([
        torch.logsumexp(lp_masked[answer_ids == a], dim=0)
        if (answer_ids == a).any()
        else torch.tensor(float("-inf"))
        for a in range(num_clusters)
    ])
    margin = float(
        (cluster_lse[target_cluster]
         - (cluster_lse[other].max() if other.any() else 0.0)).item()
    )

    valid = counts > 0
    snis_p_gold = None
    snis_reward_gap = None
    if valid.any():
        log_w = lp_masked[valid] - lp_clean[valid] + counts[valid].log()
        w = torch.softmax(log_w, dim=0)
        p_snis = torch.zeros(num_clusters).index_add(0, answer_ids[valid], w)
        snis_p_gold = float(p_snis[target_cluster].item())
        snis_best_other = float(p_snis[other].max().item()) if other.any() else 0.0
        snis_reward_gap = snis_p_gold - snis_best_other

    p_cluster_clean = _cluster_probs(lp_clean, answer_ids, num_clusters)
    eps = 1e-12
    pc = p_cluster_clean.clamp_min(eps)
    pm = p_cluster.clamp_min(eps)
    candidate_kl = float((pc * (pc.log() - pm.log())).sum().item())

    return {
        "candidate_logprobs": lp_masked.tolist(),
        "cluster_probs": p_cluster.tolist(),
        "p_gold": p_gold,
        "reward_gap": reward_gap,
        "logprob_margin": margin,
        "snis_p_gold": snis_p_gold,
        "snis_reward_gap": snis_reward_gap,
        "candidate_kl": candidate_kl,
    }


def evaluate_candidate_mask(args, nm: NodeMask):
    """Candidate-bank evaluation entry point (called from eval_log_alpha)."""
    # Import here to avoid a circular import at module load.
    from expts.direct_answer_circuit_discovery.eval_log_alpha import (
        _scores_to_log_alpha,
        _build_binary_mask,
        _all_zero_mask,
        _binary_to_per_layer_masks,
    )
    from expts.direct_answer_circuit_discovery.learn import (
        _build_prefix, load_model_eager,
    )
    from transformers import AutoTokenizer

    set_seed(args.seed)
    bank_path = nm.metadata["answer_bank_path"]
    with open(bank_path) as f:
        bank = json.load(f)
    if (
        bank["data_path"] != args.data_path
        or bank["prompt_index"] != args.prompt_index
    ):
        raise ValueError(
            f"Bank {bank_path} is for ({bank['data_path']}, "
            f"{bank['prompt_index']}), eval got ({args.data_path}, "
            f"{args.prompt_index})."
        )

    from expts.direct_answer_circuit_discovery.answer_bank_utils import (
        flatten_bank_candidates,
    )
    candidates = bank["candidates"]
    _token_lists, _cluster_ids, _counts = flatten_bank_candidates(bank)
    answer_ids = torch.tensor(_cluster_ids)
    counts = torch.tensor(_counts, dtype=torch.float)
    num_clusters = int(bank["num_clusters"])
    target_cluster = int(bank["target_cluster"])
    training_objective = nm.metadata.get("objective")

    score_readout = nm.metadata.get("score_readout", "hard_concrete_mean")
    granularity = nm.metadata.get("mask_granularity") or nm.granularity or "pair"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    (
        prefix_ids, sentences, _prompt, _corr, _fmt, num_prompt_sentences,
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
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    prefix_dev = prefix_ids.to(device)
    continuations = [
        torch.tensor([ids], device=device) for ids in _token_lists
    ]
    max_cont = max(c.shape[-1] for c in continuations)
    full_len = prefix_len + max_cont

    sdpa_clean_handles = None
    if args.backend == "sdpa":
        sdpa_clean_handles = install_clean_sdpa_forward(model)
        print(f"  Backend: sdpa (patched {len(sdpa_clean_handles)} layers)")
    else:
        print("  Backend: eager")

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen_prompt = int(nm.metadata.get("num_frozen_prompt_sentences", 0) or 0)
    if getattr(args, "force_freeze_prompt", False) and not num_frozen_prompt:
        num_frozen_prompt = num_prompt_sentences
        print(f"  --force_freeze_prompt: freezing {num_frozen_prompt} prompt sentences")
    if num_frozen_prompt:
        if num_frozen_prompt < num_prompt_sentences:
            raise ValueError(
                f"Mask metadata says {num_frozen_prompt} frozen prompt "
                f"sentences but _build_prefix found {num_prompt_sentences} "
                "prompt sentences (metadata may exceed this for window-"
                "restricted masks, never undercut it)."
            )
        print(f"  Frozen prompt sentences: {num_frozen_prompt}")
    prompt_filter = (
        build_prompt_filter(num_frozen_prompt, num_sents, device=device)
        if num_frozen_prompt
        else None
    )
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter
    )

    token_to_sent = torch.full((full_len,), -1, dtype=torch.long, device=device)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx

    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    num_layers = len(layers)
    per_layer = granularity in ("layer", "head")

    if granularity == "pair":
        log_alpha = _scores_to_log_alpha(nm.scores, score_readout)
        if log_alpha.shape != (num_sents, num_sents):
            raise ValueError(
                f"mask shape {tuple(log_alpha.shape)} != ({num_sents}, {num_sents})"
            )
    elif granularity == "layer":
        log_alpha = torch.stack([
            _scores_to_log_alpha(nm.scores[l], score_readout)
            .unsqueeze(0).expand(num_heads, -1, -1)
            for l in range(num_layers)
        ])
    elif granularity == "head":
        log_alpha = torch.stack([
            torch.stack([
                _scores_to_log_alpha(nm.scores[l][h], score_readout)
                for h in range(num_heads)
            ])
            for l in range(num_layers)
        ])
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")
    print(f"  Granularity: {granularity}, log_alpha shape: {tuple(log_alpha.shape)}")
    print(f"  Candidates: {len(candidates)}, clusters: {num_clusters}, "
          f"gold cluster: {target_cluster}")

    # ----- Clean candidate log-probs (no mask) -----
    lp_clean = _candidate_logprobs(model, prefix_dev, continuations, device)
    clean_metrics = _metrics_from_logprobs(
        lp_clean, lp_clean, answer_ids, num_clusters, target_cluster, counts,
    )
    rows = [{
        "row": "clean", "mode": "clean", "sparsity": 0.0, **clean_metrics,
    }]

    log_alpha_dev = log_alpha.to(device)
    valid_filter = ~combined_filter.bool()

    def eval_binary(binary):
        binary_masks = _binary_to_per_layer_masks(binary, layers, num_heads)
        lp_m = _masked_candidate_logprobs(
            model, layers, binary_masks, token_to_sent, combined_filter,
            args.renormalize_masked_attention, args.backend,
            prefix_dev, continuations, device,
        )
        return _metrics_from_logprobs(
            lp_m, lp_clean, answer_ids, num_clusters, target_cluster, counts,
        )

    # ----- kl_max: all learnable edges ablated -----
    zero_metrics = eval_binary(
        _all_zero_mask(num_sents, num_layers, num_heads, device, per_layer=per_layer)
    )
    kl_max = zero_metrics["candidate_kl"]
    rows.append({
        "row": "kl_max", "mode": "all_zero", "sparsity": 1.0, **zero_metrics,
    })

    # ----- Threshold modes -----
    sparsities = [float(s) for s in args.top_k_sparsities.split(",") if s.strip()]
    eval_specs = [("m_gt_0", None), ("m_gt_0.5", None)]
    eval_specs += [("top_k", s) for s in sparsities]

    for mode, sp in eval_specs:
        binary = _build_binary_mask(log_alpha_dev, mode, sp, valid_filter)
        with torch.no_grad():
            valid = valid_filter.bool()
            if binary.dim() > 2:
                valid_expanded = valid.unsqueeze(0).unsqueeze(0).expand_as(binary)
            else:
                valid_expanded = valid
            n_valid = int(valid_expanded.sum().item())
            n_pruned = int(((binary == 0) & valid_expanded).sum().item())
            sp_actual = n_pruned / n_valid if n_valid > 0 else 0.0
        metrics = eval_binary(binary)
        rows.append({
            "row": "mask_eval",
            "mode": mode,
            "target_sparsity": sp,
            "sparsity": sp_actual,
            "kl_normalized": (
                metrics["candidate_kl"] / kl_max if kl_max > 0 else float("nan")
            ),
            **metrics,
        })

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
        "eval_kind": "candidate_bank",
        "training_objective": training_objective,
        "answer_bank_path": bank_path,
        "gold_answer": bank.get("gold_answer_normalized"),
        "num_candidates": len(candidates),
        "num_clusters": num_clusters,
        "clean_fraction_correct": bank.get("clean_fraction_correct"),
        "kl_max": kl_max,
        "backend": args.backend,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved candidate-bank evaluation to {args.output}")
    print(f"  clean p_gold = {clean_metrics['p_gold']:.4f}, kl_max = {kl_max:.4f}")
    for r in rows:
        if r["row"] == "mask_eval":
            print(
                f"  {r['mode']:>10s} target_sp={r.get('target_sparsity')} "
                f"sp={r['sparsity']:.3f} p_gold={r['p_gold']:.4f} "
                f"reward_gap={r['reward_gap']:.4f} "
                f"kl_norm={r['kl_normalized']:.4f}"
            )
