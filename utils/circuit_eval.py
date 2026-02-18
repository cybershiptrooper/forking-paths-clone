"""Evaluate a learned circuit mask at different sparsity thresholds.

Provides helpers for:
- Building binary masks from a NodeMask at a given threshold
- Installing / removing attention ablation hooks
- Running the model with masked attention and computing objectives
- Orchestrating threshold sweeps with random-mask baselines
"""

import types
from typing import Callable

import torch

from utils.utils import Sentence, get_attention_module
from utils.masks import NodeMask, build_gap_filter, apply_gap_filter, build_mode_filter, build_combined_filter
from utils.cot_analysis import split_tokens_into_sentences
from utils.circuit_discovery.nodewise_attribution import (
    llama_attention_forward_with_differentiable_mask,
)
from utils.circuit_discovery.base import AblationHandle


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def build_token_to_sent_map(
    sentences: list[Sentence],
    total_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each token position to its sentence index (-1 if none)."""
    token_to_sent = torch.full((total_seq_len,), -1, dtype=torch.long)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx
    return token_to_sent.to(device)


def build_binary_masks(
    node_mask: NodeMask,
    threshold: float,
    layers: list[int],
    num_heads: int,
    num_sents: int,
    gap_filter: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Threshold *node_mask* scores into binary masks (1 = keep, 0 = ablate).

    Entries with ``score >= threshold`` are kept; the rest are zeroed.
    Gap-filtered positions are always kept (filled with 1).
    """
    binary_masks: dict[int, torch.Tensor] = {}
    for layer in layers:
        m = torch.ones(num_heads, num_sents, num_sents, device=device)
        for h in range(num_heads):
            scores = node_mask.scores[layer][h]
            for i in range(num_sents):
                for j in range(num_sents):
                    if scores[i][j] < threshold:
                        m[h, i, j] = 0.0
        binary_masks[layer] = apply_gap_filter(m, gap_filter, fill_value=1.0)
    return binary_masks


def build_random_masks(
    keep_prob: float,
    layers: list[int],
    num_heads: int,
    num_sents: int,
    gap_filter: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Build random binary masks that keep *keep_prob* fraction of edges."""
    random_masks: dict[int, torch.Tensor] = {}
    for layer in layers:
        rand = torch.rand(num_heads, num_sents, num_sents, device=device)
        random_masks[layer] = apply_gap_filter(
            (rand < keep_prob).float(), gap_filter, fill_value=1.0
        )
    return random_masks


def compute_clean_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Run the (unpatched) model on each continuation and return logits on CPU."""
    clean_logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            clean_logits.append(model(full_input).logits.cpu())
    return clean_logits


# ------------------------------------------------------------------
# Attention-hook management
# ------------------------------------------------------------------


def install_mask_hooks(
    model: torch.nn.Module,
    layers: list[int],
    binary_masks: dict[int, torch.Tensor],
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
) -> list[AblationHandle]:
    """Monkey-patch attention modules with *binary_masks* and return handles."""
    handles: list[AblationHandle] = []
    for layer_idx in layers:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = binary_masks[layer_idx]
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(
            llama_attention_forward_with_differentiable_mask, attn_module
        )
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def install_non_target_ablation(
    model: torch.nn.Module,
    target_layers: list[int],
    num_heads: int,
    num_sents: int,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    device: torch.device,
) -> list[AblationHandle]:
    """Zero-out attention in every layer *not* in *target_layers*."""
    num_total_layers = model.config.num_hidden_layers
    target_set = set(target_layers)
    non_target = [l for l in range(num_total_layers) if l not in target_set]
    print(f"Ablating {len(non_target)} non-target layers for evaluation...")

    zero_mask = torch.zeros(num_heads, num_sents, num_sents, device=device)
    filled_mask = apply_gap_filter(zero_mask, gap_filter, fill_value=1.0)

    handles: list[AblationHandle] = []
    for layer_idx in non_target:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = filled_mask
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(
            llama_attention_forward_with_differentiable_mask, attn_module
        )
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def remove_handles(handles: list[AblationHandle]):
    """Remove all ablation handles (restore original forwards)."""
    for h in handles:
        h.remove()


# ------------------------------------------------------------------
# Single masked evaluation pass
# ------------------------------------------------------------------


def eval_with_masks(
    model: torch.nn.Module,
    binary_masks: dict[int, torch.Tensor],
    layers: list[int],
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
    clean_logits_list: list[torch.Tensor],
    objective_fn: Callable,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    collect_per_token: bool = False,
    collect_per_sentence: bool = False,
    tokenizer=None,
    min_sentence_length: int = 10,
) -> tuple[float, list[list[float]], list[list[dict]]]:
    """Run model with *binary_masks* applied and compute objective.

    Returns ``(avg_objective, per_token_kl_branches, per_sent_kl_branches)``.
    """
    device = next(model.parameters()).device
    prefix_len = input_ids.shape[-1]

    handles = install_mask_hooks(
        model, layers, binary_masks, token_to_sent, gap_filter, renormalize
    )

    total_objective = 0.0
    total_branches = 0
    per_token_kl_branches: list[list[float]] = []
    per_sent_kl_branches: list[list[dict]] = []

    with torch.no_grad():
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            pos_mask = torch.zeros(1, full_len, device=device)
            pos_mask[0, prefix_len - 1 : full_len - 1] = 1.0

            logits = model(full_input).logits
            clean = clean_logits_list[cont_idx][:, :full_len].to(device)
            objective_value = objective_fn(clean, logits, pos_mask)
            total_objective += objective_value.item()
            total_branches += 1

            if collect_per_token or collect_per_sentence:
                log_clean = torch.nn.functional.log_softmax(clean.detach(), dim=-1)
                log_masked = torch.nn.functional.log_softmax(logits, dim=-1)
                kl_tokens = torch.nn.functional.kl_div(
                    log_masked, log_clean, log_target=True, reduction="none"
                ).sum(dim=-1)
                branch_kl = kl_tokens[0, prefix_len - 1 : full_len - 1].cpu().tolist()

                if collect_per_token:
                    per_token_kl_branches.append(branch_kl)

                if collect_per_sentence and tokenizer is not None:
                    cont_token_ids = cont[0]
                    cont_sents = split_tokens_into_sentences(
                        cont_token_ids,
                        tokenizer,
                        min_sentence_length=min_sentence_length,
                    )
                    sent_kl_list = []
                    for s in cont_sents:
                        s_kl = branch_kl[s.start : s.end + 1]
                        avg = sum(s_kl) / max(len(s_kl), 1)
                        text = tokenizer.decode(
                            cont_token_ids[s.start : s.end + 1].tolist()
                        )
                        sent_kl_list.append({"text": text, "mean_kl": avg})
                    per_sent_kl_branches.append(sent_kl_list)

    remove_handles(handles)

    avg_objective = total_objective / max(total_branches, 1)
    return avg_objective, per_token_kl_branches, per_sent_kl_branches


# ------------------------------------------------------------------
# Main threshold sweep
# ------------------------------------------------------------------


def evaluate_at_thresholds(
    model: torch.nn.Module,
    node_mask: NodeMask,
    input_ids: torch.Tensor,
    sentences: list[Sentence],
    continuations: list[torch.Tensor],
    objective_fn: Callable,
    thresholds: list[float],
    layers: list[int],
    ablate_non_target_layers: bool = False,
    renormalize_masked_attention: bool = True,
    tokenizer=None,
    min_sentence_length: int = 10,
) -> list[dict]:
    """Evaluate objective value and sparsity at different mask thresholds.

    For each threshold:
    - Compute sparsity from the mask
    - Re-run model with thresholded binary mask
    - Compute objective with clean output

    Thresholding uses score sign semantics from node_mask metadata:
    - negate_scores=True (default): keep score >= threshold
    - negate_scores=False: keep score <= -threshold
    """
    device = next(model.parameters()).device
    num_heads = model.config.num_attention_heads
    prefix_len = input_ids.shape[-1]
    num_sents = len(sentences)
    max_cont_len = max(c.shape[-1] for c in continuations)
    total_seq_len = prefix_len + max_cont_len

    token_to_sent = build_token_to_sent_map(sentences, total_seq_len, device)

    sentence_gap = 0
    if hasattr(node_mask, "metadata"):
        sentence_gap = node_mask.metadata.get("sentence_gap", 0)
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=device)

    # Build combined filter (gap + mode) to match discovery-time filtering
    mask_mode = node_mask.metadata.get("mask_mode", "prefix")
    num_prefix_sents = node_mask.metadata.get("num_prefix_sentences", num_sents)
    mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode, device=device)
    combined_filter = build_combined_filter(gap_filter, mode_filter)
    combined_filter_cpu = combined_filter.cpu()

    # Optionally ablate non-target layers
    non_target_handles: list[AblationHandle] = []
    if ablate_non_target_layers:
        non_target_handles = install_non_target_ablation(
            model,
            layers,
            num_heads,
            num_sents,
            token_to_sent,
            combined_filter,
            renormalize_masked_attention,
            device,
        )

    # Compute clean logits
    print("Computing clean logits for threshold evaluation...")
    clean_logits_list = compute_clean_logits(model, input_ids, continuations)

    results = []
    for threshold in thresholds:
        sparsity = node_mask.sparsity(threshold, gap_filter=combined_filter_cpu)

        binary_masks = build_binary_masks(
            node_mask,
            threshold,
            layers,
            num_heads,
            num_sents,
            combined_filter,
            device,
        )

        avg_objective, per_token_kl_branches, per_sent_kl_branches = eval_with_masks(
            model=model,
            binary_masks=binary_masks,
            layers=layers,
            input_ids=input_ids,
            continuations=continuations,
            clean_logits_list=clean_logits_list,
            objective_fn=objective_fn,
            token_to_sent=token_to_sent,
            gap_filter=combined_filter,
            renormalize=renormalize_masked_attention,
            collect_per_token=True,
            collect_per_sentence=True,
            tokenizer=tokenizer,
            min_sentence_length=min_sentence_length,
        )

        keep_prob = max(0.0, min(1.0, 1.0 - sparsity))
        random_masks = build_random_masks(
            keep_prob,
            layers,
            num_heads,
            num_sents,
            combined_filter,
            device,
        )

        random_objective, _, _ = eval_with_masks(
            model=model,
            binary_masks=random_masks,
            layers=layers,
            input_ids=input_ids,
            continuations=continuations,
            clean_logits_list=clean_logits_list,
            objective_fn=objective_fn,
            token_to_sent=token_to_sent,
            gap_filter=combined_filter,
            renormalize=renormalize_masked_attention,
        )

        entry = {
            "threshold": threshold,
            "sparsity": sparsity,
            "kl_divergence": avg_objective,
            "random_kl_divergence": random_objective,
            "per_token_kl": per_token_kl_branches,
        }
        if per_sent_kl_branches:
            entry["per_sentence_kl"] = per_sent_kl_branches
        results.append(entry)
        print(
            f"  threshold={threshold:.1e} | sparsity={sparsity:.2%} | objective={avg_objective:.6f}"
        )

    remove_handles(non_target_handles)
    return results
