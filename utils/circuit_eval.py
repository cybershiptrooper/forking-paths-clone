"""Evaluate a learned circuit mask at different sparsity thresholds.

Provides helpers for:
- Building binary masks from a NodeMask at a given threshold
- Installing / removing attention ablation hooks
- Running the model with masked attention and computing all metrics
- Orchestrating threshold sweeps with random-mask baselines
"""

import types
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from utils.utils import Sentence, get_attention_module
from utils.masks import NodeMask, build_gap_filter, apply_gap_filter, build_mode_filter, build_combined_filter, build_causal_filter
from utils.cot_analysis import split_tokens_into_sentences
from utils.importance_sampling import chain_log_prob, importance_weights, effective_sample_size, snis_answer_probs
from utils.circuit_discovery.common import (
    make_attention_forward,
    apply_sentence_mask,
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


def _threshold_2d(
    scores_2d: list[list[float]], threshold: float, num_sents: int, device: torch.device,
) -> torch.Tensor:
    """Threshold a 2D score matrix into a binary (S, S) tensor."""
    m = torch.ones(num_sents, num_sents, device=device)
    for i in range(num_sents):
        for j in range(num_sents):
            if scores_2d[i][j] < threshold:
                m[i, j] = 0.0
    return m


def build_binary_masks(
    scores: NodeMask | dict | list,
    threshold: float,
    layers: list[int],
    num_heads: int,
    num_sents: int,
    gap_filter: torch.Tensor,
    device: torch.device,
    granularity: str = "head",
) -> dict[int, torch.Tensor]:
    """Threshold scores into binary masks of shape ``(H, S, S)`` per layer.

    Handles all three granularities:

    - ``"head"``: ``scores[layer][head][i][j]``
    - ``"layer"``: ``scores[layer][i][j]`` — broadcast to all heads
    - ``"pair"``: ``scores[i][j]`` — broadcast to all heads and layers

    Returns ``{layer: (num_heads, num_sents, num_sents)}`` tensors.
    """
    if isinstance(scores, NodeMask):
        granularity = scores.granularity
        scores_data = scores.scores
    else:
        scores_data = scores

    binary_masks: dict[int, torch.Tensor] = {}

    if granularity == "pair":
        # scores_data is a 2D list
        base = _threshold_2d(scores_data, threshold, num_sents, device)
        base = apply_gap_filter(base, gap_filter, fill_value=1.0)
        expanded = base.unsqueeze(0).expand(num_heads, -1, -1)
        for layer in layers:
            binary_masks[layer] = expanded
    elif granularity == "layer":
        for layer in layers:
            base = _threshold_2d(scores_data[layer], threshold, num_sents, device)
            base = apply_gap_filter(base, gap_filter, fill_value=1.0)
            binary_masks[layer] = base.unsqueeze(0).expand(num_heads, -1, -1)
    else:  # "head"
        for layer in layers:
            m = torch.ones(num_heads, num_sents, num_sents, device=device)
            for h in range(num_heads):
                layer_scores = scores_data[layer][h]
                for i in range(num_sents):
                    for j in range(num_sents):
                        if layer_scores[i][j] < threshold:
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


def build_random_score_masks(
    node_mask: NodeMask,
    num_samples: int,
    layers: list[int],
    combined_filter: torch.Tensor,
):
    """Create *num_samples* random score masks by permuting learned scores.

    Permutes at the native granularity of the mask so the random baseline
    has the same structural constraints as the learned mask:

    - ``"head"``: permute ``(layer, head, i, j)`` positions
    - ``"layer"``: permute ``(layer, i, j)`` positions
    - ``"pair"``: permute ``(i, j)`` positions
    """
    num_sents = combined_filter.shape[0]
    filter_bool = combined_filter.bool()
    g = node_mask.granularity

    if g == "head":
        positions: list[tuple] = []
        score_values: list[float] = []
        for layer in layers:
            for h in node_mask.scores[layer]:
                scores_2d = node_mask.scores[layer][h]
                for i in range(num_sents):
                    for j in range(num_sents):
                        if not filter_bool[i, j]:
                            positions.append((layer, h, i, j))
                            score_values.append(scores_2d[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_dict: dict = {}
            for layer in layers:
                scores_dict[layer] = {}
                for h in node_mask.scores[layer]:
                    scores_dict[layer][h] = [
                        [0.0] * num_sents for _ in range(num_sents)
                    ]
            for idx, (layer, h, i, j) in enumerate(positions):
                scores_dict[layer][h][i][j] = permuted[idx]
            random_masks.append(scores_dict)
        return random_masks

    elif g == "layer":
        positions = []
        score_values = []
        for layer in layers:
            scores_2d = node_mask.scores[layer]
            for i in range(num_sents):
                for j in range(num_sents):
                    if not filter_bool[i, j]:
                        positions.append((layer, i, j))
                        score_values.append(scores_2d[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_dict = {}
            for layer in layers:
                scores_dict[layer] = [
                    [0.0] * num_sents for _ in range(num_sents)
                ]
            for idx, (layer, i, j) in enumerate(positions):
                scores_dict[layer][i][j] = permuted[idx]
            random_masks.append(scores_dict)
        return random_masks

    else:  # "pair"
        positions = []
        score_values = []
        for i in range(num_sents):
            for j in range(num_sents):
                if not filter_bool[i, j]:
                    positions.append((i, j))
                    score_values.append(node_mask.scores[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_2d = [[0.0] * num_sents for _ in range(num_sents)]
            for idx, (i, j) in enumerate(positions):
                scores_2d[i][j] = permuted[idx]
            random_masks.append(scores_2d)
        return random_masks


def compute_clean_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
) -> list[torch.Tensor]:
    """Run the model on each continuation and return logits on CPU.

    Called after non-target-layer ablation hooks are installed, so the
    resulting logits reflect the baseline with non-target layers ablated
    but target layers unmasked.  Uses chunked forward when sequences are
    long to avoid OOM from eager attention.
    """
    prefix_len = input_ids.shape[-1]
    max_cont_len = max(c.shape[-1] for c in continuations)
    _do_chunk = use_chunked_forward and (prefix_len + max_cont_len) > chunk_size * 2

    clean_logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            if _do_chunk:
                logits = _chunked_forward(
                    model, full_input, prefix_len, chunk_size
                )
            else:
                logits = model(full_input).logits
            clean_logits.append(logits.cpu())
            del logits
            torch.cuda.empty_cache()
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
    forward_fn = make_attention_forward(model.config.model_type, apply_sentence_mask)
    handles: list[AblationHandle] = []
    for layer_idx in layers:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = binary_masks[layer_idx]
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
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

    forward_fn = make_attention_forward(model.config.model_type, apply_sentence_mask)
    handles: list[AblationHandle] = []
    for layer_idx in non_target:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = filled_mask
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def remove_handles(handles: list[AblationHandle]):
    """Remove all ablation handles (restore original forwards)."""
    for h in handles:
        h.remove()


# ------------------------------------------------------------------
# Unified evaluation pass — all metrics from a single forward pass
# ------------------------------------------------------------------


def _chunked_forward(
    model: torch.nn.Module,
    full_input: torch.Tensor,
    prefix_len: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Run a forward pass in chunks using KV cache to reduce peak memory.

    Splits the input into prefix + continuation chunks.  Each chunk's
    attention matrix is only ``(heads, chunk_len, accumulated_len)`` instead
    of ``(heads, full_len, full_len)``.  The model's attention hooks (masks)
    are active for all chunks because hooks are installed by monkey-patching
    each attention module's ``.forward`` — ``model(chunk, past_key_values=...)``
    still calls the hooked forward with ``cache_position`` so the mask
    correctly indexes sentence positions for each chunk.

    This is mathematically equivalent to a single full forward pass because
    causal attention means token *i*'s output only depends on tokens 0..i,
    all of which are in the KV cache by the time token *i* is processed.

    Args:
        model: The (hooked) model.
        full_input: ``(1, full_len)`` input token IDs.
        prefix_len: Number of prefix tokens (first chunk boundary).
        chunk_size: Max tokens per continuation chunk.

    Returns:
        ``(1, full_len, vocab)`` logits — identical to ``model(full_input).logits``.
    """
    from transformers import DynamicCache

    device = full_input.device
    full_len = full_input.shape[-1]
    past_key_values = DynamicCache()
    all_logits: list[torch.Tensor] = []

    # Build chunk boundaries: [0, prefix_len, prefix_len+chunk, ...]
    boundaries = [0, prefix_len]
    pos = prefix_len
    while pos < full_len:
        boundaries.append(min(pos + chunk_size, full_len))
        pos += chunk_size

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        chunk_ids = full_input[:, start:end]
        cache_position = torch.arange(start, end, device=device)

        outputs = model(
            chunk_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        all_logits.append(outputs.logits)
        past_key_values = outputs.past_key_values

    # Free KV cache
    del past_key_values

    return torch.cat(all_logits, dim=1)


def eval_all_metrics(
    model: torch.nn.Module,
    binary_masks: dict[int, torch.Tensor],
    layers: list[int],
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
    clean_logits_list: list[torch.Tensor],
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    tokenizer=None,
    min_sentence_length: int = 10,
    branch_rewards: list[float] | None = None,
    position_mask_overrides: list[torch.Tensor | None] | None = None,
    chain_logprobs_clean: Optional[torch.Tensor] = None,
    answer_ids: Optional[torch.Tensor] = None,
    num_answers: Optional[int] = None,
    num_tokens_to_analyse: Optional[int] = None,
    collect_per_sentence: bool = True,
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
) -> dict:
    """Run model with *binary_masks* and compute all metrics in a single pass.

    For each continuation, one forward pass yields logits from which we extract:

    **Local metrics (always computed):**
    - ``kl_divergence``: plain unweighted mean per-token KL over the
      ``num_tokens_to_analyse`` window (or all continuation tokens if None).
    - ``reward_weighted_kl``: same KL multiplied by per-branch reward weights
      (only when ``branch_rewards`` is provided).
    - ``per_sentence_kl``: per-sentence mean KL for each branch.

    **IS-based metrics (when ``answer_ids`` is provided):**
    - ``answer_kl``: KL(P_clean || P_m) over the answer distribution.
    - ``reward_gap``, ``p_target``, ``p_best_other``: reward gap decomposition.
    - ``answer_probs_masked``: per-answer probabilities under the masked model.
    - ``n_eff``, ``n_eff_ratio``: effective sample size diagnostics.
    - ``log_weights``: raw log importance weights.

    **Contrastive metrics (when ``answer_ids`` is provided):**
    - ``kl_a``: mean per-token KL over target-answer chains.
    - ``kl_b``: mean per-token KL over other-answer chains.
    - ``contrastive_loss``: ``kl_a - kl_b``.

    Args:
        num_tokens_to_analyse: If set, local KL is computed over only the
            first N continuation tokens. Chain logprobs for IS metrics
            always use the full branch.
        collect_per_sentence: Whether to compute per-sentence KL breakdown.
        use_chunked_forward: If True, use KV-cache chunked forward to reduce
            peak attention memory from O(seq^2) to O(chunk * seq).  Set to
            False for raw full forward passes (faster for short sequences).
        chunk_size: Max tokens per chunk when using chunked forward.
    """
    from utils.objectives import (
        answer_distribution_kl_loss,
        reward_gap_loss,
    )

    device = next(model.parameters()).device
    prefix_len = input_ids.shape[-1]
    compute_is = (
        answer_ids is not None
        and num_answers is not None
        and chain_logprobs_clean is not None
    )

    # Auto-select: only chunk when sequences are long enough to benefit
    max_cont_len = max(c.shape[-1] for c in continuations)
    _do_chunk = use_chunked_forward and (prefix_len + max_cont_len) > chunk_size * 2

    handles = install_mask_hooks(
        model, layers, binary_masks, token_to_sent, gap_filter, renormalize
    )

    # Accumulators
    total_kl = 0.0
    total_weighted_kl = 0.0
    per_branch_kl: list[float] = []  # mean KL per branch (for contrastive)
    per_sent_kl_branches: list[list[dict]] = []
    chain_lps: list[torch.Tensor] = []
    total_branches = 0

    with torch.no_grad():
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]

            if _do_chunk:
                logits = _chunked_forward(
                    model, full_input, prefix_len, chunk_size
                )
            else:
                logits = model(full_input).logits
            out_device = logits.device
            clean = clean_logits_list[cont_idx][:, :full_len].to(out_device)

            # --- Per-token KL (computed for all metrics, not saved raw) ---
            log_clean = F.log_softmax(clean.detach().float(), dim=-1)
            log_masked = F.log_softmax(logits.float(), dim=-1)
            kl_tokens = F.kl_div(
                log_masked, log_clean, log_target=True, reduction="none"
            ).sum(dim=-1)  # (1, seq_len)

            # Local KL window: first num_tokens_to_analyse continuation tokens
            cont_len = full_len - prefix_len
            if num_tokens_to_analyse is not None:
                analyse_len = min(num_tokens_to_analyse, cont_len)
            else:
                analyse_len = cont_len
            analyse_end = prefix_len + analyse_len

            # Build position mask for local KL
            local_pos_mask = torch.zeros(1, full_len, device=out_device)
            local_pos_mask[0, prefix_len - 1 : analyse_end - 1] = 1.0

            # Apply position_mask_overrides if provided (e.g. answer-only)
            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                pos_mask_override = position_mask_overrides[cont_idx].to(out_device)
                # Intersect: override mask AND local window
                effective_mask = torch.zeros(1, full_len, device=out_device)
                effective_mask[0, :pos_mask_override.shape[-1]] = pos_mask_override[0, :full_len]
                effective_mask[0, analyse_end - 1 :] = 0.0  # clip to analysis window
                local_kl = (kl_tokens * effective_mask).sum() / effective_mask.sum().clamp(min=1)
            else:
                local_kl = (kl_tokens * local_pos_mask).sum() / local_pos_mask.sum().clamp(min=1)

            branch_kl_val = local_kl.item()
            total_kl += branch_kl_val
            per_branch_kl.append(branch_kl_val)

            # Reward-weighted KL
            if branch_rewards is not None:
                total_weighted_kl += branch_kl_val * branch_rewards[cont_idx]

            # --- Per-sentence KL (within analysis window) ---
            if collect_per_sentence and tokenizer is not None:
                # Per-token KL for the continuation (full, for sentence splitting)
                branch_kl_tokens = kl_tokens[0, prefix_len - 1 : full_len - 1].cpu().tolist()
                cont_token_ids = cont[0]
                cont_sents = split_tokens_into_sentences(
                    cont_token_ids,
                    tokenizer,
                    min_sentence_length=min_sentence_length,
                )
                sent_kl_list = []
                for s in cont_sents:
                    s_kl = branch_kl_tokens[s.start : s.end + 1]
                    avg = sum(s_kl) / max(len(s_kl), 1)
                    text = tokenizer.decode(
                        cont_token_ids[s.start : s.end + 1].tolist()
                    )
                    sent_kl_list.append({"text": text, "mean_kl": avg})
                per_sent_kl_branches.append(sent_kl_list)

            # --- Chain log-prob for IS metrics (over FULL branch) ---
            if compute_is:
                lp = chain_log_prob(logits.float(), full_input, prefix_len)
                chain_lps.append(lp)

            total_branches += 1

            # Free GPU memory between branches
            del logits, clean, log_clean, log_masked, kl_tokens
            del full_input, local_pos_mask
            torch.cuda.empty_cache()

    remove_handles(handles)

    # Assemble results
    result: dict = {
        "kl_divergence": total_kl / max(total_branches, 1),
    }

    if branch_rewards is not None:
        result["reward_weighted_kl"] = total_weighted_kl / max(total_branches, 1)

    if per_sent_kl_branches:
        result["per_sentence_kl"] = per_sent_kl_branches

    # --- IS-based metrics ---
    if compute_is:
        chain_lps_t = torch.stack(chain_lps).to(device)
        clean_lps = chain_logprobs_clean.to(device)
        answer_ids_dev = answer_ids.to(device)

        w = importance_weights(chain_lps_t, clean_lps)
        n_eff = effective_sample_size(w)
        p_m = snis_answer_probs(w, answer_ids_dev, num_answers)

        # Answer KL (Objective 1)
        answer_kl = answer_distribution_kl_loss(
            chain_lps_t, clean_lps, answer_ids_dev, num_answers,
        ).item()

        # Reward gap (Objective 2)
        target_answer = 0
        p_target = p_m[target_answer].item()
        other_mask = torch.ones(num_answers, dtype=torch.bool, device=device)
        other_mask[target_answer] = False
        p_best_other = p_m[other_mask].max().item() if other_mask.any() else 0.0
        reward_gap = p_target - p_best_other

        result["answer_kl"] = answer_kl
        result["reward_gap"] = reward_gap
        result["p_target"] = p_target
        result["p_best_other"] = p_best_other
        result["answer_probs_masked"] = p_m.detach().cpu().tolist()
        result["n_eff"] = n_eff
        result["n_eff_ratio"] = n_eff / len(continuations)
        result["log_weights"] = (
            (chain_lps_t - clean_lps).detach().cpu().tolist()
        )

    # --- Contrastive metrics (group per-branch KL by answer_ids) ---
    if answer_ids is not None and num_answers is not None:
        target_answer = 0
        kl_a_vals = []
        kl_b_vals = []
        for i, bkl in enumerate(per_branch_kl):
            if answer_ids[i].item() == target_answer:
                kl_a_vals.append(bkl)
            else:
                kl_b_vals.append(bkl)
        kl_a = sum(kl_a_vals) / max(len(kl_a_vals), 1) if kl_a_vals else 0.0
        kl_b = sum(kl_b_vals) / max(len(kl_b_vals), 1) if kl_b_vals else 0.0
        result["kl_a"] = kl_a
        result["kl_b"] = kl_b
        result["contrastive_loss"] = kl_a - kl_b

    return result


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
    num_random_samples: int = 5,
    branch_rewards: list[float] | None = None,
    position_mask_overrides: list[torch.Tensor | None] | None = None,
    answer_ids: Optional[torch.Tensor] = None,
    num_answers: Optional[int] = None,
    num_tokens_to_analyse: Optional[int] = None,
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
) -> list[dict]:
    """Evaluate all metrics at different mask thresholds.

    For each threshold, runs :func:`eval_all_metrics` once on the learned mask
    and *K* times on random baseline masks.  All available metrics (local, IS,
    contrastive) are computed in a single forward pass per continuation.

    Args:
        num_tokens_to_analyse: If set, local KL metrics use only the first N
            continuation tokens.  IS metrics always use the full branch.
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

    # Build combined filter (gap + mode + causal) to match discovery-time filtering
    mask_mode = node_mask.metadata.get("mask_mode", "prefix")
    num_prefix_sents = node_mask.metadata.get("num_prefix_sentences", num_sents)
    mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
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
    clean_logits_list = compute_clean_logits(
        model, input_ids, continuations,
        use_chunked_forward=use_chunked_forward,
        chunk_size=chunk_size,
    )

    # Compute clean chain logprobs when answer_ids are available (for IS metrics)
    chain_logprobs_clean = None
    if answer_ids is not None and num_answers is not None:
        chain_logprobs_clean = []
        for ci, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
            lp = chain_log_prob(clean_logits, full_input.cpu(), prefix_len)
            chain_logprobs_clean.append(lp.detach())
        chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

    # Shared kwargs for eval_all_metrics
    shared_kwargs = dict(
        layers=layers,
        input_ids=input_ids,
        continuations=continuations,
        clean_logits_list=clean_logits_list,
        token_to_sent=token_to_sent,
        gap_filter=combined_filter,
        renormalize=renormalize_masked_attention,
        tokenizer=tokenizer,
        min_sentence_length=min_sentence_length,
        branch_rewards=branch_rewards,
        position_mask_overrides=position_mask_overrides,
        chain_logprobs_clean=chain_logprobs_clean,
        answer_ids=answer_ids,
        num_answers=num_answers,
        num_tokens_to_analyse=num_tokens_to_analyse,
        use_chunked_forward=use_chunked_forward,
        chunk_size=chunk_size,
    )

    # Pre-generate K random score masks by permuting learned scores
    print(f"Generating {num_random_samples} random score masks (permuted)...")
    random_score_masks = build_random_score_masks(
        node_mask, num_random_samples, layers, combined_filter_cpu
    )

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

        # Single call: all metrics from one forward pass per continuation
        learned = eval_all_metrics(
            model=model, binary_masks=binary_masks, **shared_kwargs,
        )

        # Evaluate K random score masks at this threshold
        random_results: list[dict] = []
        granularity = node_mask.granularity
        for k in range(num_random_samples):
            rand_binary = build_binary_masks(
                random_score_masks[k],
                threshold,
                layers,
                num_heads,
                num_sents,
                combined_filter,
                device,
                granularity=granularity,
            )
            rand_result = eval_all_metrics(
                model=model,
                binary_masks=rand_binary,
                # Skip per-sentence KL for random baselines (expensive, not needed)
                collect_per_sentence=False,
                **shared_kwargs,
            )
            random_results.append(rand_result)
            del rand_binary
        del binary_masks
        torch.cuda.empty_cache()

        # --- Build entry dict ---
        entry: dict = {
            "threshold": threshold,
            "sparsity": sparsity,
            "kl_divergence": learned["kl_divergence"],
            "random_kl_divergence": _mean_field(random_results, "kl_divergence"),
            "random_kl_divergences": [r["kl_divergence"] for r in random_results],
        }

        # Reward-weighted KL
        if "reward_weighted_kl" in learned:
            entry["reward_weighted_kl"] = learned["reward_weighted_kl"]
            entry["random_reward_weighted_kl"] = _mean_field(random_results, "reward_weighted_kl")
            entry["random_reward_weighted_kls"] = [
                r.get("reward_weighted_kl", 0.0) for r in random_results
            ]

        # Per-sentence KL
        if "per_sentence_kl" in learned:
            entry["per_sentence_kl"] = learned["per_sentence_kl"]

        # IS-based metrics
        if "answer_kl" in learned:
            entry["answer_kl"] = learned["answer_kl"]
            entry["reward_gap"] = learned["reward_gap"]
            entry["p_target"] = learned["p_target"]
            entry["p_best_other"] = learned["p_best_other"]
            entry["answer_probs_masked"] = learned["answer_probs_masked"]
            entry["n_eff"] = learned["n_eff"]
            entry["n_eff_ratio"] = learned["n_eff_ratio"]
            entry["log_weights"] = learned["log_weights"]

            entry["random_answer_kl"] = _mean_field(random_results, "answer_kl")
            entry["random_answer_kls"] = [r.get("answer_kl", 0.0) for r in random_results]
            entry["random_reward_gap"] = _mean_field(random_results, "reward_gap")
            entry["random_reward_gaps"] = [r.get("reward_gap", 0.0) for r in random_results]
            entry["random_p_target"] = _mean_field(random_results, "p_target")
            entry["random_p_targets"] = [r.get("p_target", 0.0) for r in random_results]
            entry["random_p_best_other"] = _mean_field(random_results, "p_best_other")
            entry["random_p_best_others"] = [r.get("p_best_other", 0.0) for r in random_results]
            entry["random_n_effs"] = [r.get("n_eff", 0.0) for r in random_results]

        # Contrastive metrics
        if "kl_a" in learned:
            entry["kl_a"] = learned["kl_a"]
            entry["kl_b"] = learned["kl_b"]
            entry["contrastive_loss"] = learned["contrastive_loss"]
            entry["random_kl_a"] = _mean_field(random_results, "kl_a")
            entry["random_kl_as"] = [r.get("kl_a", 0.0) for r in random_results]
            entry["random_kl_b"] = _mean_field(random_results, "kl_b")
            entry["random_kl_bs"] = [r.get("kl_b", 0.0) for r in random_results]
            entry["random_contrastive_loss"] = _mean_field(random_results, "contrastive_loss")
            entry["random_contrastive_losses"] = [
                r.get("contrastive_loss", 0.0) for r in random_results
            ]

        results.append(entry)

        # --- Logging ---
        extra_parts = []
        if "answer_kl" in learned:
            extra_parts.append(f"answer_kl={learned['answer_kl']:.6f}")
            extra_parts.append(f"reward_gap={learned['reward_gap']:.4f}")
            extra_parts.append(
                f"N_eff={learned['n_eff']:.1f} ({learned['n_eff_ratio']:.1%})"
            )
        if "kl_a" in learned:
            extra_parts.append(
                f"kl_a={learned['kl_a']:.6f} kl_b={learned['kl_b']:.6f}"
            )
        extra = (" | " + " | ".join(extra_parts)) if extra_parts else ""
        print(
            f"  threshold={threshold:.1e} | sparsity={sparsity:.2%} "
            f"| KL={learned['kl_divergence']:.6f} "
            f"| random KL={entry['random_kl_divergence']:.6f} "
            f"(K={num_random_samples})"
            f"{extra}"
        )

    remove_handles(non_target_handles)
    return results


def _mean_field(results: list[dict], key: str) -> float:
    """Mean of a field across result dicts, skipping missing entries."""
    vals = [r[key] for r in results if key in r]
    return sum(vals) / max(len(vals), 1)
