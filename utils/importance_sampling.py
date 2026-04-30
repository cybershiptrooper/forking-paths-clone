"""Importance sampling utilities for outcome-level circuit objectives.

Provides chain-level log-probability computation, importance weighting,
effective sample size, and SNIS estimation for answer distributions.
Used by global objectives (Objectives 1 & 2 from the roadmap).
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F

from utils.rewards import extract_boxed


def chain_log_prob(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prefix_len: int,
    temperature: float = 1.0,
    apply_temperature: bool = True,
) -> torch.Tensor:
    """Log-probability of a continuation under a model.

    Computes sum of per-token log-probs for tokens after prefix_len.

    Args:
        logits: (1, seq_len, vocab) model output logits (may require grad)
        token_ids: (1, seq_len) full token sequence (prompt + continuation)
        prefix_len: number of prompt tokens (continuation starts here)
        temperature: sampling temperature used during generation.
        apply_temperature: if True (default), scale logits by 1/temperature
            before computing log-probs so they match the actual sampling
            distribution. Set to False to use raw (temperature=1) logits.

    Returns:
        Scalar log-probability (differentiable w.r.t. logits).
    """
    seq_len = token_ids.shape[-1]

    # 1. Slice FIRST so cast/logsumexp only touch the (cont_len, V) window
    pred_logits = logits[:, prefix_len - 1 : seq_len - 1]   # view, no copy
    targets     = token_ids[:, prefix_len : seq_len]

    # 2. Cast only the slice to fp32 (precision needed for log-softmax)
    pred_logits = pred_logits.float()
    if apply_temperature and temperature != 1.0:
        pred_logits = pred_logits / temperature

    # 3. gather + logsumexp instead of log_softmax + gather
    target_logits = pred_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (1, L)
    lse           = torch.logsumexp(pred_logits, dim=-1)                        # (1, L)
    token_lp      = target_logits - lse                                         # (1, L)

    return token_lp.sum(dim=-1).squeeze(0)


_LIGER_LOSS = None


def _liger_loss():
    global _LIGER_LOSS
    if _LIGER_LOSS is None:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
        _LIGER_LOSS = LigerFusedLinearCrossEntropyLoss(
            reduction="sum",
            accum_dtype=torch.float32,
        )
    return _LIGER_LOSS


def chain_log_prob_fused(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_ids: torch.Tensor,
    prefix_len: int,
    temperature: float = 1.0,
    apply_temperature: bool = True,
    lm_head_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Memory-efficient chain log-probability via fused linear+cross-entropy.

    Drop-in replacement for ``chain_log_prob`` that takes hidden states and
    the LM head weight instead of pre-materialised logits. Liger's Triton
    kernel fuses the lm_head matmul with log-softmax+gather, so the
    ``(seq_len, vocab)`` tensor that drives chain_log_prob's OOM is never
    realized. fp32 accumulation preserves precision.

    Args:
        hidden: (1, seq_len, hidden_size) last hidden state from the base
            transformer (e.g. ``model.model(input_ids).last_hidden_state``).
        lm_head_weight: (vocab, hidden_size) — typically ``model.lm_head.weight``.
        token_ids: (1, seq_len) full token sequence.
        prefix_len: number of prompt tokens.
        temperature, apply_temperature: as in ``chain_log_prob``.
        lm_head_bias: optional (vocab,) — Llama/Qwen3 have no LM-head bias,
            so this is normally ``None``.

    Returns:
        Scalar log-probability of the continuation, differentiable w.r.t.
        ``hidden`` and ``lm_head_weight``.
    """
    seq_len = token_ids.shape[-1]
    pred_h = hidden[0, prefix_len - 1 : seq_len - 1]                    # (cont_len, d)
    targets = token_ids[0, prefix_len:seq_len].to(torch.long)            # (cont_len,)

    if apply_temperature and temperature != 1.0:
        # logits = h @ W.T (+ b). With bias=None, scaling h scales logits
        # exactly. With bias, scale W and b copies instead.
        if lm_head_bias is None:
            pred_h = pred_h / temperature
        else:
            inv_T = 1.0 / temperature
            lm_head_weight = lm_head_weight * inv_T
            lm_head_bias = lm_head_bias * inv_T

    # Multi-GPU sharded models: lm_head and the final transformer block can
    # live on different devices (accelerate dispatches them independently).
    # The fused kernel needs all tensors co-located. Move the small tensors
    # (hidden slice, targets) onto lm_head_weight's device.
    target_device = lm_head_weight.device
    pred_h = pred_h.to(target_device).contiguous()
    targets = targets.to(target_device).contiguous()
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.to(target_device)

    loss = _liger_loss()(lm_head_weight, pred_h, targets, bias=lm_head_bias)
    # Return on the caller-facing device (matches the original chain_log_prob
    # contract, which returned on logits.device — typically the same device
    # as token_ids in the caller's frame).
    return (-loss).to(token_ids.device)


@torch.compile(dynamic=True)
def _chunk_logp_compiled(h_chunk: torch.Tensor, W: torch.Tensor, tgt_chunk: torch.Tensor):
    """Compiled per-chunk log-prob: (h_chunk @ W.T) -> log_softmax-gather.

    Inputs are bf16, internal logits/logsumexp/gather run in fp32.
    Boundary chunks (smaller than the chunk_size used elsewhere) work via
    ``dynamic=True`` shape specialisation.
    """
    logits = (h_chunk @ W.T).float()                       # (C, V) fp32
    lse = torch.logsumexp(logits, dim=-1)                  # (C,)
    tgt_lp = logits.gather(-1, tgt_chunk[:, None]).squeeze(-1)
    return tgt_lp - lse                                    # (C,)


def chain_log_prob_chunked(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_ids: torch.Tensor,
    prefix_len: int,
    temperature: float = 1.0,
    apply_temperature: bool = True,
    lm_head_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Chunked-along-seq, fp32-internal chain log-probability.

    Manual alternative to ``chain_log_prob_fused`` (Liger): no fused Triton
    kernel — instead, the LM-head matmul + log-softmax + gather are run in
    chunks of ``chunk_size`` query positions, each chunk compiled via
    ``torch.compile``. Logits live only inside the per-chunk closure, in
    fp32 for precision; never materialises the full ``(seq, V)`` tensor.

    Args:
        hidden: (1, seq_len, hidden_size) last hidden state.
        lm_head_weight: (vocab, hidden_size).
        token_ids: (1, seq_len) full token sequence.
        prefix_len: number of prompt tokens.
        temperature, apply_temperature: as in ``chain_log_prob``.
        lm_head_bias: optional (vocab,). Llama/Qwen3 use ``None``.
        chunk_size: query-positions per chunk. 1024 is a good default for
            long sequences with vocabularies in the 100k–200k range.

    Returns:
        Scalar log-probability of the continuation.
    """
    seq_len = token_ids.shape[-1]
    pred_h = hidden[0, prefix_len - 1 : seq_len - 1]                    # (cont_len, d)
    targets = token_ids[0, prefix_len:seq_len].to(torch.long)            # (cont_len,)

    if apply_temperature and temperature != 1.0:
        if lm_head_bias is None:
            pred_h = pred_h / temperature
        else:
            inv_T = 1.0 / temperature
            lm_head_weight = lm_head_weight * inv_T
            lm_head_bias = lm_head_bias * inv_T

    target_device = lm_head_weight.device
    pred_h = pred_h.to(target_device).contiguous()
    targets = targets.to(target_device).contiguous()
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.to(target_device)

    total = pred_h.new_zeros((), dtype=torch.float32)
    L = pred_h.shape[0]
    for i in range(0, L, chunk_size):
        end = min(i + chunk_size, L)
        h_chunk = pred_h[i:end]
        tgt_chunk = targets[i:end]
        if lm_head_bias is None:
            chunk_lp = _chunk_logp_compiled(h_chunk, lm_head_weight, tgt_chunk)
        else:
            # Bias-aware path (no torch.compile) — Llama/Qwen3 don't hit this.
            logits = (h_chunk @ lm_head_weight.T + lm_head_bias).float()
            lse = torch.logsumexp(logits, dim=-1)
            tgt_lp = logits.gather(-1, tgt_chunk[:, None]).squeeze(-1)
            chunk_lp = tgt_lp - lse
        total = total + chunk_lp.sum()

    return total.to(token_ids.device)


def importance_weights(
    log_p_target: torch.Tensor,
    log_p_proposal: torch.Tensor,
    method: str = "snis",
    chain_lengths: Optional[torch.Tensor] = None,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    """Self-normalised importance weights with configurable length handling.

    Args:
        log_p_target: (N,) log-probs under target (masked) model.
        log_p_proposal: (N,) log-probs under proposal (clean) model.
        method: IS method. One of:
            - "snis" (default): standard self-normalised importance sampling.
            - "geometric_mean": divide log-ratio by chain length before softmax.
              Mitigates SNIS collapse on long chains; see
              notes/reward_gap_goodhart.md and notes/answer_kl_objectives.md.
            - "tempered_snis": divide log-ratio by a fixed scalar temperature
              T (independent of chain length) before softmax. T=1 recovers
              SNIS; T -> inf recovers uniform. See
              notes/geometric_mean_collapse.md.
        chain_lengths: (N,) int tensor of per-chain continuation lengths.
            Required for method="geometric_mean". Ignored otherwise.
        temperature: positive scalar. Required for method="tempered_snis".
            Ignored otherwise.

    Returns:
        (N,) normalized weights summing to 1, differentiable w.r.t.
        log_p_target.
    """
    log_w = log_p_target - log_p_proposal.detach()
    if method == "snis":
        pass
    elif method == "geometric_mean":
        if chain_lengths is None:
            raise ValueError(
                "importance_weights(method='geometric_mean') requires chain_lengths"
            )
        lengths = chain_lengths.to(
            device=log_w.device, dtype=log_w.dtype,
        ).clamp_min(1.0)
        log_w = log_w / lengths
    elif method == "tempered_snis":
        if temperature is None:
            raise ValueError(
                "importance_weights(method='tempered_snis') requires temperature"
            )
        if temperature <= 0:
            raise ValueError(
                f"tempered_snis temperature must be > 0, got {temperature}"
            )
        log_w = log_w / float(temperature)
    else:
        raise ValueError(
            f"Unknown importance sampling method: {method!r}. "
            f"Available: 'snis', 'geometric_mean', 'tempered_snis'."
        )
    # Shift for numerical stability before exp
    log_w = log_w - log_w.detach().max()
    w = torch.exp(log_w)
    return w / w.sum()


def effective_sample_size(weights: torch.Tensor) -> float:
    """Effective sample size from importance weights.

    Args:
        weights: (N,) importance weights (normalized or unnormalized)

    Returns:
        N_eff as a float. Equal weights → N, one dominant → ~1.
    """
    w_norm = weights.detach() / weights.detach().sum()
    return (1.0 / (w_norm**2).sum()).item()


def snis_answer_probs(
    weights: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
) -> torch.Tensor:
    """Self-normalized importance sampling estimator for P(answer).

    Args:
        weights: (N,) normalized importance weights (sum to 1)
        answer_ids: (N,) integer answer ID per chain
        num_answers: total number of distinct answers

    Returns:
        (num_answers,) estimated probabilities, differentiable w.r.t. weights
    """
    one_hot = F.one_hot(answer_ids.long(), num_answers).float()  # (N, num_answers)
    one_hot = one_hot.to(weights.device)
    return (weights.unsqueeze(-1) * one_hot).sum(dim=0)  # (num_answers,)


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for robust comparison.

    Applied automatically by :func:`extract_answer_ids`.  Handles:

    * Whitespace and surrounding ``$`` signs
    * Trailing zeros (``42.0`` → ``42``)
    * Simple ``\\frac{a}{b}`` → decimal
    * Cosmetic LaTeX commands (``\\left``, ``\\right``, ``\\,``, etc.)
    """
    s = answer.strip()
    # Strip surrounding dollar signs
    s = s.strip("$").strip()

    # Remove cosmetic LaTeX spacing / delimiter commands
    s = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)\b", "", s)
    # Remove \text{...} wrappers (keep inner text)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    # \dfrac / \tfrac are display-size variants of \frac; collapse to \frac
    s = re.sub(r"\\(?:d|t)frac\b", r"\\frac", s)
    s = s.strip()

    # Try \\frac{num}{den} → decimal
    frac_match = re.fullmatch(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                val = num / den
                if val == int(val):
                    return str(int(val))
                return f"{val:.10g}"
        except ValueError:
            pass

    # Try parsing as a number (handles "42", "42.0", "-3.00", "1e2", etc.)
    try:
        val = float(s)
        if val == int(val) and "e" not in s.lower():
            return str(int(val))
        return f"{val:.10g}"
    except ValueError:
        pass

    # Fallback: collapse whitespace
    return " ".join(s.split())


def extract_answer_ids(
    branches: list[dict],
    prefix_text: str = "",
    judge_client=None,
    judge_model: str | None = None,
    question: str | None = None,
) -> tuple[list[int], list[str]]:
    """Extract answers from branches and assign integer IDs.

    Uses ``\\boxed{}`` extraction with :func:`normalize_answer` applied to
    every extracted answer (Option A — always on).

    If *judge_client*, *judge_model*, and *question* are all provided, answers
    are clustered by mathematical equivalence via an LLM judge instead of
    exact string matching (Option B).

    Branches without a ``\\boxed{}`` answer each get a unique label.

    Args:
        branches: list of dicts with ``"text"`` key
        prefix_text: text preceding all branches
        judge_client: OpenAI-compatible client for LLM judge (Option B)
        judge_model: model identifier for the judge
        question: the original question text (needed for judge prompt)

    Returns:
        answer_ids: list of int, one per branch
        answer_labels: list of unique answer strings
    """
    # --- Extract and normalize raw answers ---
    raw_answers: list[str] = []
    no_answer_counter = 0
    for b in branches:
        full_text = prefix_text + b["text"]
        boxed = extract_boxed(full_text)
        if boxed is not None:
            raw_answers.append(normalize_answer(boxed))
        else:
            raw_answers.append(f"__no_answer_{no_answer_counter}")
            no_answer_counter += 1

    # --- Option B: cluster with LLM judge ---
    use_judge = (
        judge_client is not None
        and judge_model is not None
        and question is not None
    )
    if use_judge:
        return _cluster_with_judge(raw_answers, question, judge_client, judge_model)

    # --- Default (Option A): exact match on normalized strings ---
    unique: list[str] = []
    seen: dict[str, int] = {}
    for a in raw_answers:
        if a not in seen:
            seen[a] = len(unique)
            unique.append(a)

    answer_ids = [seen[a] for a in raw_answers]
    return answer_ids, unique


def _cluster_with_judge(
    raw_answers: list[str],
    question: str,
    client,
    judge_model: str,
) -> tuple[list[int], list[str]]:
    """Cluster answers by mathematical equivalence using an LLM judge.

    For each answer, check it against existing cluster representatives.
    If the judge says it's equivalent, merge; otherwise start a new cluster.
    """
    from utils.rewards import judge_answer

    clusters: list[tuple[str, list[int]]] = []  # (canonical, [indices])

    for i, ans in enumerate(raw_answers):
        if ans.startswith("__no_answer_"):
            clusters.append((ans, [i]))
            continue

        matched = False
        for canonical, indices in clusters:
            if canonical.startswith("__no_answer_"):
                continue
            if judge_answer(question, canonical, ans, client, judge_model):
                indices.append(i)
                matched = True
                break

        if not matched:
            clusters.append((ans, [i]))

    answer_ids = [0] * len(raw_answers)
    labels: list[str] = []
    for cid, (canonical, indices) in enumerate(clusters):
        labels.append(canonical)
        for idx in indices:
            answer_ids[idx] = cid

    return answer_ids, labels


def merge_no_answer_variants(
    answer_ids: list[int],
    answer_labels: list[str],
) -> tuple[list[int], list[str], int]:
    """Merge all ``__no_answer_*`` labels into a single ``__no_answer`` bucket.

    Args:
        answer_ids: per-branch integer answer IDs
        answer_labels: unique answer label strings

    Returns:
        (new_answer_ids, new_answer_labels, new_num_answers)
    """
    no_answer_indices = [
        i for i, label in enumerate(answer_labels) if label.startswith("__no_answer")
    ]
    if len(no_answer_indices) <= 1:
        return answer_ids, answer_labels, len(answer_labels)

    # Build old-to-new ID mapping
    new_labels: list[str] = []
    old_to_new: dict[int, int] = {}
    merged_id: int | None = None
    for old_id, label in enumerate(answer_labels):
        if label.startswith("__no_answer"):
            if merged_id is None:
                merged_id = len(new_labels)
                new_labels.append("__no_answer")
            old_to_new[old_id] = merged_id
        else:
            old_to_new[old_id] = len(new_labels)
            new_labels.append(label)

    new_ids = [old_to_new[a] for a in answer_ids]
    return new_ids, new_labels, len(new_labels)


def build_binary_answer_ids(
    answer_ids: list[int],
    answer_labels: list[str],
    correct_answer: str,
) -> tuple[list[int], list[str], int]:
    """Build binary (correct / incorrect) answer IDs from fine-grained labels.

    Matches *correct_answer* against *answer_labels* using
    :func:`normalize_answer`.  Any label that matches is mapped to bucket 0
    (``"correct"``), everything else to bucket 1 (``"incorrect"``).

    Args:
        answer_ids: per-branch fine-grained integer answer IDs
        answer_labels: unique answer label strings
        correct_answer: ground-truth answer string

    Returns:
        (binary_answer_ids, ["correct", "incorrect"], 2)
    """
    target_norm = normalize_answer(correct_answer)
    correct_old_ids = set()
    for old_id, label in enumerate(answer_labels):
        if label.startswith("__no_answer"):
            continue
        if normalize_answer(label) == target_norm:
            correct_old_ids.add(old_id)

    binary_ids = [0 if a in correct_old_ids else 1 for a in answer_ids]
    return binary_ids, ["correct", "incorrect"], 2


def reward_based_answer_ids(
    branch_rewards: list[float],
    binary: bool = True,
) -> tuple[list[int], list[str]]:
    """Create answer groups from per-branch reward values (Option C).

    Args:
        branch_rewards: per-branch reward values (e.g. +1/-1 for correctness)
        binary: if True, group into ``"correct"`` (reward > 0) and
            ``"incorrect"`` (reward <= 0).  If False, each distinct reward
            value becomes its own group.

    Returns:
        answer_ids: list of int, one per branch
        answer_labels: list of unique group label strings
    """
    if binary:
        raw_ids = [0 if r > 0 else 1 for r in branch_rewards]
        used = set(raw_ids)
        if used == {0}:
            return [0] * len(branch_rewards), ["correct"]
        if used == {1}:
            return [0] * len(branch_rewards), ["incorrect"]
        return raw_ids, ["correct", "incorrect"]

    # Per-distinct-reward bucketing
    unique_rewards = sorted(set(branch_rewards))
    reward_to_id = {r: i for i, r in enumerate(unique_rewards)}
    answer_ids = [reward_to_id[r] for r in branch_rewards]
    labels = [f"reward={r:+g}" for r in unique_rewards]
    return answer_ids, labels
