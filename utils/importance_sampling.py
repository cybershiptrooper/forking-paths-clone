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
    scaled = logits.float() / temperature if apply_temperature and temperature != 1.0 else logits.float()
    log_probs = F.log_softmax(scaled, dim=-1)
    seq_len = token_ids.shape[-1]
    # logits[t] predicts token_ids[t+1]
    # Continuation tokens: token_ids[prefix_len .. seq_len-1]
    # Predicted by: logits[prefix_len-1 .. seq_len-2]
    targets = token_ids[:, prefix_len:seq_len]  # (1, cont_len)
    preds = log_probs[:, prefix_len - 1 : seq_len - 1]  # (1, cont_len, V)
    token_lp = preds.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, cont_len)
    return token_lp.sum(dim=-1).squeeze(0)  # scalar


def importance_weights(
    log_p_target: torch.Tensor,
    log_p_proposal: torch.Tensor,
) -> torch.Tensor:
    """Self-normalized importance weights.

    Args:
        log_p_target: (N,) log-probs under target (masked) model
        log_p_proposal: (N,) log-probs under proposal (clean) model

    Returns:
        (N,) normalized weights summing to 1, differentiable w.r.t. log_p_target
    """
    log_w = log_p_target - log_p_proposal.detach()
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
