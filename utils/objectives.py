"""Loss functions for circuit discovery optimization.

Local objectives (Objective 3 / contrastive KL):
    Take (clean_logits, masked_logits, position_mask) and return a
    differentiable scalar loss. Per-token, per-chain.

Global objectives (Objectives 1 & 2 / faithfulness, reward):
    Take (chain_logprobs_masked, chain_logprobs_clean, answer_ids, num_answers)
    and return a differentiable scalar loss. Operate across all chains via
    importance sampling.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from utils.importance_sampling import importance_weights, snis_answer_probs


def kl_divergence_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(softmax(clean) || softmax(masked)), averaged over valid positions and batch.

    Args:
        clean_logits: (batch, seq_len, vocab) — detached reference logits
        masked_logits: (batch, seq_len, vocab) — logits from masked model (has grad)
        position_mask: (batch, seq_len) — 1 for tokens to include, 0 to ignore.
            Typically 1 for continuation tokens, 0 for prompt prefix.

    Returns:
        Scalar KL divergence loss (differentiable w.r.t. masked_logits).
    """
    clean_log_probs = F.log_softmax(clean_logits.detach().float(), dim=-1)
    masked_log_probs = F.log_softmax(masked_logits.float(), dim=-1)

    # KL(P || Q) = sum P * (log P - log Q)
    kl = F.kl_div(
        masked_log_probs, clean_log_probs, log_target=True, reduction="none"
    ).sum(dim=-1)  # (batch, seq_len)

    if position_mask is not None:
        return (kl * position_mask).sum() / position_mask.sum().clamp(min=1)
    return kl.mean()


def log_prob_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-probability of actual tokens under the masked model.

    Maximising log-prob (minimising this loss) encourages the masked model
    to assign high probability to the same tokens as the original sequence.

    Args:
        clean_logits: Unused (kept for interface compatibility).
        masked_logits: (batch, seq_len, vocab) — logits from masked model.
        position_mask: (batch, seq_len) — 1 for tokens to include.
        token_ids: (batch, seq_len) — the actual input token IDs.
            Required for this objective.

    Returns:
        Scalar negative mean log-prob (differentiable w.r.t. masked_logits).
    """
    if token_ids is None:
        raise ValueError("log_prob_loss requires token_ids argument")

    log_probs = F.log_softmax(masked_logits.float(), dim=-1)
    # logits at position i predict token i+1
    targets = token_ids[:, 1:]  # (batch, seq_len - 1)
    token_lp = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    if position_mask is not None:
        pm = position_mask[:, :-1]
        return -(token_lp * pm).sum() / pm.sum().clamp(min=1)
    return -token_lp.mean()


OBJECTIVES = {
    "kl_divergence": kl_divergence_loss,
    "log_prob": log_prob_loss,
}


# ---------------------------------------------------------------------------
# Global objectives (Objectives 1 & 2)
# ---------------------------------------------------------------------------


def answer_distribution_kl_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    **kwargs,
) -> torch.Tensor:
    """KL(P_clean || P_m) over the answer distribution — Objective 1 (Faithfulness).

    P_clean is estimated by counting (chains were sampled from clean model).
    P_m is estimated via self-normalized importance sampling.

    Args:
        chain_logprobs_masked: (N,) log-probs under masked model (has grad)
        chain_logprobs_clean: (N,) log-probs under clean model (detached)
        answer_ids: (N,) integer answer IDs
        num_answers: number of distinct answers

    Returns:
        Scalar KL divergence (differentiable w.r.t. chain_logprobs_masked).
    """
    N = chain_logprobs_masked.shape[0]
    device = chain_logprobs_masked.device

    # P_clean: simple counting (chains were sampled from the clean model)
    p_clean = torch.zeros(num_answers, device=device)
    for a in range(num_answers):
        p_clean[a] = (answer_ids == a).float().sum() / N

    # P_m: importance sampling
    w = importance_weights(chain_logprobs_masked, chain_logprobs_clean)
    p_m = snis_answer_probs(w, answer_ids, num_answers)

    # KL(P_clean || P_m) — only over answers with non-zero P_clean
    active = p_clean > 0
    kl = (
        p_clean[active]
        * torch.log(p_clean[active] / p_m[active].clamp(min=1e-10))
    ).sum()
    return kl


def reward_gap_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Negative reward gap -(P_m(A) - max_{a!=A} P_m(a)) — Objective 2 (Reward).

    Minimizing this loss maximizes the probability gap for the target answer.

    Args:
        chain_logprobs_masked: (N,) log-probs under masked model (has grad)
        chain_logprobs_clean: (N,) log-probs under clean model (detached)
        answer_ids: (N,) integer answer IDs
        num_answers: number of distinct answers
        target_answer: which answer ID to promote (default 0)

    Returns:
        Scalar loss (differentiable w.r.t. chain_logprobs_masked).
    """
    w = importance_weights(chain_logprobs_masked, chain_logprobs_clean)
    p_m = snis_answer_probs(w, answer_ids, num_answers)

    p_target = p_m[target_answer]
    other_mask = torch.ones(num_answers, dtype=torch.bool, device=p_m.device)
    other_mask[target_answer] = False

    if other_mask.any():
        p_best_other = p_m[other_mask].max()
    else:
        p_best_other = torch.zeros(1, device=p_m.device)

    return -(p_target - p_best_other)


GLOBAL_OBJECTIVES = {
    "answer_kl": answer_distribution_kl_loss,
    "reward_gap": reward_gap_loss,
}


_GLOBAL_FUNC_NAMES = {fn.__name__ for fn in GLOBAL_OBJECTIVES.values()}


def is_global_objective(name: str) -> bool:
    """Check if an objective is a global (outcome-level, IS-based) objective.

    Accepts both the registry key (e.g. ``"answer_kl"``) and the Python
    function name (e.g. ``"answer_distribution_kl_loss"``).
    """
    return name in GLOBAL_OBJECTIVES or name in _GLOBAL_FUNC_NAMES


def get_objective(name: str):
    """Get objective function by name (local or global)."""
    if name in OBJECTIVES:
        return OBJECTIVES[name]
    if name in GLOBAL_OBJECTIVES:
        return GLOBAL_OBJECTIVES[name]
    all_names = list(OBJECTIVES.keys()) + list(GLOBAL_OBJECTIVES.keys())
    raise ValueError(f"Unknown objective: {name}. Available: {all_names}")
