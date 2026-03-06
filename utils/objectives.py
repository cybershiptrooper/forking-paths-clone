"""Loss functions for circuit discovery optimization.

Each objective takes (clean_logits, masked_logits, position_mask) and returns
a differentiable scalar loss. clean_logits should be detached (no grad);
masked_logits should have grad for backpropagation through the mask.
"""

import torch
import torch.nn.functional as F
from typing import Optional


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


def get_objective(name: str):
    """Get objective function by name."""
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective: {name}. Available: {list(OBJECTIVES.keys())}"
        )
    return OBJECTIVES[name]
