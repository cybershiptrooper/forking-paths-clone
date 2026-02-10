"""Loss functions for circuit discovery.

Each objective is a callable: (clean_logits, masked_logits, positions?) -> scalar loss.
The loss must support .backward() for gradient-based attribution.
"""

from typing import Optional, Protocol

import torch
import torch.nn.functional as F


class CircuitObjective(Protocol):
    """Protocol for circuit discovery objectives."""

    def __call__(
        self,
        clean_logits: torch.Tensor,
        masked_logits: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


def kl_divergence_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL(clean || masked) averaged over specified positions and batch.

    Args:
        clean_logits: (batch, seq_len, vocab) detached reference logits.
        masked_logits: (batch, seq_len, vocab) logits with mask applied (grad enabled).
        positions: Optional (num_positions,) LongTensor of seq positions to evaluate.
            If None, uses all positions.

    Returns:
        Scalar KL divergence loss.
    """
    if positions is not None:
        clean_logits = clean_logits[:, positions, :]
        masked_logits = masked_logits[:, positions, :]

    clean_probs = F.softmax(clean_logits.detach(), dim=-1)
    masked_log_probs = F.log_softmax(masked_logits, dim=-1)

    kl = F.kl_div(masked_log_probs, clean_probs, reduction="batchmean", log_target=False)
    return kl


OBJECTIVES = {
    "kl_divergence": kl_divergence_loss,
}


def get_objective(name: str):
    """Get objective function by name."""
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective: {name}. Available: {list(OBJECTIVES.keys())}"
        )
    return OBJECTIVES[name]
