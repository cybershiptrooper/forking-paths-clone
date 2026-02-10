from typing import Optional
import torch
import torch.nn.functional as F


def kl_logits_objective(
    baseline_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute mean KL divergence between masked and baseline logits.

    Args:
        baseline_logits: Tensor of shape (T, V) or (B, T, V)
        masked_logits: Tensor of shape (T, V) or (B, T, V)
        token_mask: Optional tensor of shape (T,) or (B, T) with 1 for
            positions to include and 0 for positions to exclude.

    Returns:
        Scalar tensor (mean KL divergence).
    """
    if baseline_logits.dim() == 2:
        baseline_logits = baseline_logits.unsqueeze(0)
    if masked_logits.dim() == 2:
        masked_logits = masked_logits.unsqueeze(0)

    log_probs_masked = F.log_softmax(masked_logits, dim=-1)
    probs_baseline = F.softmax(baseline_logits, dim=-1)

    # KL per token: sum over vocab
    kl_per_token = F.kl_div(log_probs_masked, probs_baseline, reduction="none").sum(-1)

    if token_mask is not None:
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)
        masked_kl = kl_per_token * token_mask
        denom = token_mask.sum().clamp_min(1.0)
        return masked_kl.sum() / denom

    return kl_per_token.mean()

