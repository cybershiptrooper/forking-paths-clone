"""Mask thresholding helpers, copied from
``expts/direct_answer_circuit_discovery/eval_log_alpha.py`` (a copy, not
an import — that folder changes between experiments).  Plus a
random-mask constructor used for the matched-sparsity baseline.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_HC_BETA = 2.0 / 3.0
_HC_GAMMA = -0.1
_HC_ZETA = 1.1
M_GT_0_LOG_ALPHA_THRESHOLD = _HC_BETA * math.log(-_HC_GAMMA / _HC_ZETA)
M_GT_HALF_LOG_ALPHA_THRESHOLD = 0.0


def scores_to_log_alpha(scores, score_readout: str) -> torch.Tensor:
    """See eval_log_alpha._scores_to_log_alpha (copied verbatim)."""
    arr = torch.tensor(scores, dtype=torch.float32)
    if arr.dim() == 0:
        arr = arr.unsqueeze(0)
    if score_readout == "log_alpha":
        return arr
    if score_readout == "raw_score":
        return arr
    if score_readout in (None, "None"):
        if float(arr.min().item()) < 0.0 or float(arr.max().item()) > 1.0:
            return arr
    m = arr.clamp(0.0, 1.0)
    s = (m - _HC_GAMMA) / (_HC_ZETA - _HC_GAMMA)
    s = s.clamp(1e-6, 1 - 1e-6)
    la = _HC_BETA * (torch.log(s) - torch.log1p(-s))
    la = torch.where(m == 0.0, torch.full_like(la, -1e6), la)
    return la


def build_binary_mask(
    log_alpha: torch.Tensor,
    mode: str,
    target_sparsity: Optional[float],
    valid_filter: torch.Tensor,
) -> torch.Tensor:
    """See eval_log_alpha._build_binary_mask (copied verbatim)."""
    if mode == "m_gt_0":
        keep = (log_alpha > M_GT_0_LOG_ALPHA_THRESHOLD).float()
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


def random_binary_mask(
    shape, target_sparsity: float, valid_filter: torch.Tensor, seed: int,
) -> torch.Tensor:
    """Random mask at matched sparsity: random scores over valid cells,
    thresholded with the same top-k rule as a learned mask."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(shape, generator=g)
    return build_binary_mask(scores, "top_k", target_sparsity, valid_filter)


def binary_to_per_layer_masks(binary, layers, num_heads):
    if binary.dim() == 2:
        # Head-uniform (sentence-pair) mask: keep the head dim at size 1 and
        # let SDPA broadcast. Expanding to num_heads made the additive bias
        # (1, H, q, k); combined with the batched causal mask it broadcast
        # to (B, H, q, k) — 197 GiB at batch 16 on a 14.4k-token prefix.
        # With H = 1 the combined mask is (B, 1, q, k), identical math.
        expanded = binary.unsqueeze(0).contiguous()
        return {l: expanded for l in layers}
    return {l: binary[i].contiguous() for i, l in enumerate(layers)}
