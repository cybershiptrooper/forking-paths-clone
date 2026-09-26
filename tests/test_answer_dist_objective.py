"""Tests for boundary_answer_dist_kl_length_loss (fix 2 of the 2026-09-21
meeting notes)."""

import math

import torch

from utils import objectives
from utils.objectives import boundary_answer_dist_kl_length_loss


def _toy(h_first):
    """Two continuations X (ends on A) and Y (ends on B), both with a first
    break at token 10 whose probe says B, and a late break (X: A, Y: B)."""
    H = 4096
    log_h = [
        torch.log(torch.tensor([h_first, 0.5])).requires_grad_(True),
        torch.log(torch.tensor([h_first, 0.5])).requires_grad_(True),
    ]
    positions = [torch.tensor([10, 1000]), torch.tensor([10, 1000])]
    probe = [
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    ]
    final = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    empty = [torch.zeros(2, dtype=torch.bool)] * 2
    loss = boundary_answer_dist_kl_length_loss(
        log_h=log_h, eligible=empty, clean_log_h=log_h, gaps=positions,
        horizon=H, positions=positions, probe_probs=probe, final_probs=final,
        answer_kl_weight=1.0,
    )
    return loss, log_h, dict(objectives.LAST_DIAGNOSTICS)


def test_no_stops_matches_reference():
    loss, _, d = _toy(1e-9)
    # With h at the first break ~0: each chain stops at its late break
    # with probability 0.5 on its own answer, otherwise S_all on its own
    # final answer, so the predicted distribution equals the reference.
    assert abs(d["answer_dist_kl"]) < 1e-6
    assert abs(d["p_pred_0"] - 0.5) < 1e-5 and abs(d["p_ref_0"] - 0.5) < 1e-5
    expected_len = (0.5 * 1000 + 0.5 * 4096) / 4096
    assert abs(d["expected_length_frac_of_horizon"] - expected_len) < 1e-4


def test_shared_early_stop_moves_distribution_to_b():
    _, _, d0 = _toy(1e-9)
    _, _, d1 = _toy(0.7)
    assert d1["expected_length_frac_of_horizon"] < d0["expected_length_frac_of_horizon"]
    # X now stops on B with probability 0.7: p_pred(A) = 0.5 * 0.3
    assert abs(d1["p_pred_0"] - 0.15) < 1e-4
    kl = 0.5 * math.log(0.5 / 0.15) + 0.5 * math.log(0.5 / 0.85)
    assert abs(d1["answer_dist_kl"] - kl) < 1e-4


def test_gradient_finite():
    loss, log_h, _ = _toy(0.3)
    loss.backward()
    for lh in log_h:
        assert torch.isfinite(lh.grad).all()
