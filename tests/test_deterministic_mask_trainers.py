"""Unit tests for the tensor-level pieces of the deterministic mask trainers
(no model needed)."""
import torch

from utils.circuit_discovery.edits.nodewise_deterministic_mask_trainers import (
    n_keep_at_target, n_keep_on_ramp, topk_keep_mask, straight_through_mask,
)


def test_ramp_counts():
    assert n_keep_at_target(100, 0.3) == 70
    assert n_keep_on_ramp(100, 0.3, step=0, ramp_steps=500) == 100
    assert n_keep_on_ramp(100, 0.3, step=250, ramp_steps=500) == 85
    assert n_keep_on_ramp(100, 0.3, step=500, ramp_steps=500) == 70
    assert n_keep_on_ramp(100, 0.3, step=999, ramp_steps=500) == 70
    assert n_keep_on_ramp(100, 0.3, step=0, ramp_steps=0) == 70


def test_topk_keep_mask_respects_learnable_set():
    scores = torch.arange(16, dtype=torch.float32).view(4, 4)
    learnable = torch.zeros(4, 4, dtype=torch.bool)
    learnable[1:, :] = True                       # 12 learnable cells
    keep = topk_keep_mask(scores, learnable, n_keep=5)
    assert keep.sum() == 5
    assert not keep[0].any()                      # never keeps a non-learnable cell
    assert keep[3].all() and keep[2, 3]           # the 5 highest learnable scores


def test_straight_through_mask_value_and_gradient():
    torch.manual_seed(0)
    scores = torch.randn(3, 3, requires_grad=True)
    learnable = torch.ones(3, 3, dtype=torch.bool)
    learnable[0, 0] = False
    keep = topk_keep_mask(scores, learnable, n_keep=4)
    mask = straight_through_mask(scores, keep, learnable, removed_value=1e-4, surrogate="identity")
    # forward value: 1 on kept and non-learnable cells, removed_value elsewhere
    expected = torch.where(keep | ~learnable, torch.ones(3, 3), torch.full((3, 3), 1e-4))
    assert torch.allclose(mask.detach(), expected)
    # backward: identity surrogate passes the mask gradient to every learnable score
    upstream = torch.randn(3, 3)
    (mask * upstream).sum().backward()
    assert torch.allclose(scores.grad[learnable], upstream[learnable])
    assert scores.grad[0, 0] == 0.0
    # sigmoid surrogate scales the gradient by sigmoid'(score)
    scores2 = scores.detach().clone().requires_grad_(True)
    mask2 = straight_through_mask(scores2, keep, learnable, 1e-4, "sigmoid")
    (mask2 * upstream).sum().backward()
    sp = torch.sigmoid(scores2.detach()) * (1 - torch.sigmoid(scores2.detach()))
    assert torch.allclose(scores2.grad[learnable], (upstream * sp)[learnable])
