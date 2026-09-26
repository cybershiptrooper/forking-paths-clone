"""Unit tests for the 2026-08-20 SNP fixes and the DCM+PID additions.

Covers (CPU only, no model):
- hc_beta_anneal consistency: the L0 budget, sparsity metrics, and
  readout all accept and use the annealed temperature.
- Normalized hinge: bounded loss value, gradient scaling.
- Lambda schedule with l0_warmup_frac = 0 (pressure from step 0).
- Polyak EMA state round-trips through the checkpoint.
- The log-space PID controller and the DCM+PID ramp/snapshot logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _HC_BETA,
    _HC_GAMMA,
    _HC_ZETA,
    _hard_concrete_l0_probs,
    _hard_concrete_mean,
)
from utils.circuit_discovery.edits.nodewise_dcm_pid_sdpa import (
    NodewiseDCMPIDSDPA,
    _LogSpacePID,
)


def _make_snp(**over):
    """Build an SNP instance without a model (enough for the pure-math
    helpers under test)."""
    kw = dict(
        model=None,
        tokenizer=None,
        layers=[0],
        objective_fn=None,
        sparsity_loss_mode="target_size_relu",
        target_sparsity=0.5,
        num_training_steps=1000,
    )
    kw.update(over)
    obj = NodewiseSubnetworkProbingSDPA.__new__(NodewiseSubnetworkProbingSDPA)
    # Bypass CircuitDiscovery.__init__ (needs a model); set the attributes
    # the helpers under test read.
    obj.layers = kw["layers"]
    obj.sparsity_loss_mode = kw["sparsity_loss_mode"]
    obj.target_sparsity = kw["target_sparsity"]
    obj.l0_normalize_hinge = kw.get("l0_normalize_hinge", False)
    obj.l0_lambda = kw.get("l0_lambda", 100.0)
    obj.l0_lambda_schedule = kw.get("l0_lambda_schedule", True)
    obj.l0_warmup_frac = kw.get("l0_warmup_frac", 0.25)
    obj.l0_ramp_frac = kw.get("l0_ramp_frac", 0.50)
    obj.num_training_steps = kw["num_training_steps"]
    obj.hc_beta_anneal = kw.get("hc_beta_anneal", False)
    obj.hc_beta_start = kw.get("hc_beta_start", _HC_BETA)
    obj.hc_beta_end = kw.get("hc_beta_end", _HC_BETA / 2.0)
    obj.polyak_ema_log_alpha = kw.get("polyak_ema_log_alpha", 0.0)
    obj.checkpoint_path = None
    return obj


def test_l0_uses_supplied_beta():
    snp = _make_snp()
    la = torch.zeros(1, 4, 4)
    filt = torch.zeros(4, 4, dtype=torch.bool)  # everything learnable
    l0_default = snp._l0(la, "pair", filt, beta=_HC_BETA)
    l0_soft = snp._l0(la, "pair", filt, beta=2.0)
    # At log_alpha = 0, P(z>0) = sigmoid(-beta*log(-gamma/zeta)) grows with
    # beta (since log(-gamma/zeta) < 0), so the active count grows too.
    assert l0_soft.item() > l0_default.item()


def test_current_and_expected_sparsity_use_beta():
    snp = _make_snp()
    # log_alpha just below the beta=2/3 clamp point (~ -1.60): clamped to
    # zero at beta=2/3, NOT clamped at beta=2 (clamp point ~ -4.80).
    la = torch.full((1, 4, 4), -2.0)
    filt = torch.zeros(4, 4, dtype=torch.bool)
    assert snp._current_sparsity(la, "pair", filt, beta=_HC_BETA) == 1.0
    assert snp._current_sparsity(la, "pair", filt, beta=2.0) == 0.0
    exp_hard = snp._expected_sparsity(la, "pair", filt, beta=_HC_BETA)
    exp_soft = snp._expected_sparsity(la, "pair", filt, beta=2.0)
    assert exp_hard > exp_soft  # softer temperature keeps more mass active


def test_normalized_hinge_bounds_loss():
    la = torch.full((1, 10, 10), 4.0, requires_grad=True)
    filt = torch.zeros(10, 10, dtype=torch.bool)
    plain = _make_snp(target_sparsity=0.5)
    normed = _make_snp(target_sparsity=0.5, l0_normalize_hinge=True)

    h_plain = plain._l0(la, "pair", filt)
    h_norm = normed._l0(la, "pair", filt)
    # ~100 valid cells, nearly all active, budget 50 → plain hinge ≈ 48;
    # normalized ≈ 1 (excess / excess).
    assert h_plain.item() > 10.0
    assert h_norm.item() == pytest.approx(1.0, abs=0.05)

    g_plain = torch.autograd.grad(h_plain, la, retain_graph=False)[0]
    la2 = la.detach().clone().requires_grad_(True)
    h_norm2 = normed._l0(la2, "pair", filt)
    g_norm = torch.autograd.grad(h_norm2, la2)[0]
    excess = h_plain.item()
    ratio = g_plain.abs().sum().item() / max(g_norm.abs().sum().item(), 1e-12)
    assert ratio == pytest.approx(excess, rel=0.05)


def test_lambda_schedule_warmup_zero():
    snp = _make_snp(l0_warmup_frac=0.0, l0_ramp_frac=0.9, l0_lambda=100.0,
                    num_training_steps=1000)
    lam0 = snp._lambda_at_step(0)
    lam1 = snp._lambda_at_step(1)
    assert lam0 == 0.0  # ramp starts at 0 exactly at step 0...
    assert lam1 > 0.0   # ...and is strictly positive from the next step on
    assert snp._lambda_at_step(999) == pytest.approx(100.0, rel=1e-6)
    # canonical schedule for contrast: flat 0 through the warmup
    snp_c = _make_snp()
    assert snp_c._lambda_at_step(249) == 0.0
    assert snp_c._lambda_at_step(251) > 0.0


def test_checkpoint_roundtrips_ema(tmp_path):
    snp = _make_snp(polyak_ema_log_alpha=0.9)
    la = torch.zeros(1, 3, 3, requires_grad=True)
    ema = torch.full((1, 3, 3), 5.0)
    optim = torch.optim.Adam([la], lr=0.1)
    path = str(tmp_path / "ckpt.pt")
    snp._save_checkpoint(path, 7, la, "pair", optim, ema_log_alpha=ema)

    la2 = torch.ones(1, 3, 3, requires_grad=True)
    ema2 = torch.zeros(1, 3, 3)
    optim2 = torch.optim.Adam([la2], lr=0.1)
    step = snp._load_checkpoint(path, la2, "pair", optim2, "cpu",
                                ema_log_alpha=ema2)
    assert step == 7
    assert torch.allclose(la2, la.detach())
    assert torch.allclose(ema2, ema)


def test_checkpoint_without_ema_is_backward_compatible(tmp_path):
    snp = _make_snp()
    la = torch.zeros(1, 3, 3, requires_grad=True)
    optim = torch.optim.Adam([la], lr=0.1)
    path = str(tmp_path / "ckpt.pt")
    snp._save_checkpoint(path, 3, la, "pair", optim)
    la2 = torch.ones(1, 3, 3, requires_grad=True)
    optim2 = torch.optim.Adam([la2], lr=0.1)
    step = snp._load_checkpoint(path, la2, "pair", optim2, "cpu")
    assert step == 3


def test_pid_controller_direction():
    pid = _LogSpacePID(0.1, 0.001, 0.0, 1e-3)
    m0 = pid.mult
    pid.step(actual_rate=0.0, target_rate=1.0, count_error=50.0)
    assert pid.mult > m0  # behind the ramp → raise pressure
    pid2 = _LogSpacePID(0.1, 0.001, 0.0, 1.0)
    pid2.step(actual_rate=5.0, target_rate=0.0, count_error=-50.0)
    assert pid2.mult < 1.0  # ahead of the ramp → lower pressure


def test_dcm_pid_ramp_and_validation():
    obj = NodewiseDCMPIDSDPA.__new__(NodewiseDCMPIDSDPA)
    obj.num_training_steps = 1000
    obj.pid_ramp_end_frac = 0.9
    obj.pid_max_target_sparsity = 0.95
    assert obj._target_zero_frac(0) == 0.0
    mid = obj._target_zero_frac(450)
    assert 0.4 < mid < 0.55
    assert obj._target_zero_frac(999) == pytest.approx(0.95)

    with pytest.raises(ValueError):
        NodewiseDCMPIDSDPA.__init__(
            NodewiseDCMPIDSDPA.__new__(NodewiseDCMPIDSDPA),
            pid_max_target_sparsity=0.5,
            snapshot_sparsities=[0.9],
            model=None, tokenizer=None, layers=[0], objective_fn=None,
        )

def test_flip_cap_reverts_excess_crossings():
    import torch

    # 6 learnable cells in a (1, 3, 3) mask; 4 cross below 0.5 this step.
    learnable = torch.zeros(3, 3, dtype=torch.bool)
    learnable[0, 1] = learnable[0, 2] = learnable[1, 2] = True
    learnable[1, 0] = learnable[2, 0] = learnable[2, 1] = True
    pre_vals = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.55, 0.3])
    post = torch.tensor([0.05, 0.20, 0.10, 0.45, 0.60, 0.40])
    mask_param = torch.ones(1, 3, 3)
    mask_param[0][learnable] = post

    n_rev = NodewiseDCMPIDSDPA._apply_flip_cap(
        mask_param, learnable, pre_vals, flip_cap=2,
    )
    vals = mask_param[0][learnable]
    # 4 new crossings (cells 0-3; cell 5 was already below, cell 4 rose).
    # The 2 deepest (0.05, 0.10 -> cells 0 and 2) survive; cells 1 and 3
    # revert to their pre-step values.
    assert n_rev == 2
    assert vals[0] == pytest.approx(0.05)
    assert vals[2] == pytest.approx(0.10)
    assert vals[1] == pytest.approx(0.8)
    assert vals[3] == pytest.approx(0.6)
    # Untouched: the reopening (cell 4) and the already-off cell (5).
    assert vals[4] == pytest.approx(0.60)
    assert vals[5] == pytest.approx(0.40)

    # Under the cap: nothing reverted.
    mask_param2 = torch.ones(1, 3, 3)
    mask_param2[0][learnable] = post
    assert NodewiseDCMPIDSDPA._apply_flip_cap(
        mask_param2, learnable, pre_vals, flip_cap=10,
    ) == 0
    assert torch.allclose(mask_param2[0][learnable], post)


def test_dcm_new_option_validation():
    def _init(**over):
        kwargs = dict(
            model=None, tokenizer=None, layers=[0], objective_fn=None,
        )
        kwargs.update(over)
        return NodewiseDCMPIDSDPA.__init__(
            NodewiseDCMPIDSDPA.__new__(NodewiseDCMPIDSDPA), **kwargs
        )

    with pytest.raises(ValueError):
        _init(dcm_task_optimizer="rmsprop")
    with pytest.raises(ValueError):
        _init(dcm_max_flips_per_step=0)
    with pytest.raises(ValueError):
        _init(dcm_flip_cap_ramp_mult=0.0)
    # Valid combinations construct fine (dummy model for the base class).
    from types import SimpleNamespace

    dummy_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    _init(model=dummy_model, dcm_task_optimizer="sgd_norm",
          dcm_max_flips_per_step=10, dcm_flip_cap_ramp_mult=2.0)
