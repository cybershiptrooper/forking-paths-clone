"""Early stopping in NodewiseSubnetworkProbingSDPA (off by default).

Runs ``discover`` on the tiny Qwen3 fixture of test_snp_hc_batched.py with
the task step replaced by a scripted loss sequence, and checks which step
training stops at, which step's log_alpha is saved, and that nothing
changes when early stopping is off.
"""
import pytest
import torch

from tests.test_snp_hc_batched import _fixture_inputs, _make_algo, _tiny_qwen3
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
)


def _run(losses, steps=40, **extra):
    model = _tiny_qwen3()
    algo = _make_algo(NodewiseSubnetworkProbingSDPA, model, K=1, **extra)
    algo.num_training_steps = steps
    algo.l0_lambda_schedule = True  # warmup 0.25 + ramp 0.5 of the defaults
    algo.l0_warmup_frac, algo.l0_ramp_frac = 0.25, 0.5
    calls = {"n": 0}

    def scripted(*a, **k):
        v = losses(calls["n"])
        calls["n"] += 1
        return v

    algo._step_local = scripted
    inputs, sentences, conts, pm, _ = _fixture_inputs()
    nm = algo.discover(
        input_ids=inputs, sentences=sentences, continuations=conts,
        mask_mode="prefix", num_prefix_sentences=len(sentences),
        branch_rewards=None, position_mask_overrides=[pm],
        num_frozen_prompt_sentences=0,
    )
    return nm, calls["n"]


def test_off_by_default_runs_every_step():
    nm, n = _run(lambda i: 1.0)
    assert n == 40
    assert nm.metadata["early_stop_step"] is None
    assert nm.metadata["early_stopping_patience"] is None


def test_stops_after_patience_on_a_flat_loss():
    # full penalty from ceil(0.75 * 39) = 30; window 3 -> first mean at 32,
    # which is the best; no improvement after it -> stop at 32 + 5 = 37.
    nm, n = _run(lambda i: 1.0, early_stopping_patience=5,
                 early_stopping_window=3)
    md = nm.metadata
    assert md["early_stopping_start_step"] == 30
    assert md["early_stopping_best_step"] == 32
    assert md["early_stop_step"] == 37
    assert n == 38  # steps 0..37


def test_improvements_before_the_penalty_is_full_are_ignored():
    # Loss falls until step 20 then stays flat: steps before 30 do not count.
    nm, _ = _run(lambda i: max(0.0, 1.0 - 0.05 * i), early_stopping_patience=5,
                 early_stopping_window=3)
    assert nm.metadata["early_stopping_best_step"] == 32


def test_saved_mask_is_the_best_step_snapshot():
    # Loss improves at every eligible step until 34, then worsens; the saved
    # scores must equal log_alpha as it was at step 34, not at the stop.
    snaps = {}
    orig = NodewiseSubnetworkProbingSDPA._current_sparsity

    def losses(i):
        return 1.0 - 0.1 * min(i, 34) + (0.5 if i > 34 else 0.0)

    model = _tiny_qwen3()
    algo = _make_algo(NodewiseSubnetworkProbingSDPA, model, K=1,
                      early_stopping_patience=3, early_stopping_window=1)
    algo.num_training_steps = 40
    algo.l0_lambda_schedule = True
    algo.l0_warmup_frac, algo.l0_ramp_frac = 0.25, 0.5
    algo.log_every = 1
    step_counter = {"n": 0}

    def scripted(*a, **k):
        v = losses(step_counter["n"])
        step_counter["n"] += 1
        return v

    def record(self, log_alpha, *a, **k):
        la = log_alpha if isinstance(log_alpha, torch.Tensor) else next(iter(log_alpha.values()))
        snaps[step_counter["n"] - 1] = la.detach().clone()
        return orig(self, log_alpha, *a, **k)

    algo._step_local = scripted
    NodewiseSubnetworkProbingSDPA._current_sparsity = record
    try:
        inputs, sentences, conts, pm, _ = _fixture_inputs()
        nm = algo.discover(
            input_ids=inputs, sentences=sentences, continuations=conts,
            mask_mode="prefix", num_prefix_sentences=len(sentences),
            branch_rewards=None, position_mask_overrides=[pm],
            num_frozen_prompt_sentences=0,
        )
    finally:
        NodewiseSubnetworkProbingSDPA._current_sparsity = orig
    md = nm.metadata
    assert md["early_stopping_best_step"] == 34
    assert md["early_stop_step"] == 37
    saved = torch.tensor(nm.scores)
    at_best = snaps[34].reshape(saved.shape)
    torch.testing.assert_close(saved, at_best.float())
    assert not torch.allclose(saved, snaps[37].reshape(saved.shape).float())
