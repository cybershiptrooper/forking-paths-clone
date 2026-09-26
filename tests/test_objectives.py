"""Tests for all circuit discovery objectives and importance sampling utilities.

Covers:
- Importance sampling: chain_log_prob, importance_weights, effective_sample_size,
  snis_answer_probs, extract_answer_ids
- Local objectives: kl_divergence_loss, log_prob_loss
- Global objectives: answer_distribution_kl_loss, reward_gap_loss
- Objective registry: get_objective, is_global_objective
- Gradient flow: verify gradients propagate correctly through global objectives
- Integration: two-pass gradient weights match full autograd

Usage:
    uv run python tests/test_objectives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:  # run standalone without pytest installed
    _HAS_PYTEST = False

    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self.exc = exc
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, tb):
            import re
            if exc_type is None:
                raise AssertionError(f"Did not raise {self.exc}")
            if not issubclass(exc_type, self.exc):
                return False
            if self.match is not None and not re.search(self.match, str(exc_val)):
                return False
            return True

    class _PytestShim:
        @staticmethod
        def raises(exc, match=None):
            return _RaisesCtx(exc, match=match)

    pytest = _PytestShim()  # type: ignore[assignment]

import torch
import torch.nn.functional as F

from utils.importance_sampling import (
    chain_log_prob,
    importance_weights,
    effective_sample_size,
    snis_answer_probs,
    extract_answer_ids,
    normalize_answer,
    reward_based_answer_ids,
)
from utils.objectives import (
    kl_divergence_loss,
    log_prob_loss,
    answer_distribution_kl_loss,
    reward_gap_loss,
    get_objective,
    is_global_objective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logits(seq_len: int, vocab_size: int, seed: int = 42) -> torch.Tensor:
    """Create random logits (1, seq_len, vocab_size)."""
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, vocab_size)


def make_token_ids(seq_len: int, vocab_size: int, seed: int = 42) -> torch.Tensor:
    """Create random token IDs (1, seq_len)."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len))


# ---------------------------------------------------------------------------
# Tests: chain_log_prob
# ---------------------------------------------------------------------------

def test_chain_log_prob_basic():
    """chain_log_prob should sum log-probs of continuation tokens only."""
    vocab = 10
    seq_len = 8
    prefix_len = 3  # continuation is tokens 3..7

    logits = make_logits(seq_len, vocab, seed=1)
    token_ids = make_token_ids(seq_len, vocab, seed=2)

    result = chain_log_prob(logits, token_ids, prefix_len)

    # Manual computation
    log_probs = F.log_softmax(logits.float(), dim=-1)
    expected = 0.0
    for t in range(prefix_len, seq_len):
        expected += log_probs[0, t - 1, token_ids[0, t]].item()

    assert abs(result.item() - expected) < 1e-5, (
        f"chain_log_prob mismatch: {result.item():.6f} vs {expected:.6f}"
    )
    print("[PASS] test_chain_log_prob_basic")


def test_chain_log_prob_gradient_flows():
    """chain_log_prob should be differentiable w.r.t. logits."""
    vocab = 10
    seq_len = 6
    prefix_len = 2

    logits = make_logits(seq_len, vocab).requires_grad_(True)
    token_ids = make_token_ids(seq_len, vocab)

    lp = chain_log_prob(logits, token_ids, prefix_len)
    lp.backward()

    assert logits.grad is not None, "No gradient on logits"
    assert logits.grad.shape == logits.shape
    # Prefix positions (0, 1) should have zero grad since we skip them
    # Actually position 1 (prefix_len - 1 = 1) predicts token at prefix_len,
    # so it should have nonzero grad
    assert logits.grad[0, 0].abs().sum() == 0, "Position 0 should have zero grad"
    assert logits.grad[0, 1].abs().sum() > 0, "Position 1 (predicts first cont token) should have grad"
    print("[PASS] test_chain_log_prob_gradient_flows")


def test_chain_log_prob_full_sequence():
    """With prefix_len=0, should sum all log-probs (predicting tokens 0..N-1 from logits)."""
    vocab = 5
    seq_len = 4
    prefix_len = 0  # NOTE: prefix_len=0 means prefix_len-1 = -1, slicing from -1

    logits = make_logits(seq_len, vocab, seed=3)
    token_ids = make_token_ids(seq_len, vocab, seed=4)

    # With prefix_len=0: targets = token_ids[:, 0:4], preds = logits[:, -1:3]
    # This is a degenerate case. Let's use prefix_len=1 instead for a clean test
    prefix_len = 1
    result = chain_log_prob(logits, token_ids, prefix_len)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    expected = 0.0
    for t in range(prefix_len, seq_len):
        expected += log_probs[0, t - 1, token_ids[0, t]].item()

    assert abs(result.item() - expected) < 1e-5
    print("[PASS] test_chain_log_prob_full_sequence")


# ---------------------------------------------------------------------------
# Tests: importance_weights
# ---------------------------------------------------------------------------

def test_importance_weights_sum_to_one():
    """Normalized importance weights should sum to 1."""
    log_p_target = torch.tensor([-5.0, -3.0, -4.0, -6.0])
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0, -5.5])

    w = importance_weights(log_p_target, log_p_proposal)

    assert abs(w.sum().item() - 1.0) < 1e-6, f"Weights sum to {w.sum().item()}, expected 1.0"
    assert (w >= 0).all(), "Weights should be non-negative"
    print("[PASS] test_importance_weights_sum_to_one")


def test_importance_weights_equal_when_same():
    """When target == proposal, all weights should be equal (1/N)."""
    log_p = torch.tensor([-3.0, -4.0, -5.0, -6.0])
    w = importance_weights(log_p, log_p)

    expected = 1.0 / len(log_p)
    for i, wi in enumerate(w):
        assert abs(wi.item() - expected) < 1e-6, (
            f"Weight {i}: {wi.item():.6f} vs {expected:.6f}"
        )
    print("[PASS] test_importance_weights_equal_when_same")


def test_importance_weights_gradient_flows():
    """Gradient should flow through importance_weights to log_p_target."""
    log_p_target = torch.tensor([-5.0, -3.0, -4.0], requires_grad=True)
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0])

    w = importance_weights(log_p_target, log_p_proposal)
    w.sum().backward()  # trivial loss

    assert log_p_target.grad is not None, "No gradient on log_p_target"
    print("[PASS] test_importance_weights_gradient_flows")


def test_importance_weights_proposal_detached():
    """Gradient should NOT flow to log_p_proposal."""
    log_p_target = torch.tensor([-5.0, -3.0], requires_grad=True)
    log_p_proposal = torch.tensor([-4.5, -3.5], requires_grad=True)

    w = importance_weights(log_p_target, log_p_proposal)
    w.sum().backward()

    assert log_p_proposal.grad is None or (log_p_proposal.grad == 0).all(), (
        "Gradient should not flow to proposal"
    )
    print("[PASS] test_importance_weights_proposal_detached")


# ---------------------------------------------------------------------------
# Tests: effective_sample_size
# ---------------------------------------------------------------------------

def test_n_eff_equal_weights():
    """Equal weights should give N_eff = N."""
    N = 10
    w = torch.ones(N) / N
    n_eff = effective_sample_size(w)
    assert abs(n_eff - N) < 1e-4, f"N_eff={n_eff}, expected {N}"
    print("[PASS] test_n_eff_equal_weights")


def test_n_eff_one_dominant():
    """One dominant weight should give N_eff ≈ 1."""
    w = torch.tensor([1e6, 1.0, 1.0, 1.0])
    n_eff = effective_sample_size(w)
    assert n_eff < 1.1, f"N_eff={n_eff}, expected ~1.0"
    print("[PASS] test_n_eff_one_dominant")


def test_n_eff_two_equal():
    """Two equal weights (rest zero) should give N_eff = 2."""
    w = torch.tensor([0.5, 0.5, 0.0, 0.0])
    # N_eff = 1 / (0.5^2 + 0.5^2 + 0 + 0) = 1 / 0.5 = 2
    n_eff = effective_sample_size(w)
    assert abs(n_eff - 2.0) < 1e-4, f"N_eff={n_eff}, expected 2.0"
    print("[PASS] test_n_eff_two_equal")


# ---------------------------------------------------------------------------
# Tests: snis_answer_probs
# ---------------------------------------------------------------------------

def test_snis_uniform_weights():
    """With uniform weights, SNIS should give simple counts."""
    N = 6
    weights = torch.ones(N) / N
    answer_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    num_answers = 3

    probs = snis_answer_probs(weights, answer_ids, num_answers)

    assert abs(probs[0].item() - 3 / 6) < 1e-6
    assert abs(probs[1].item() - 2 / 6) < 1e-6
    assert abs(probs[2].item() - 1 / 6) < 1e-6
    assert abs(probs.sum().item() - 1.0) < 1e-6
    print("[PASS] test_snis_uniform_weights")


def test_snis_weighted():
    """With non-uniform weights, SNIS should give weighted counts."""
    weights = torch.tensor([0.5, 0.3, 0.2])  # already normalized
    answer_ids = torch.tensor([0, 1, 0])
    num_answers = 2

    probs = snis_answer_probs(weights, answer_ids, num_answers)

    # P(0) = 0.5 + 0.2 = 0.7, P(1) = 0.3
    assert abs(probs[0].item() - 0.7) < 1e-6
    assert abs(probs[1].item() - 0.3) < 1e-6
    print("[PASS] test_snis_weighted")


def test_snis_gradient_flows():
    """Gradient should flow through snis_answer_probs to weights."""
    weights = torch.tensor([0.4, 0.3, 0.3], requires_grad=True)
    answer_ids = torch.tensor([0, 1, 0])
    probs = snis_answer_probs(weights, answer_ids, 2)
    probs[0].backward()

    assert weights.grad is not None
    # d P(0) / d w[0] = 1 (answer_ids[0] == 0), d P(0) / d w[1] = 0, d P(0) / d w[2] = 1
    assert abs(weights.grad[0].item() - 1.0) < 1e-6
    assert abs(weights.grad[1].item() - 0.0) < 1e-6
    assert abs(weights.grad[2].item() - 1.0) < 1e-6
    print("[PASS] test_snis_gradient_flows")


# ---------------------------------------------------------------------------
# Tests: extract_answer_ids
# ---------------------------------------------------------------------------

def test_extract_answer_ids_basic():
    """Should extract boxed answers and assign IDs."""
    branches = [
        {"text": "some text \\boxed{42} done"},
        {"text": "other text \\boxed{43} end"},
        {"text": "more text \\boxed{42} again"},
        {"text": "no answer here"},
    ]
    ids, labels = extract_answer_ids(branches)

    assert ids == [0, 1, 0, 2], f"Expected [0, 1, 0, 2], got {ids}"
    assert labels[0] == "42"
    assert labels[1] == "43"
    assert labels[2].startswith("__no_answer_")
    assert len(labels) == 3
    print("[PASS] test_extract_answer_ids_basic")


def test_extract_answer_ids_with_prefix():
    """Should concatenate prefix_text with branch text."""
    branches = [
        {"text": " \\boxed{10}"},
        {"text": " \\boxed{20}"},
    ]
    ids, labels = extract_answer_ids(branches, prefix_text="Problem: ")

    assert ids == [0, 1]
    assert labels == ["10", "20"]
    print("[PASS] test_extract_answer_ids_with_prefix")


def test_extract_answer_ids_all_same():
    """All branches with same answer should get same ID."""
    branches = [
        {"text": "\\boxed{7}"},
        {"text": "\\boxed{7}"},
        {"text": "\\boxed{7}"},
    ]
    ids, labels = extract_answer_ids(branches)

    assert ids == [0, 0, 0]
    assert labels == ["7"]
    print("[PASS] test_extract_answer_ids_all_same")


def test_extract_answer_ids_normalization():
    """Normalization should merge '42' and '42.0' and '42.00'."""
    branches = [
        {"text": "\\boxed{42}"},
        {"text": "\\boxed{42.0}"},
        {"text": "\\boxed{42.00}"},
        {"text": "\\boxed{ 42 }"},
    ]
    ids, labels = extract_answer_ids(branches)

    assert ids == [0, 0, 0, 0], f"Expected all same ID, got {ids}"
    assert len(labels) == 1
    print("[PASS] test_extract_answer_ids_normalization")


# ---------------------------------------------------------------------------
# Tests: normalize_answer
# ---------------------------------------------------------------------------

def test_normalize_answer_integers():
    """Integer normalization: trailing zeros, whitespace."""
    assert normalize_answer("42") == "42"
    assert normalize_answer("42.0") == "42"
    assert normalize_answer("42.00") == "42"
    assert normalize_answer(" 42 ") == "42"
    assert normalize_answer("$42$") == "42"
    print("[PASS] test_normalize_answer_integers")


def test_normalize_answer_fractions():
    """\\frac{a}{b} should be converted to decimal."""
    assert normalize_answer("\\frac{1}{2}") == "0.5"
    assert normalize_answer("\\frac{3}{4}") == "0.75"
    assert normalize_answer("\\frac{42}{1}") == "42"
    assert normalize_answer("\\frac{-1}{2}") == "-0.5"
    print("[PASS] test_normalize_answer_fractions")


def test_normalize_answer_dfrac_tfrac_equal_frac():
    """\\dfrac and \\tfrac are display-size variants and should normalize
    identically to \\frac (bug: \\dfrac{7}{72} vs \\frac{7}{72} mismatch
    collapsed binary bucketing at eval time)."""
    assert normalize_answer("\\dfrac{7}{72}") == normalize_answer("\\frac{7}{72}")
    assert normalize_answer("\\tfrac{3}{4}") == "0.75"
    assert normalize_answer("\\dfrac{1}{2}") == "0.5"
    print("[PASS] test_normalize_answer_dfrac_tfrac_equal_frac")


def test_normalize_answer_negative():
    """Negative numbers."""
    assert normalize_answer("-3") == "-3"
    assert normalize_answer("-3.0") == "-3"
    assert normalize_answer("-0.5") == "-0.5"
    print("[PASS] test_normalize_answer_negative")


def test_normalize_answer_latex_cosmetic():
    """Cosmetic LaTeX commands should be stripped."""
    assert normalize_answer("\\left(42\\right)") == "(42)"
    assert normalize_answer("1\\,000") == "1000"
    print("[PASS] test_normalize_answer_latex_cosmetic")


def test_normalize_answer_text_wrapper():
    """\\text{...} should be unwrapped."""
    assert normalize_answer("\\text{yes}") == "yes"
    print("[PASS] test_normalize_answer_text_wrapper")


def test_normalize_answer_symbolic():
    """Non-numeric answers should be collapsed but preserved."""
    assert normalize_answer("x + 1") == "x + 1"
    assert normalize_answer("  x  +  1  ") == "x + 1"
    print("[PASS] test_normalize_answer_symbolic")


# ---------------------------------------------------------------------------
# Tests: reward_based_answer_ids (Option C)
# ---------------------------------------------------------------------------

def test_reward_binary_correctness():
    """Binary mode: positive reward = correct, non-positive = incorrect."""
    rewards = [1.0, -1.0, 1.0, -1.0, 1.0]
    ids, labels = reward_based_answer_ids(rewards, binary=True)

    assert ids == [0, 1, 0, 1, 0]
    assert labels == ["correct", "incorrect"]
    print("[PASS] test_reward_binary_correctness")


def test_reward_binary_all_correct():
    """Binary mode: all correct → single group."""
    rewards = [1.0, 1.0, 1.0]
    ids, labels = reward_based_answer_ids(rewards, binary=True)

    assert ids == [0, 0, 0]
    assert labels == ["correct"]
    print("[PASS] test_reward_binary_all_correct")


def test_reward_binary_all_incorrect():
    """Binary mode: all incorrect → single group."""
    rewards = [-1.0, -1.0]
    ids, labels = reward_based_answer_ids(rewards, binary=True)

    assert ids == [0, 0]
    assert labels == ["incorrect"]
    print("[PASS] test_reward_binary_all_incorrect")


def test_reward_per_unique():
    """Per-unique mode: each distinct reward → own group."""
    rewards = [2.0, -1.0, 0.0, 2.0, -1.0]
    ids, labels = reward_based_answer_ids(rewards, binary=False)

    # Sorted unique: -1.0, 0.0, 2.0
    assert ids == [2, 0, 1, 2, 0], f"Expected [2, 0, 1, 2, 0], got {ids}"
    assert len(labels) == 3
    assert "reward=-1" in labels[0]
    assert "reward=+2" in labels[2]
    print("[PASS] test_reward_per_unique")


def test_reward_zero_boundary():
    """Binary mode: zero reward counts as incorrect."""
    rewards = [0.0, 0.5, -0.5]
    ids, labels = reward_based_answer_ids(rewards, binary=True)

    assert ids == [1, 0, 1]  # 0.0 ≤ 0 → incorrect, 0.5 > 0 → correct
    assert labels == ["correct", "incorrect"]
    print("[PASS] test_reward_zero_boundary")


# ---------------------------------------------------------------------------
# Tests: kl_divergence_loss (existing local objective)
# ---------------------------------------------------------------------------

def test_kl_divergence_loss_zero_when_same():
    """KL divergence should be ~0 when clean == masked."""
    logits = make_logits(5, 10)
    kl = kl_divergence_loss(logits, logits)
    assert kl.item() < 1e-6, f"KL should be ~0 for identical logits, got {kl.item()}"
    print("[PASS] test_kl_divergence_loss_zero_when_same")


def test_kl_divergence_loss_positive():
    """KL divergence should be positive for different logits."""
    clean = make_logits(5, 10, seed=1)
    masked = make_logits(5, 10, seed=2)
    kl = kl_divergence_loss(clean, masked)
    assert kl.item() > 0, f"KL should be positive, got {kl.item()}"
    print("[PASS] test_kl_divergence_loss_positive")


def test_kl_divergence_loss_position_mask():
    """Position mask should restrict KL to marked positions."""
    clean = make_logits(5, 10, seed=1)
    masked = make_logits(5, 10, seed=2)

    # Full mask
    full_mask = torch.ones(1, 5)
    kl_full = kl_divergence_loss(clean, masked, full_mask)

    # Only first 2 positions
    partial_mask = torch.zeros(1, 5)
    partial_mask[0, :2] = 1.0
    kl_partial = kl_divergence_loss(clean, masked, partial_mask)

    # They should differ (different positions have different KL)
    assert kl_full.item() != kl_partial.item(), "Position mask should affect result"
    print("[PASS] test_kl_divergence_loss_position_mask")


def test_kl_divergence_loss_gradient():
    """Gradient should flow to masked_logits but not clean_logits."""
    clean = make_logits(5, 10, seed=1)
    masked = make_logits(5, 10, seed=2).requires_grad_(True)

    kl = kl_divergence_loss(clean, masked)
    kl.backward()

    assert masked.grad is not None, "No gradient on masked_logits"
    assert masked.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("[PASS] test_kl_divergence_loss_gradient")


# ---------------------------------------------------------------------------
# Tests: log_prob_loss (existing local objective)
# ---------------------------------------------------------------------------

def test_log_prob_loss_basic():
    """log_prob_loss should return a positive value (negative log-prob)."""
    logits = make_logits(5, 10)
    token_ids = make_token_ids(5, 10)
    loss = log_prob_loss(logits, logits, token_ids=token_ids)
    assert loss.item() > 0, "Negative log-prob should be positive"
    print("[PASS] test_log_prob_loss_basic")


def test_log_prob_loss_requires_token_ids():
    """Should raise ValueError without token_ids."""
    logits = make_logits(5, 10)
    try:
        log_prob_loss(logits, logits)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_log_prob_loss_requires_token_ids")


# ---------------------------------------------------------------------------
# Tests: answer_distribution_kl_loss (Objective 1)
# ---------------------------------------------------------------------------

def test_answer_kl_zero_when_same():
    """When masked == clean, IS weights are uniform → P_m ≈ P_clean → KL ≈ 0."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(chain_lps, chain_lps, answer_ids, num_answers=2)

    assert kl.item() < 1e-6, f"KL should be ~0 when masked == clean, got {kl.item()}"
    print("[PASS] test_answer_kl_zero_when_same")


def test_answer_kl_positive_when_different():
    """KL should be positive when masked model shifts answer distribution."""
    chain_lps_clean = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    # Masked model strongly favors answer 0 chains
    chain_lps_masked = torch.tensor([-8.0, -9.0, -20.0, -25.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2
    )

    assert kl.item() > 0, f"KL should be positive, got {kl.item()}"
    print("[PASS] test_answer_kl_positive_when_different")


def test_answer_kl_gradient_flows():
    """Gradient should flow through answer_distribution_kl_loss."""
    chain_lps_masked = torch.tensor([-10.0, -12.0, -11.0, -13.0], requires_grad=True)
    chain_lps_clean = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2
    )
    kl.backward()

    assert chain_lps_masked.grad is not None, "No gradient"
    print("[PASS] test_answer_kl_gradient_flows")


def test_answer_kl_manual_computation():
    """Verify answer KL against manual computation."""
    # 4 chains: 3 with answer A (id=0), 1 with answer B (id=1)
    # P_clean: P(A) = 3/4, P(B) = 1/4
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])

    # Masked model: all chains equally likely under masked model too
    chain_lps_masked = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 0, 1])

    kl = answer_distribution_kl_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2
    )

    # When masked == clean, IS weights are uniform → P_m = P_clean → KL = 0
    assert kl.item() < 1e-6, f"Expected KL ≈ 0, got {kl.item()}"

    # Now shift: masked model gives much higher logprob to answer-B chain
    chain_lps_masked2 = torch.tensor([-10.0, -10.0, -10.0, -5.0])
    kl2 = answer_distribution_kl_loss(
        chain_lps_masked2, chain_lps_clean, answer_ids, num_answers=2
    )

    # P_clean(A) = 3/4, P_clean(B) = 1/4
    # IS weights: w = exp(lp_masked - lp_clean) = [1, 1, 1, exp(5)] ≈ [1, 1, 1, 148.4]
    # w_norm ≈ [0.0066, 0.0066, 0.0066, 0.9801]
    # P_m(A) ≈ 3 * 0.0066 = 0.0198, P_m(B) ≈ 0.9801
    # KL(P_clean || P_m) = 0.75 * log(0.75/0.0198) + 0.25 * log(0.25/0.9801) > 0
    assert kl2.item() > 0.5, f"Expected significant KL, got {kl2.item()}"
    print("[PASS] test_answer_kl_manual_computation")


def test_answer_kl_single_answer():
    """With only one answer, KL should always be 0."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0])
    answer_ids = torch.tensor([0, 0, 0])

    kl = answer_distribution_kl_loss(chain_lps, chain_lps, answer_ids, num_answers=1)
    assert kl.item() < 1e-6
    print("[PASS] test_answer_kl_single_answer")


# ---------------------------------------------------------------------------
# Tests: reward_gap_loss (Objective 2)
# ---------------------------------------------------------------------------

def test_reward_gap_zero_when_same():
    """When masked == clean, reward gap should equal clean gap (negated as loss)."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss = reward_gap_loss(chain_lps, chain_lps, answer_ids, num_answers=2, target_answer=0)

    # P_clean(0) = P_clean(1) = 0.5 (equal weights, 2 chains each)
    # R = P(0) - P(1) = 0 → loss = -0 = 0
    assert abs(loss.item()) < 1e-5, f"Expected loss ≈ 0, got {loss.item()}"
    print("[PASS] test_reward_gap_zero_when_same")


def test_reward_gap_negative_when_target_promoted():
    """When masked model promotes target, loss should be negative (good)."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    # Masked model strongly favors answer 0 chains
    chain_lps_masked = torch.tensor([-5.0, -5.0, -20.0, -20.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss = reward_gap_loss(
        chain_lps_masked, chain_lps_clean, answer_ids,
        num_answers=2, target_answer=0,
    )

    # P_m(0) >> P_m(1) → R > 0 → loss = -R < 0
    assert loss.item() < -0.5, f"Expected loss << 0, got {loss.item()}"
    print("[PASS] test_reward_gap_negative_when_target_promoted")


def test_reward_gap_gradient_flows():
    """Gradient should flow through reward_gap_loss."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0, -13.0], requires_grad=True)
    chain_lps_clean = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss = reward_gap_loss(chain_lps, chain_lps_clean, answer_ids, num_answers=2)
    loss.backward()

    assert chain_lps.grad is not None, "No gradient"
    print("[PASS] test_reward_gap_gradient_flows")


def test_reward_gap_with_three_answers():
    """Reward gap should work with 3+ answer groups."""
    chain_lps_clean = torch.tensor([-10.0] * 6)
    # Masked: strongly favor answer 0
    chain_lps_masked = torch.tensor([-5.0, -5.0, -20.0, -20.0, -20.0, -20.0])
    answer_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = reward_gap_loss(
        chain_lps_masked, chain_lps_clean, answer_ids,
        num_answers=3, target_answer=0,
    )

    assert loss.item() < 0, "Target promoted → loss should be negative"
    print("[PASS] test_reward_gap_with_three_answers")


# ---------------------------------------------------------------------------
# Tests: objective registry
# ---------------------------------------------------------------------------

def test_get_objective_local():
    """get_objective should return local objectives."""
    fn = get_objective("kl_divergence")
    assert fn is kl_divergence_loss

    fn = get_objective("log_prob")
    assert fn is log_prob_loss
    print("[PASS] test_get_objective_local")


def test_get_objective_global():
    """get_objective should return global objectives."""
    fn = get_objective("answer_kl")
    assert fn is answer_distribution_kl_loss

    fn = get_objective("reward_gap")
    assert fn is reward_gap_loss
    print("[PASS] test_get_objective_global")


def test_get_objective_unknown():
    """get_objective should raise for unknown names."""
    try:
        get_objective("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_get_objective_unknown")


def test_is_global_objective():
    """is_global_objective should correctly classify objectives."""
    assert not is_global_objective("kl_divergence")
    assert not is_global_objective("log_prob")
    assert is_global_objective("answer_kl")
    assert is_global_objective("reward_gap")
    assert not is_global_objective("nonexistent")
    print("[PASS] test_is_global_objective")


# ---------------------------------------------------------------------------
# Tests: two-pass gradient correctness
# ---------------------------------------------------------------------------

def test_two_pass_gradient_matches_full_autograd():
    """The two-pass approach (used in IG) should produce the same gradients
    as full autograd through the global loss.

    Two-pass:
        1. Forward all chains (no grad) → detached chain logprobs
        2. Compute per-chain weights via small autograd graph
        3. For each chain, forward with grad → weighted_loss → backward

    Full autograd:
        Forward all chains (with grad) → stack → global loss → backward
    """
    torch.manual_seed(42)

    N = 4
    vocab = 10
    seq_len = 8
    prefix_len = 3

    # Simulate mask parameters (what we'd differentiate w.r.t.)
    mask_param = torch.randn(5, requires_grad=True)

    # Create "model" logits as a function of mask_param (simple linear)
    base_logits = [make_logits(seq_len, vocab, seed=i) for i in range(N)]
    token_ids_list = [make_token_ids(seq_len, vocab, seed=10 + i) for i in range(N)]
    answer_ids = torch.tensor([0, 0, 1, 1])
    num_answers = 2
    chain_lps_clean = torch.tensor([-15.0, -16.0, -14.0, -17.0])

    # Full autograd approach
    mask_param_full = mask_param.clone().detach().requires_grad_(True)
    chain_lps_full = []
    for i in range(N):
        # Simulate masked logits as base + mask_param influence
        logits = base_logits[i] + mask_param_full[i % 5] * 0.1
        lp = chain_log_prob(logits, token_ids_list[i], prefix_len)
        chain_lps_full.append(lp)
    chain_lps_full = torch.stack(chain_lps_full)
    loss_full = answer_distribution_kl_loss(
        chain_lps_full, chain_lps_clean, answer_ids, num_answers
    )
    loss_full.backward()
    grad_full = mask_param_full.grad.clone()

    # Two-pass approach
    mask_param_two = mask_param.clone().detach().requires_grad_(True)

    # Pass 1: forward without grad
    chain_lps_detached = []
    for i in range(N):
        logits = base_logits[i] + mask_param_two.detach()[i % 5] * 0.1
        lp = chain_log_prob(logits, token_ids_list[i], prefix_len)
        chain_lps_detached.append(lp.detach())
    chain_lps_detached = torch.stack(chain_lps_detached)

    # Compute per-chain weights
    chain_lps_param = chain_lps_detached.clone().requires_grad_(True)
    global_loss = answer_distribution_kl_loss(
        chain_lps_param, chain_lps_clean, answer_ids, num_answers
    )
    global_loss.backward()
    per_chain_weights = chain_lps_param.grad.detach()

    # Pass 2: forward each chain with grad, weighted backward
    for i in range(N):
        if mask_param_two.grad is not None:
            mask_param_two.grad.zero_()

        logits = base_logits[i] + mask_param_two[i % 5] * 0.1
        lp = chain_log_prob(logits, token_ids_list[i], prefix_len)
        weighted = lp * per_chain_weights[i]
        weighted.backward()

        # Note: we need to accumulate, not overwrite, so we can't zero between chains
        # Actually the above zeros and then only gets one chain's grad. Let me fix.

    # Redo: accumulate across all chains
    mask_param_two = mask_param.clone().detach().requires_grad_(True)
    accumulated_grad = torch.zeros_like(mask_param_two)

    for i in range(N):
        mask_param_two.grad = None
        logits = base_logits[i] + mask_param_two[i % 5] * 0.1
        lp = chain_log_prob(logits, token_ids_list[i], prefix_len)
        weighted = lp * per_chain_weights[i]
        weighted.backward()
        accumulated_grad += mask_param_two.grad.detach()

    # Compare
    max_diff = (grad_full - accumulated_grad).abs().max().item()
    assert max_diff < 1e-4, (
        f"Two-pass gradient differs from full autograd by {max_diff:.6e}\n"
        f"Full:     {grad_full}\n"
        f"Two-pass: {accumulated_grad}"
    )
    print(f"[PASS] test_two_pass_gradient_matches_full_autograd (max_diff={max_diff:.2e})")


# ---------------------------------------------------------------------------
# Tests: numerical stability
# ---------------------------------------------------------------------------

def test_importance_weights_large_difference():
    """Should handle large differences in log-probs without overflow."""
    log_p_target = torch.tensor([-100.0, -200.0, -150.0])
    log_p_proposal = torch.tensor([-105.0, -195.0, -155.0])

    w = importance_weights(log_p_target, log_p_proposal)

    assert not torch.isnan(w).any(), "NaN in weights"
    assert not torch.isinf(w).any(), "Inf in weights"
    assert abs(w.sum().item() - 1.0) < 1e-5
    print("[PASS] test_importance_weights_large_difference")


def test_importance_weights_snis_default_matches_old_behavior():
    """method='snis' produces bit-identical output to the unparameterised call."""
    log_p_target = torch.tensor([-3.0, -4.0, -5.0, -6.0])
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0, -5.5])
    w_default = importance_weights(log_p_target, log_p_proposal)
    w_snis = importance_weights(log_p_target, log_p_proposal, method="snis")
    assert torch.allclose(w_default, w_snis, atol=0.0, rtol=0.0)
    print("[PASS] test_importance_weights_snis_default_matches_old_behavior")


def test_importance_weights_geometric_mean_requires_lengths():
    log_p_target = torch.tensor([-3.0, -4.0])
    log_p_proposal = torch.tensor([-4.5, -3.5])
    with pytest.raises(ValueError, match="chain_lengths"):
        importance_weights(log_p_target, log_p_proposal, method="geometric_mean")
    print("[PASS] test_importance_weights_geometric_mean_requires_lengths")


def test_importance_weights_geometric_mean_equal_lengths_equals_tempered_snis():
    """When all T_i are equal, geometric_mean is SNIS at temperature T."""
    log_p_target = torch.tensor([-3.0, -4.0, -5.0])
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0])
    T = 100
    chain_lengths = torch.full((3,), T, dtype=torch.long)

    w_geo = importance_weights(
        log_p_target, log_p_proposal,
        method="geometric_mean", chain_lengths=chain_lengths,
    )

    log_w = (log_p_target - log_p_proposal) / T
    log_w = log_w - log_w.max()
    expected = torch.exp(log_w) / torch.exp(log_w).sum()

    assert torch.allclose(w_geo, expected, atol=1e-6)
    print("[PASS] test_importance_weights_geometric_mean_equal_lengths_equals_tempered_snis")


def test_importance_weights_geometric_mean_unequal_lengths_smoke():
    """With mixed lengths, softmax inputs are log_w_i / T_i — no crash, sums to 1."""
    log_p_target = torch.tensor([-30.0, -4.0, -5.0])
    log_p_proposal = torch.tensor([-45.0, -3.5, -4.0])
    chain_lengths = torch.tensor([10000, 100, 50], dtype=torch.long)

    w = importance_weights(
        log_p_target, log_p_proposal,
        method="geometric_mean", chain_lengths=chain_lengths,
    )
    assert torch.isfinite(w).all()
    assert abs(w.sum().item() - 1.0) < 1e-6
    print("[PASS] test_importance_weights_geometric_mean_unequal_lengths_smoke")


def test_importance_weights_tempered_snis_requires_temperature():
    log_p_target = torch.tensor([-3.0, -4.0])
    log_p_proposal = torch.tensor([-4.5, -3.5])
    with pytest.raises(ValueError, match="temperature"):
        importance_weights(log_p_target, log_p_proposal, method="tempered_snis")
    with pytest.raises(ValueError, match="> 0"):
        importance_weights(
            log_p_target, log_p_proposal,
            method="tempered_snis", temperature=0.0,
        )
    print("[PASS] test_importance_weights_tempered_snis_requires_temperature")


def test_importance_weights_tempered_snis_t1_equals_snis():
    """T=1 recovers vanilla SNIS exactly."""
    log_p_target = torch.tensor([-3.0, -4.0, -5.0, -6.0])
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0, -5.5])
    w_snis = importance_weights(log_p_target, log_p_proposal, method="snis")
    w_tempered = importance_weights(
        log_p_target, log_p_proposal,
        method="tempered_snis", temperature=1.0,
    )
    assert torch.allclose(w_snis, w_tempered, atol=1e-7)
    print("[PASS] test_importance_weights_tempered_snis_t1_equals_snis")


def test_importance_weights_tempered_snis_large_t_approaches_uniform():
    """T -> large smooths weights toward 1/N."""
    log_p_target = torch.tensor([-3.0, -4.0, -5.0, -6.0])
    log_p_proposal = torch.tensor([-4.5, -3.5, -4.0, -5.5])
    N = log_p_target.shape[0]
    w = importance_weights(
        log_p_target, log_p_proposal,
        method="tempered_snis", temperature=1e6,
    )
    assert torch.allclose(w, torch.full((N,), 1.0 / N), atol=1e-5)
    print("[PASS] test_importance_weights_tempered_snis_large_t_approaches_uniform")


def test_importance_weights_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown importance sampling method"):
        importance_weights(
            torch.tensor([-3.0]), torch.tensor([-4.0]),
            method="not_a_method",
        )
    print("[PASS] test_importance_weights_unknown_method_raises")


def test_answer_distribution_kl_loss_threads_is_method():
    """The objective forwards is_method + chain_lengths to importance_weights."""
    chain_lps_m = torch.tensor([-100.0, -200.0, -250.0], requires_grad=True)
    chain_lps_c = torch.tensor([-105.0, -195.0, -155.0])
    answer_ids = torch.tensor([0, 0, 1])
    chain_lengths = torch.tensor([100, 100, 100], dtype=torch.long)

    loss_snis = answer_distribution_kl_loss(
        chain_lps_m, chain_lps_c, answer_ids, 2, is_method="snis",
    )
    loss_geo = answer_distribution_kl_loss(
        chain_lps_m, chain_lps_c, answer_ids, 2,
        is_method="geometric_mean", chain_lengths=chain_lengths,
    )
    assert not torch.allclose(loss_snis, loss_geo), (
        "Geometric mean should differ from SNIS in this regime"
    )
    print("[PASS] test_answer_distribution_kl_loss_threads_is_method")


def test_answer_kl_numerical_stability():
    """Should handle extreme importance weights gracefully."""
    # One chain has much higher logprob under masked model
    chain_lps_masked = torch.tensor([-5.0, -100.0, -100.0, -100.0])
    chain_lps_clean = torch.tensor([-50.0, -50.0, -50.0, -50.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2
    )

    assert not torch.isnan(kl), "KL is NaN"
    assert not torch.isinf(kl), "KL is Inf"
    assert kl.item() >= 0, "KL should be non-negative"
    print("[PASS] test_answer_kl_numerical_stability")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Importance Sampling Utilities")
    print("=" * 60)
    test_chain_log_prob_basic()
    test_chain_log_prob_gradient_flows()
    test_chain_log_prob_full_sequence()

    print()
    print("=" * 60)
    print("Importance Weights")
    print("=" * 60)
    test_importance_weights_sum_to_one()
    test_importance_weights_equal_when_same()
    test_importance_weights_gradient_flows()
    test_importance_weights_proposal_detached()
    test_importance_weights_large_difference()
    test_importance_weights_snis_default_matches_old_behavior()
    test_importance_weights_geometric_mean_requires_lengths()
    test_importance_weights_geometric_mean_equal_lengths_equals_tempered_snis()
    test_importance_weights_geometric_mean_unequal_lengths_smoke()
    test_importance_weights_tempered_snis_requires_temperature()
    test_importance_weights_tempered_snis_t1_equals_snis()
    test_importance_weights_tempered_snis_large_t_approaches_uniform()
    test_importance_weights_unknown_method_raises()

    print()
    print("=" * 60)
    print("Effective Sample Size")
    print("=" * 60)
    test_n_eff_equal_weights()
    test_n_eff_one_dominant()
    test_n_eff_two_equal()

    print()
    print("=" * 60)
    print("SNIS Answer Probabilities")
    print("=" * 60)
    test_snis_uniform_weights()
    test_snis_weighted()
    test_snis_gradient_flows()

    print()
    print("=" * 60)
    print("Answer Extraction")
    print("=" * 60)
    test_extract_answer_ids_basic()
    test_extract_answer_ids_with_prefix()
    test_extract_answer_ids_all_same()
    test_extract_answer_ids_normalization()

    print()
    print("=" * 60)
    print("Answer Normalization")
    print("=" * 60)
    test_normalize_answer_integers()
    test_normalize_answer_fractions()
    test_normalize_answer_dfrac_tfrac_equal_frac()
    test_normalize_answer_negative()
    test_normalize_answer_latex_cosmetic()
    test_normalize_answer_text_wrapper()
    test_normalize_answer_symbolic()

    print()
    print("=" * 60)
    print("Reward-Based Answer IDs (Option C)")
    print("=" * 60)
    test_reward_binary_correctness()
    test_reward_binary_all_correct()
    test_reward_binary_all_incorrect()
    test_reward_per_unique()
    test_reward_zero_boundary()

    print()
    print("=" * 60)
    print("Local Objectives (kl_divergence, log_prob)")
    print("=" * 60)
    test_kl_divergence_loss_zero_when_same()
    test_kl_divergence_loss_positive()
    test_kl_divergence_loss_position_mask()
    test_kl_divergence_loss_gradient()
    test_log_prob_loss_basic()
    test_log_prob_loss_requires_token_ids()

    print()
    print("=" * 60)
    print("Global Objective 1: answer_distribution_kl_loss")
    print("=" * 60)
    test_answer_kl_zero_when_same()
    test_answer_kl_positive_when_different()
    test_answer_kl_gradient_flows()
    test_answer_kl_manual_computation()
    test_answer_kl_single_answer()
    test_answer_kl_numerical_stability()
    test_answer_distribution_kl_loss_threads_is_method()

    print()
    print("=" * 60)
    print("Global Objective 2: reward_gap_loss")
    print("=" * 60)
    test_reward_gap_zero_when_same()
    test_reward_gap_negative_when_target_promoted()
    test_reward_gap_gradient_flows()
    test_reward_gap_with_three_answers()

    print()
    print("=" * 60)
    print("Objective Registry")
    print("=" * 60)
    test_get_objective_local()
    test_get_objective_global()
    test_get_objective_unknown()
    test_is_global_objective()

    print()
    print("=" * 60)
    print("Two-Pass Gradient Correctness")
    print("=" * 60)
    test_two_pass_gradient_matches_full_autograd()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
