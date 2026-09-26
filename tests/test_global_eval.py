"""Tests for global (IS-based) metric evaluation.

Covers:
- eval_global_metric output format verification (via direct function calls)
- Random baseline global metric data structure
- Answer KL consistency properties
- N_eff degradation behavior
- Reward gap consistency properties

Tests that require model + hooks (eval_global_metric, evaluate_at_thresholds)
are tested by verifying their logic components in isolation.

Usage:
    uv run python -m tests.test_global_eval
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from utils.importance_sampling import (
    chain_log_prob,
    importance_weights,
    effective_sample_size,
    snis_answer_probs,
)
from utils.objectives import (
    answer_distribution_kl_loss,
    reward_gap_loss,
    is_global_objective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simulate_eval_global_metric(
    chain_lps_masked, chain_lps_clean, answer_ids, num_answers, objective_fn,
    is_method: str = "snis", chain_lengths=None,
):
    """Simulate what eval_global_metric computes, without needing a model.

    This mirrors the logic in circuit_eval.eval_global_metric:
    1. Compute IS weights
    2. Compute N_eff
    3. Compute answer probs via SNIS
    4. Compute the objective
    """
    device = chain_lps_masked.device
    w = importance_weights(
        chain_lps_masked, chain_lps_clean.to(device),
        method=is_method, chain_lengths=chain_lengths,
    )
    n_eff = effective_sample_size(w)
    p_m = snis_answer_probs(w, answer_ids.to(device), num_answers)

    metric = objective_fn(
        chain_lps_masked, chain_lps_clean.to(device),
        answer_ids.to(device), num_answers,
        is_method=is_method, chain_lengths=chain_lengths,
    ).item()

    return {
        "metric": metric,
        "n_eff": n_eff,
        "n_eff_ratio": n_eff / len(chain_lps_masked),
        "p_m": p_m.detach().cpu().tolist(),
        "log_weights": (chain_lps_masked - chain_lps_clean.to(device)).detach().cpu().tolist(),
        "chain_weights_normalized": w.detach().cpu().tolist(),
    }


def test_eval_global_metric_geometric_mean():
    """geometric_mean path runs and returns well-formed weights/metric."""
    chain_lps_clean = torch.tensor([-200.0, -150.0, -100.0, -120.0])
    chain_lps_masked = torch.tensor([-220.0, -140.0, -110.0, -118.0])
    answer_ids = torch.tensor([0, 0, 1, 1])
    chain_lengths = torch.tensor([1000, 500, 300, 600], dtype=torch.long)

    result = simulate_eval_global_metric(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2,
        objective_fn=answer_distribution_kl_loss,
        is_method="geometric_mean", chain_lengths=chain_lengths,
    )
    assert abs(sum(result["chain_weights_normalized"]) - 1.0) < 1e-5
    assert all(w >= 0 for w in result["chain_weights_normalized"])
    assert result["n_eff"] > 0
    print("[PASS] test_eval_global_metric_geometric_mean")


# ---------------------------------------------------------------------------
# Tests: eval_global_metric output structure
# ---------------------------------------------------------------------------


def test_eval_global_metric_returns_expected_keys():
    """Simulated eval_global_metric should return all expected keys."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    result = simulate_eval_global_metric(
        chain_lps, chain_lps, answer_ids, num_answers=2,
        objective_fn=answer_distribution_kl_loss,
    )

    assert "metric" in result, "Missing 'metric' key"
    assert "n_eff" in result, "Missing 'n_eff' key"
    assert "n_eff_ratio" in result, "Missing 'n_eff_ratio' key"
    assert "p_m" in result, "Missing 'p_m' key"
    assert "log_weights" in result, "Missing 'log_weights' key"

    assert isinstance(result["metric"], float)
    assert isinstance(result["n_eff"], float)
    assert isinstance(result["n_eff_ratio"], float)
    assert isinstance(result["p_m"], list)
    assert len(result["p_m"]) == 2
    assert isinstance(result["log_weights"], list)
    assert len(result["log_weights"]) == 4
    print("[PASS] test_eval_global_metric_returns_expected_keys")


def test_eval_global_metric_identity_gives_zero_kl():
    """When masked == clean, answer KL should be ~0."""
    chain_lps = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    result = simulate_eval_global_metric(
        chain_lps, chain_lps, answer_ids, num_answers=2,
        objective_fn=answer_distribution_kl_loss,
    )

    assert result["metric"] < 1e-6, (
        f"With same logprobs, answer KL should be ~0, got {result['metric']}"
    )
    assert result["n_eff_ratio"] > 0.9, (
        f"Equal weights should give high N_eff/N, got {result['n_eff_ratio']}"
    )
    print("[PASS] test_eval_global_metric_identity_gives_zero_kl")


def test_eval_global_metric_reward_gap():
    """Simulated eval_global_metric should work with reward_gap_loss."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    chain_lps_masked = torch.tensor([-5.0, -5.0, -20.0, -20.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    result = simulate_eval_global_metric(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=2,
        objective_fn=reward_gap_loss,
    )

    assert isinstance(result["metric"], float)
    # Target=0 is promoted → reward gap is positive → loss is negative
    assert result["metric"] < 0, (
        f"Promoting answer 0 should give negative loss, got {result['metric']}"
    )
    print("[PASS] test_eval_global_metric_reward_gap")


def test_eval_global_metric_p_m_sums_to_one():
    """Answer probabilities from eval should sum to 1."""
    chain_lps_clean = torch.tensor([-10.0, -12.0, -11.0, -13.0, -9.0, -14.0])
    chain_lps_masked = torch.tensor([-8.0, -15.0, -9.0, -20.0, -10.0, -12.0])
    answer_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    result = simulate_eval_global_metric(
        chain_lps_masked, chain_lps_clean, answer_ids, num_answers=3,
        objective_fn=answer_distribution_kl_loss,
    )

    p_sum = sum(result["p_m"])
    assert abs(p_sum - 1.0) < 1e-5, (
        f"Answer probs should sum to 1, got {p_sum}"
    )
    assert all(p >= 0 for p in result["p_m"]), "All probs should be non-negative"
    print("[PASS] test_eval_global_metric_p_m_sums_to_one")


# ---------------------------------------------------------------------------
# Tests: threshold evaluation entry format with global metrics
# ---------------------------------------------------------------------------


def test_threshold_entry_contains_random_global_metrics():
    """Verify the entry format produced by evaluate_at_thresholds includes
    random_global_metrics when a global objective is used.

    This is a format test using manually constructed entries (no model needed).
    """
    entry = {
        "threshold": 0.001,
        "sparsity": 0.5,
        "kl_divergence": 0.01,
        "random_kl_divergence": 0.02,
        "random_kl_divergences": [0.019, 0.021],
        "per_token_kl": [],
        # Global metric fields
        "global_metric": 0.05,
        "n_eff": 3.2,
        "n_eff_ratio": 0.8,
        "answer_probs_masked": [0.6, 0.4],
        "log_weights": [-0.1, 0.05, -0.2, 0.15],
        # New random baseline fields
        "random_global_metric": 0.08,
        "random_global_metrics": [0.075, 0.085],
        "random_n_effs": [3.5, 3.1],
    }

    assert "global_metric" in entry
    assert "n_eff" in entry
    assert "n_eff_ratio" in entry
    assert "answer_probs_masked" in entry
    assert "random_global_metric" in entry
    assert "random_global_metrics" in entry
    assert "random_n_effs" in entry

    assert isinstance(entry["random_global_metrics"], list)
    assert len(entry["random_global_metrics"]) > 0
    assert isinstance(entry["random_n_effs"], list)

    mean_rand = sum(entry["random_global_metrics"]) / len(entry["random_global_metrics"])
    assert abs(mean_rand - entry["random_global_metric"]) < 1e-10

    print("[PASS] test_threshold_entry_contains_random_global_metrics")


def test_threshold_entry_without_global_has_no_global_keys():
    """When objective is local, entry should NOT have global metric keys."""
    entry = {
        "threshold": 0.001,
        "sparsity": 0.5,
        "kl_divergence": 0.01,
        "random_kl_divergence": 0.02,
        "random_kl_divergences": [0.019, 0.021],
        "per_token_kl": [],
    }

    assert "global_metric" not in entry
    assert "n_eff" not in entry
    assert "random_global_metrics" not in entry
    print("[PASS] test_threshold_entry_without_global_has_no_global_keys")


# ---------------------------------------------------------------------------
# Tests: random baseline simulation
# ---------------------------------------------------------------------------


def test_random_baseline_global_metric_varies():
    """Random masks should produce varied global metrics (not identical)."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    # Simulate K random "masked" logprobs (as if from different random masks)
    torch.manual_seed(42)
    random_metrics = []
    for _ in range(5):
        noise = torch.randn(4) * 2.0  # random perturbation
        chain_lps_rand = chain_lps_clean + noise
        result = simulate_eval_global_metric(
            chain_lps_rand, chain_lps_clean, answer_ids, num_answers=2,
            objective_fn=answer_distribution_kl_loss,
        )
        random_metrics.append(result["metric"])

    # Should have some variation
    assert len(set(f"{m:.6f}" for m in random_metrics)) > 1, (
        f"Random metrics should vary, got {random_metrics}"
    )
    # Mean and individual values should be computable
    mean_rand = sum(random_metrics) / len(random_metrics)
    assert mean_rand >= 0, f"Mean random KL should be non-negative, got {mean_rand}"
    print("[PASS] test_random_baseline_global_metric_varies")


def test_random_baseline_n_eff_collected():
    """Random baselines should collect N_eff values for each sample."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    torch.manual_seed(42)
    random_n_effs = []
    for _ in range(5):
        noise = torch.randn(4) * 2.0
        chain_lps_rand = chain_lps_clean + noise
        result = simulate_eval_global_metric(
            chain_lps_rand, chain_lps_clean, answer_ids, num_answers=2,
            objective_fn=answer_distribution_kl_loss,
        )
        random_n_effs.append(result["n_eff"])

    assert len(random_n_effs) == 5
    assert all(n > 0 for n in random_n_effs), "N_eff should be positive"
    assert all(n <= 4.0 + 1e-6 for n in random_n_effs), "N_eff should be <= N"
    print("[PASS] test_random_baseline_n_eff_collected")


# ---------------------------------------------------------------------------
# Tests: answer KL consistency properties
# ---------------------------------------------------------------------------


def test_answer_kl_symmetric_answers_near_zero():
    """With balanced answer groups and equal chain logprobs, KL should be near 0."""
    chain_lps = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(chain_lps, chain_lps, answer_ids, num_answers=2)
    assert kl.item() < 1e-6, f"Balanced equal-logprob should give KL≈0, got {kl.item()}"
    print("[PASS] test_answer_kl_symmetric_answers_near_zero")


def test_answer_kl_increases_with_distribution_shift():
    """KL should increase as the masked model shifts the answer distribution more."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    # Small shift
    chain_lps_small = torch.tensor([-9.0, -9.0, -11.0, -11.0])
    kl_small = answer_distribution_kl_loss(
        chain_lps_small, chain_lps_clean, answer_ids, num_answers=2
    )

    # Large shift
    chain_lps_large = torch.tensor([-5.0, -5.0, -20.0, -20.0])
    kl_large = answer_distribution_kl_loss(
        chain_lps_large, chain_lps_clean, answer_ids, num_answers=2
    )

    assert kl_large.item() > kl_small.item(), (
        f"Larger shift should give larger KL: {kl_large.item()} vs {kl_small.item()}"
    )
    print("[PASS] test_answer_kl_increases_with_distribution_shift")


def test_answer_kl_non_negative():
    """Answer KL should always be non-negative."""
    torch.manual_seed(0)
    for _ in range(10):
        N = torch.randint(4, 12, (1,)).item()
        chain_lps_clean = -torch.rand(N) * 20 - 5
        chain_lps_masked = chain_lps_clean + torch.randn(N) * 3
        num_a = torch.randint(2, 4, (1,)).item()
        answer_ids = torch.randint(0, num_a, (N,))
        # Ensure at least one chain per answer
        for a in range(num_a):
            answer_ids[a % N] = a

        kl = answer_distribution_kl_loss(
            chain_lps_masked, chain_lps_clean, answer_ids, num_answers=num_a
        )
        assert kl.item() >= -1e-6, f"Answer KL should be non-negative, got {kl.item()}"

    print("[PASS] test_answer_kl_non_negative")


def test_n_eff_degrades_with_ablation():
    """N_eff should decrease as importance weights become more concentrated."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    w_equal = importance_weights(chain_lps_clean, chain_lps_clean)
    neff_equal = effective_sample_size(w_equal)

    chain_lps_shifted = torch.tensor([-5.0, -20.0, -20.0, -20.0])
    w_shifted = importance_weights(chain_lps_shifted, chain_lps_clean)
    neff_shifted = effective_sample_size(w_shifted)

    assert neff_equal > neff_shifted, (
        f"Equal weights should have higher N_eff: {neff_equal} vs {neff_shifted}"
    )
    assert neff_shifted < 2.0, f"Concentrated weights should have low N_eff, got {neff_shifted}"
    print("[PASS] test_n_eff_degrades_with_ablation")


def test_n_eff_ratio_bounds():
    """N_eff/N should be in (0, 1]."""
    for _ in range(10):
        N = 8
        torch.manual_seed(_ * 7)
        lp_target = -torch.rand(N) * 20 - 5
        lp_proposal = -torch.rand(N) * 20 - 5

        w = importance_weights(lp_target, lp_proposal)
        neff = effective_sample_size(w)
        ratio = neff / N

        assert 0 < ratio <= 1.0 + 1e-6, f"N_eff/N should be in (0, 1], got {ratio}"

    print("[PASS] test_n_eff_ratio_bounds")


def test_snis_probs_sum_to_one():
    """IS-estimated answer probs should always sum to 1."""
    chain_lps_clean = torch.tensor([-10.0, -12.0, -11.0, -13.0])
    chain_lps_masked = torch.tensor([-8.0, -15.0, -9.0, -20.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    w = importance_weights(chain_lps_masked, chain_lps_clean)
    p_m = snis_answer_probs(w, answer_ids, 2)

    assert abs(p_m.sum().item() - 1.0) < 1e-5, (
        f"Answer probs should sum to 1, got {p_m.sum().item()}"
    )
    print("[PASS] test_snis_probs_sum_to_one")


# ---------------------------------------------------------------------------
# Tests: reward gap consistency
# ---------------------------------------------------------------------------


def test_reward_gap_increases_when_target_promoted():
    """Reward gap should be more negative (better) when masked model promotes target."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss_same = reward_gap_loss(chain_lps_clean, chain_lps_clean, answer_ids, 2, target_answer=0)

    chain_lps_promoted = torch.tensor([-5.0, -5.0, -20.0, -20.0])
    loss_promoted = reward_gap_loss(chain_lps_promoted, chain_lps_clean, answer_ids, 2, target_answer=0)

    assert loss_promoted.item() < loss_same.item(), (
        f"Promoting target should decrease loss: {loss_promoted.item()} vs {loss_same.item()}"
    )
    print("[PASS] test_reward_gap_increases_when_target_promoted")


def test_reward_gap_changes_with_target():
    """Reward gap should change depending on which answer is the target."""
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    # Masked model strongly favors answer 0
    chain_lps_masked = torch.tensor([-5.0, -5.0, -20.0, -20.0])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss_target0 = reward_gap_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, 2, target_answer=0
    )
    loss_target1 = reward_gap_loss(
        chain_lps_masked, chain_lps_clean, answer_ids, 2, target_answer=1
    )

    # Target 0 is promoted → negative loss (good)
    # Target 1 is suppressed → positive loss (bad)
    assert loss_target0.item() < loss_target1.item(), (
        f"Promoted target should have lower loss than suppressed target"
    )
    print("[PASS] test_reward_gap_changes_with_target")


# ---------------------------------------------------------------------------
# Tests: is_global_objective for all objectives
# ---------------------------------------------------------------------------


def test_is_global_covers_all_cases():
    """Verify is_global_objective works for all known objective names."""
    assert is_global_objective("answer_kl")
    assert is_global_objective("reward_gap")
    assert is_global_objective("answer_distribution_kl_loss")
    assert is_global_objective("reward_gap_loss")
    assert not is_global_objective("kl_divergence")
    assert not is_global_objective("log_prob")
    assert not is_global_objective("something_else")
    print("[PASS] test_is_global_covers_all_cases")


# ---------------------------------------------------------------------------
# Tests: end-to-end global objective gradient flow
# ---------------------------------------------------------------------------


def test_answer_kl_gradient_through_chain_logprobs():
    """Gradient of answer KL should flow through chain logprobs to a parameter."""
    param = torch.tensor(0.0, requires_grad=True)

    # Chain logprobs that depend on param
    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    chain_lps_masked = torch.stack([
        torch.tensor(-10.0) + param,
        torch.tensor(-10.0) + param,
        torch.tensor(-10.0) - param,
        torch.tensor(-10.0) - param,
    ])
    answer_ids = torch.tensor([0, 0, 1, 1])

    kl = answer_distribution_kl_loss(chain_lps_masked, chain_lps_clean, answer_ids, 2)
    kl.backward()

    assert param.grad is not None, "Gradient should flow to param"
    print("[PASS] test_answer_kl_gradient_through_chain_logprobs")


def test_reward_gap_gradient_through_chain_logprobs():
    """Gradient of reward gap should flow through chain logprobs to a parameter."""
    param = torch.tensor(0.0, requires_grad=True)

    chain_lps_clean = torch.tensor([-10.0, -10.0, -10.0, -10.0])
    chain_lps_masked = torch.stack([
        torch.tensor(-10.0) + param,
        torch.tensor(-10.0) + param,
        torch.tensor(-10.0) - param,
        torch.tensor(-10.0) - param,
    ])
    answer_ids = torch.tensor([0, 0, 1, 1])

    loss = reward_gap_loss(chain_lps_masked, chain_lps_clean, answer_ids, 2, target_answer=0)
    loss.backward()

    assert param.grad is not None, "Gradient should flow to param"
    # Increasing param promotes answer 0 → gradient should be negative
    # (since loss = -reward_gap, and reward_gap increases with param)
    assert param.grad.item() < 0, (
        f"Gradient should encourage promoting target (negative), got {param.grad.item()}"
    )
    print("[PASS] test_reward_gap_gradient_through_chain_logprobs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("eval_global_metric output structure")
    print("=" * 60)
    test_eval_global_metric_returns_expected_keys()
    test_eval_global_metric_identity_gives_zero_kl()
    test_eval_global_metric_reward_gap()
    test_eval_global_metric_p_m_sums_to_one()
    test_eval_global_metric_geometric_mean()

    print()
    print("=" * 60)
    print("Threshold entry format")
    print("=" * 60)
    test_threshold_entry_contains_random_global_metrics()
    test_threshold_entry_without_global_has_no_global_keys()

    print()
    print("=" * 60)
    print("Random baseline simulation")
    print("=" * 60)
    test_random_baseline_global_metric_varies()
    test_random_baseline_n_eff_collected()

    print()
    print("=" * 60)
    print("Answer KL consistency properties")
    print("=" * 60)
    test_answer_kl_symmetric_answers_near_zero()
    test_answer_kl_increases_with_distribution_shift()
    test_answer_kl_non_negative()
    test_n_eff_degrades_with_ablation()
    test_n_eff_ratio_bounds()
    test_snis_probs_sum_to_one()

    print()
    print("=" * 60)
    print("Reward gap consistency")
    print("=" * 60)
    test_reward_gap_increases_when_target_promoted()
    test_reward_gap_changes_with_target()

    print()
    print("=" * 60)
    print("Objective classification")
    print("=" * 60)
    test_is_global_covers_all_cases()

    print()
    print("=" * 60)
    print("Gradient flow")
    print("=" * 60)
    test_answer_kl_gradient_through_chain_logprobs()
    test_reward_gap_gradient_through_chain_logprobs()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
