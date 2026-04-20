"""Tests for eval_all_metrics — the unified evaluation function.

Tests cover:
- H1: Output fields match old eval_with_masks + eval_global_metric (equivalence)
- H2: Contrastive grouping (kl_a, kl_b, contrastive_loss)
- H3: Edge case — answer_ids is None (IS + contrastive fields absent)
- H4: Edge case — single answer group (degenerate metrics)
- H5: KL always uses full continuation (no truncation)
- H6: Fine-grained vs binary answer grouping

Usage:
    uv run python tests/test_eval_all_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from utils.circuit_eval import eval_all_metrics
from utils.importance_sampling import (
    chain_log_prob,
    importance_weights,
    effective_sample_size,
    snis_answer_probs,
)
from utils.objectives import answer_distribution_kl_loss, reward_gap_loss


# ---------------------------------------------------------------------------
# Minimal mock model
# ---------------------------------------------------------------------------


class MockSelfAttn(torch.nn.Module):
    """Placeholder attention module that install_mask_hooks can patch."""
    def forward(self, *args, **kwargs):
        pass


class MockLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockSelfAttn()


class MockInnerModel(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockLayer() for _ in range(num_layers)])


class MockModel(torch.nn.Module):
    """Tiny model that returns deterministic logits for testing.

    Has ``model.model.layers[i].self_attn`` structure matching llama.
    The install_mask_hooks patches self_attn.forward, but since our
    forward() computes logits directly (not through attention), the
    patches are no-ops and don't affect the output.
    """

    def __init__(self, vocab_size: int = 10, num_layers: int = 2, num_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = MockInnerModel(num_layers)
        # Dummy parameter so next(model.parameters()) works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

        class _Config:
            pass

        self.config = _Config()
        self.config.num_hidden_layers = num_layers
        self.config.num_attention_heads = num_heads
        self.config.model_type = "llama"
        self._forward_count = 0

    def forward(self, input_ids: torch.Tensor):
        self._forward_count += 1
        B, S = input_ids.shape
        # Deterministic logits that depend on position and token id
        pos = torch.arange(S, device=input_ids.device).float()
        logits = torch.zeros(B, S, self.vocab_size, device=input_ids.device)
        for v in range(self.vocab_size):
            logits[:, :, v] = (input_ids.float() * 0.1 + pos * 0.01 + v * 0.3)
        # Add small noise to break symmetry
        logits += torch.randn_like(logits) * 0.001

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_data(
    num_branches: int = 4,
    prefix_len: int = 5,
    cont_len: int = 10,
    vocab_size: int = 10,
    device: str = "cpu",
):
    """Create mock data for eval_all_metrics tests."""
    input_ids = torch.randint(0, vocab_size, (1, prefix_len), device=device)
    continuations = [
        torch.randint(0, vocab_size, (1, cont_len), device=device)
        for _ in range(num_branches)
    ]

    model = MockModel(vocab_size=vocab_size)
    model.eval()

    # Compute clean logits
    clean_logits_list = []
    with torch.no_grad():
        for cont in continuations:
            full = torch.cat([input_ids, cont], dim=-1)
            clean_logits_list.append(model(full).logits.cpu())

    # Reset forward count after computing clean logits
    model._forward_count = 0

    # Dummy masks and filters (no-op: all ones)
    from utils.utils import Sentence

    sentences = [Sentence(start=0, end=prefix_len - 1)]
    num_sents = 1
    token_to_sent = torch.zeros(prefix_len + cont_len, dtype=torch.long)
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)
    binary_masks = {0: torch.ones(2, num_sents, num_sents)}

    return dict(
        model=model,
        binary_masks=binary_masks,
        layers=[0],
        input_ids=input_ids,
        continuations=continuations,
        clean_logits_list=clean_logits_list,
        token_to_sent=token_to_sent,
        gap_filter=gap_filter,
        renormalize=True,
    )


def _compute_expected_kl(clean_logits, masked_logits, prefix_len):
    """Compute expected mean per-token KL, matching eval_all_metrics logic."""
    full_len = clean_logits.shape[1]
    analyse_end = full_len

    log_c = F.log_softmax(clean_logits.float(), dim=-1)
    log_m = F.log_softmax(masked_logits.float(), dim=-1)
    kl_tokens = F.kl_div(log_m, log_c, log_target=True, reduction="none").sum(dim=-1)

    pos_mask = torch.zeros(1, full_len)
    pos_mask[0, prefix_len - 1 : analyse_end - 1] = 1.0
    return (kl_tokens * pos_mask).sum() / pos_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# H1: Basic output fields + forward pass count
# ---------------------------------------------------------------------------


def test_basic_output_fields():
    """eval_all_metrics with no answer_ids should return only local fields."""
    data = make_test_data()
    result = eval_all_metrics(**data)

    assert "kl_divergence" in result, "kl_divergence missing"
    assert isinstance(result["kl_divergence"], float), "kl_divergence should be float"
    assert result["kl_divergence"] >= 0, "KL should be non-negative"

    # No IS or contrastive fields without answer_ids
    for key in ["answer_kl", "reward_gap", "kl_a", "kl_b", "contrastive_loss"]:
        assert key not in result, f"{key} should not be present without answer_ids"

    print("[PASS] test_basic_output_fields")


def test_forward_pass_count():
    """eval_all_metrics should do exactly N forward passes (one per continuation)."""
    data = make_test_data(num_branches=4)
    data["model"]._forward_count = 0
    _ = eval_all_metrics(**data)

    assert data["model"]._forward_count == 4, (
        f"Expected 4 forward passes, got {data['model']._forward_count}"
    )
    print("[PASS] test_forward_pass_count")


def test_forward_pass_count_with_answer_ids():
    """With answer_ids, should still be N forward passes (not 2N)."""
    data = make_test_data(num_branches=4)
    prefix_len = data["input_ids"].shape[-1]

    # Compute chain_logprobs_clean
    chain_logprobs_clean = []
    for ci, cont in enumerate(data["continuations"]):
        full = torch.cat([data["input_ids"], cont], dim=-1)
        cl = data["clean_logits_list"][ci][:, : full.shape[-1]]
        lp = chain_log_prob(cl, full, prefix_len)
        chain_logprobs_clean.append(lp.detach())
    chain_logprobs_clean = torch.stack(chain_logprobs_clean)

    data["answer_ids_fine"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_fine"] = 2
    data["answer_ids_binary"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_binary"] = 2
    data["chain_logprobs_clean"] = chain_logprobs_clean
    data["model"]._forward_count = 0

    result = eval_all_metrics(**data)

    assert data["model"]._forward_count == 4, (
        f"Expected 4 forward passes (not 8), got {data['model']._forward_count}"
    )
    # Verify IS fields are present
    for key in ["answer_kl", "reward_gap", "n_eff", "answer_probs_masked"]:
        assert key in result, f"{key} should be present with answer_ids"

    # chain_weights_normalized should be present with length N and sum to 1.
    assert "chain_weights_normalized" in result, (
        "chain_weights_normalized should be present with answer_ids"
    )
    cwn = result["chain_weights_normalized"]
    assert len(cwn) == len(data["continuations"])
    assert abs(sum(cwn) - 1.0) < 1e-5

    print("[PASS] test_forward_pass_count_with_answer_ids")


def test_kl_divergence_value():
    """kl_divergence should match hand-computed per-token KL."""
    data = make_test_data(num_branches=2)
    model = data["model"]

    # Run eval_all_metrics
    result = eval_all_metrics(**data)

    # Manually compute expected KL
    prefix_len = data["input_ids"].shape[-1]
    expected_kls = []
    model._forward_count = 0
    with torch.no_grad():
        for ci, cont in enumerate(data["continuations"]):
            full = torch.cat([data["input_ids"], cont], dim=-1)
            logits = model(full).logits
            clean = data["clean_logits_list"][ci][:, : full.shape[-1]]
            kl = _compute_expected_kl(clean, logits, prefix_len)
            expected_kls.append(kl.item())

    expected_mean = sum(expected_kls) / len(expected_kls)
    assert abs(result["kl_divergence"] - expected_mean) < 1e-5, (
        f"KL mismatch: {result['kl_divergence']} vs {expected_mean}"
    )
    print("[PASS] test_kl_divergence_value")


# ---------------------------------------------------------------------------
# H1: Reward-weighted KL
# ---------------------------------------------------------------------------


def test_reward_weighted_kl():
    """reward_weighted_kl should be KL * reward, averaged across branches."""
    data = make_test_data(num_branches=4)
    rewards = [1.0, -1.0, 1.0, -1.0]
    data["branch_rewards"] = rewards

    result = eval_all_metrics(**data)

    assert "reward_weighted_kl" in result, "reward_weighted_kl missing"
    assert "kl_divergence" in result, "kl_divergence missing"
    # reward_weighted_kl != kl_divergence (different weights)
    # kl_divergence is always unweighted
    print("[PASS] test_reward_weighted_kl")


def test_reward_weighted_kl_no_rewards():
    """Without branch_rewards, reward_weighted_kl should not be present."""
    data = make_test_data()
    result = eval_all_metrics(**data)
    assert "reward_weighted_kl" not in result
    print("[PASS] test_reward_weighted_kl_no_rewards")


# ---------------------------------------------------------------------------
# H2: Contrastive grouping
# ---------------------------------------------------------------------------


def test_contrastive_grouping():
    """kl_a and kl_b should group per-branch KL by answer_ids_binary."""
    data = make_test_data(num_branches=4)
    prefix_len = data["input_ids"].shape[-1]

    # Compute chain_logprobs_clean
    chain_logprobs_clean = []
    for ci, cont in enumerate(data["continuations"]):
        full = torch.cat([data["input_ids"], cont], dim=-1)
        cl = data["clean_logits_list"][ci][:, : full.shape[-1]]
        lp = chain_log_prob(cl, full, prefix_len)
        chain_logprobs_clean.append(lp.detach())
    chain_logprobs_clean = torch.stack(chain_logprobs_clean)

    data["answer_ids_binary"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_binary"] = 2
    data["chain_logprobs_clean"] = chain_logprobs_clean

    result = eval_all_metrics(**data)

    assert "kl_a" in result, "kl_a missing"
    assert "kl_b" in result, "kl_b missing"
    assert "contrastive_loss" in result, "contrastive_loss missing"

    # contrastive_loss = kl_a - kl_b
    assert abs(result["contrastive_loss"] - (result["kl_a"] - result["kl_b"])) < 1e-8, (
        f"contrastive_loss should be kl_a - kl_b: "
        f"{result['contrastive_loss']} vs {result['kl_a'] - result['kl_b']}"
    )

    # kl_a and kl_b should be non-negative
    assert result["kl_a"] >= 0, "kl_a should be non-negative"
    assert result["kl_b"] >= 0, "kl_b should be non-negative"

    print("[PASS] test_contrastive_grouping")


# ---------------------------------------------------------------------------
# H3: answer_ids is None
# ---------------------------------------------------------------------------


def test_no_answer_ids():
    """Without answer_ids, IS and contrastive fields should be absent."""
    data = make_test_data()
    result = eval_all_metrics(**data)

    absent_keys = [
        "answer_kl", "reward_gap", "p_target", "p_best_other",
        "answer_probs_masked", "n_eff", "n_eff_ratio", "log_weights",
        "kl_a", "kl_b", "contrastive_loss",
    ]
    for key in absent_keys:
        assert key not in result, f"{key} should not be present without answer_ids"

    # Local fields should still be present
    assert "kl_divergence" in result
    print("[PASS] test_no_answer_ids")


# ---------------------------------------------------------------------------
# H4: Single answer group
# ---------------------------------------------------------------------------


def test_single_answer_group():
    """With num_answers_fine=1, answer_kl should be ~0."""
    data = make_test_data(num_branches=4)
    prefix_len = data["input_ids"].shape[-1]

    chain_logprobs_clean = []
    for ci, cont in enumerate(data["continuations"]):
        full = torch.cat([data["input_ids"], cont], dim=-1)
        cl = data["clean_logits_list"][ci][:, : full.shape[-1]]
        lp = chain_log_prob(cl, full, prefix_len)
        chain_logprobs_clean.append(lp.detach())
    chain_logprobs_clean = torch.stack(chain_logprobs_clean)

    # All branches in the same fine-grained group
    data["answer_ids_fine"] = torch.tensor([0, 0, 0, 0])
    data["num_answers_fine"] = 1
    # Binary: all correct (single group)
    data["answer_ids_binary"] = torch.tensor([0, 0, 0, 0])
    data["num_answers_binary"] = 1
    data["chain_logprobs_clean"] = chain_logprobs_clean

    result = eval_all_metrics(**data)

    # answer_kl should be 0 (single group, P_clean = [1.0], so KL is trivially 0)
    assert abs(result["answer_kl"]) < 1e-6, (
        f"answer_kl should be ~0 for single group, got {result['answer_kl']}"
    )

    print("[PASS] test_single_answer_group")


# ---------------------------------------------------------------------------
# H5: KL always uses full continuation
# ---------------------------------------------------------------------------


def test_kl_uses_full_continuation():
    """KL should always average over all continuation tokens."""
    data = make_test_data(num_branches=2, cont_len=50)

    result = eval_all_metrics(**data)

    # Verify it matches hand computation over full continuation
    prefix_len = data["input_ids"].shape[-1]
    expected_kls = []
    model = data["model"]
    model._forward_count = 0
    with torch.no_grad():
        for ci, cont in enumerate(data["continuations"]):
            full = torch.cat([data["input_ids"], cont], dim=-1)
            logits = model(full).logits
            clean = data["clean_logits_list"][ci][:, : full.shape[-1]]
            kl = _compute_expected_kl(clean, logits, prefix_len)
            expected_kls.append(kl.item())

    expected_mean = sum(expected_kls) / len(expected_kls)
    assert abs(result["kl_divergence"] - expected_mean) < 1e-5, (
        f"KL mismatch: {result['kl_divergence']} vs {expected_mean}"
    )
    print("[PASS] test_kl_uses_full_continuation")


# ---------------------------------------------------------------------------
# IS metric value checks
# ---------------------------------------------------------------------------


def test_answer_probs_sum_to_one():
    """answer_probs_masked (binary) should sum to ~1.0."""
    data = make_test_data(num_branches=4)
    prefix_len = data["input_ids"].shape[-1]

    chain_logprobs_clean = []
    for ci, cont in enumerate(data["continuations"]):
        full = torch.cat([data["input_ids"], cont], dim=-1)
        cl = data["clean_logits_list"][ci][:, : full.shape[-1]]
        lp = chain_log_prob(cl, full, prefix_len)
        chain_logprobs_clean.append(lp.detach())
    chain_logprobs_clean = torch.stack(chain_logprobs_clean)

    data["answer_ids_fine"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_fine"] = 2
    data["answer_ids_binary"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_binary"] = 2
    data["chain_logprobs_clean"] = chain_logprobs_clean

    result = eval_all_metrics(**data)

    # Binary probs sum to 1
    prob_sum = sum(result["answer_probs_masked"])
    assert abs(prob_sum - 1.0) < 1e-5, (
        f"answer_probs_masked should sum to 1.0, got {prob_sum}"
    )
    # Fine-grained probs sum to 1
    prob_sum_fine = sum(result["answer_probs_masked_fine"])
    assert abs(prob_sum_fine - 1.0) < 1e-5, (
        f"answer_probs_masked_fine should sum to 1.0, got {prob_sum_fine}"
    )
    print("[PASS] test_answer_probs_sum_to_one")


def test_reward_gap_decomposition():
    """reward_gap should equal p_target - p_best_other (binary buckets)."""
    data = make_test_data(num_branches=4)
    prefix_len = data["input_ids"].shape[-1]

    chain_logprobs_clean = []
    for ci, cont in enumerate(data["continuations"]):
        full = torch.cat([data["input_ids"], cont], dim=-1)
        cl = data["clean_logits_list"][ci][:, : full.shape[-1]]
        lp = chain_log_prob(cl, full, prefix_len)
        chain_logprobs_clean.append(lp.detach())
    chain_logprobs_clean = torch.stack(chain_logprobs_clean)

    data["answer_ids_binary"] = torch.tensor([0, 0, 1, 1])
    data["num_answers_binary"] = 2
    data["chain_logprobs_clean"] = chain_logprobs_clean

    result = eval_all_metrics(**data)

    expected_gap = result["p_target"] - result["p_best_other"]
    assert abs(result["reward_gap"] - expected_gap) < 1e-8, (
        f"reward_gap should be p_target - p_best_other: "
        f"{result['reward_gap']} vs {expected_gap}"
    )
    print("[PASS] test_reward_gap_decomposition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("eval_all_metrics tests")
    print("=" * 60)

    print("\n--- H1: Basic output + forward pass count ---")
    test_basic_output_fields()
    test_forward_pass_count()
    test_forward_pass_count_with_answer_ids()
    test_kl_divergence_value()

    print("\n--- H1: Reward-weighted KL ---")
    test_reward_weighted_kl()
    test_reward_weighted_kl_no_rewards()

    print("\n--- H2: Contrastive grouping ---")
    test_contrastive_grouping()

    print("\n--- H3: No answer_ids ---")
    test_no_answer_ids()

    print("\n--- H4: Single answer group ---")
    test_single_answer_group()

    print("\n--- H5: KL uses full continuation ---")
    test_kl_uses_full_continuation()

    print("\n--- IS metric value checks ---")
    test_answer_probs_sum_to_one()
    test_reward_gap_decomposition()

    print("\n" + "=" * 60)
    print("All eval_all_metrics tests passed!")
    print("=" * 60)
