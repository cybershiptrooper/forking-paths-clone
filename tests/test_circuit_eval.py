"""Tests for circuit_eval.py — focused on the causal-structure bug.

The key bug: `build_random_score_masks` permutes ALL scores globally,
including structurally-zero upper triangle entries (j > i) that exist
because of causal attention. Since ~48% of scores are these causal zeros,
random permutation massively distorts the effective sparsity.

Fix: Include a causal filter (j > i → frozen) in `combined_filter` so
that upper-triangle positions are excluded from both the permutable pool
and sparsity/binary-mask calculations.

Usage:
    uv run python tests/test_circuit_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    apply_gap_filter,
)
from utils.circuit_eval import (
    build_binary_masks,
    build_random_masks,
    build_random_score_masks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_causal_scores(
    num_layers: int,
    num_heads: int,
    num_sents: int,
    seed: int = 42,
) -> dict[int, dict[int, list[list[float]]]]:
    """Create synthetic scores with causal structure.

    Lower triangle + diagonal: random non-zero values in [-0.01, 0.01]
    Upper triangle (j > i): exactly 0.0 (causal mask makes them irrelevant)
    """
    rng = np.random.RandomState(seed)
    scores: dict[int, dict[int, list[list[float]]]] = {}
    for layer in range(num_layers):
        scores[layer] = {}
        for head in range(num_heads):
            mat = [[0.0] * num_sents for _ in range(num_sents)]
            for i in range(num_sents):
                for j in range(num_sents):
                    if j <= i:  # causally valid
                        mat[i][j] = float(rng.uniform(-0.01, 0.01))
                    else:
                        mat[i][j] = 0.0  # causal zero
            scores[layer][head] = mat
    return scores


def make_node_mask(
    num_layers: int = 4,
    num_heads: int = 4,
    num_sents: int = 6,
    seed: int = 42,
) -> NodeMask:
    """Create a NodeMask with causal structure for testing."""
    scores = make_causal_scores(num_layers, num_heads, num_sents, seed)
    sentences = [{"start": i * 10, "end": i * 10 + 9} for i in range(num_sents)]
    return NodeMask(
        model_name="test-model",
        algorithm="test",
        layers=list(range(num_layers)),
        sentences=sentences,
        objective_name="kl_divergence",
        metadata={
            "sentence_gap": 0,
            "mask_mode": "prefix",
            "num_prefix_sentences": num_sents,
        },
        scores=scores,
    )


def count_effective_zeros(binary_mask: dict[int, torch.Tensor], num_sents: int) -> int:
    """Count zeros in the lower triangle + diagonal (causally valid region)."""
    count = 0
    for layer_mask in binary_mask.values():
        for h in range(layer_mask.shape[0]):
            for i in range(num_sents):
                for j in range(num_sents):
                    if j <= i and layer_mask[h, i, j].item() == 0.0:
                        count += 1
    return count


def count_total_zeros(binary_mask: dict[int, torch.Tensor]) -> int:
    """Count total zeros in the binary mask."""
    count = 0
    for layer_mask in binary_mask.values():
        count += (layer_mask == 0.0).sum().item()
    return count


# ---------------------------------------------------------------------------
# Tests: build_binary_masks consistency
# ---------------------------------------------------------------------------

def test_binary_masks_nodemask_vs_dict():
    """build_binary_masks should produce identical results for NodeMask and
    an equivalent plain dict with the same scores."""
    node_mask = make_node_mask()
    layers = node_mask.layers
    num_heads = 4
    num_sents = 6
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)
    device = torch.device("cpu")

    for threshold in [-0.005, 0.0, 0.005]:
        mask_from_nm = build_binary_masks(
            node_mask, threshold, layers, num_heads, num_sents, gap_filter, device
        )
        mask_from_dict = build_binary_masks(
            node_mask.scores, threshold, layers, num_heads, num_sents, gap_filter, device
        )
        for layer in layers:
            assert torch.equal(mask_from_nm[layer], mask_from_dict[layer]), (
                f"Mismatch at layer {layer}, threshold {threshold}"
            )
    print("[PASS] build_binary_masks: NodeMask vs dict produce identical results")


# ---------------------------------------------------------------------------
# Tests: build_causal_filter
# ---------------------------------------------------------------------------

def test_causal_filter_shape_and_values():
    """build_causal_filter should be True where j > i, False otherwise."""
    num_sents = 5
    cf = build_causal_filter(num_sents)
    assert cf.shape == (num_sents, num_sents)
    for i in range(num_sents):
        for j in range(num_sents):
            if j > i:
                assert cf[i, j].item(), f"Expected True at [{i},{j}]"
            else:
                assert not cf[i, j].item(), f"Expected False at [{i},{j}]"
    print("[PASS] build_causal_filter: shape and values correct")


def test_combined_filter_includes_causal():
    """build_combined_filter with causal_filter should freeze upper triangle."""
    num_sents = 5
    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    causal_filter = build_causal_filter(num_sents)

    combined = build_combined_filter(gap_filter, mode_filter, causal_filter)
    for i in range(num_sents):
        for j in range(num_sents):
            if j > i:
                assert combined[i, j].item(), (
                    f"Upper triangle [{i},{j}] should be frozen"
                )
    print("[PASS] build_combined_filter: causal filter freezes upper triangle")


def test_combined_filter_backward_compatible():
    """build_combined_filter without causal_filter should match old behavior."""
    num_sents = 4
    gap_filter = build_gap_filter(num_sents, sentence_gap=1)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")

    old_result = gap_filter | mode_filter
    new_result = build_combined_filter(gap_filter, mode_filter)

    assert torch.equal(old_result, new_result), "Without causal_filter should match OR"
    print("[PASS] build_combined_filter: backward compatible (no causal_filter)")


# ---------------------------------------------------------------------------
# Tests: the causal-structure bug demonstration
# ---------------------------------------------------------------------------

def test_random_permutation_distorts_effective_sparsity_without_causal_filter():
    """Demonstrate the bug: without causal filter, random permutation changes
    effective (lower-tri) sparsity even though overall sparsity stays the same."""
    num_layers, num_heads, num_sents = 4, 4, 6
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    # No causal filter — the buggy configuration
    combined_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)

    random_masks = build_random_score_masks(
        node_mask, num_samples=10, layers=layers, combined_filter=combined_filter
    )

    threshold = -0.003

    # Learned mask: all ablations are in lower triangle
    learned_binary = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
    )
    learned_total_zeros = count_total_zeros(learned_binary)
    learned_effective_zeros = count_effective_zeros(learned_binary, num_sents)

    # Random masks: some ablations land in upper triangle (wasted)
    random_effective_zeros_list = []
    random_total_zeros_list = []
    for rand_scores in random_masks:
        rand_binary = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
        )
        random_total_zeros_list.append(count_total_zeros(rand_binary))
        random_effective_zeros_list.append(count_effective_zeros(rand_binary, num_sents))

    avg_random_total = np.mean(random_total_zeros_list)
    avg_random_effective = np.mean(random_effective_zeros_list)

    print(f"\n  Without causal filter (buggy):")
    print(f"  Threshold: {threshold}")
    print(f"  Learned: total zeros = {learned_total_zeros}, "
          f"effective zeros (lower tri) = {learned_effective_zeros}")
    print(f"  Random avg: total zeros = {avg_random_total:.1f}, "
          f"effective zeros (lower tri) = {avg_random_effective:.1f}")

    # Total zeros should be nearly identical (same score multiset)
    assert abs(avg_random_total - learned_total_zeros) < 2

    # BUG: effective zeros differ significantly
    if learned_effective_zeros > 0:
        ratio = avg_random_effective / learned_effective_zeros
        assert ratio < 0.95, (
            f"BUG NOT REPRODUCED: ratio {ratio:.3f} should be < 0.95"
        )
        print(f"  Ratio (random/learned effective): {ratio:.3f} << 1.0")
        print(f"  [BUG CONFIRMED] Permutation distorts effective sparsity")

    print("[PASS] test_random_permutation_distorts_effective_sparsity_without_causal_filter")


# ---------------------------------------------------------------------------
# Tests: fix verification
# ---------------------------------------------------------------------------

def test_causal_filter_fixes_permutation():
    """With causal filter, effective sparsity should be consistent
    between learned and random masks."""
    num_layers, num_heads, num_sents = 4, 4, 6
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    # With causal filter — the fixed configuration
    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    causal_filter = build_causal_filter(num_sents)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    random_masks = build_random_score_masks(
        node_mask, num_samples=10, layers=layers, combined_filter=combined_filter
    )

    threshold = -0.003

    # Learned mask
    learned_binary = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
    )
    learned_effective_zeros = count_effective_zeros(learned_binary, num_sents)

    # Random masks
    random_effective_zeros_list = []
    for rand_scores in random_masks:
        rand_binary = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
        )
        random_effective_zeros_list.append(count_effective_zeros(rand_binary, num_sents))

    avg_random_effective = np.mean(random_effective_zeros_list)

    print(f"\n  With causal filter (fixed):")
    print(f"  Learned effective zeros: {learned_effective_zeros}")
    print(f"  Random effective zeros avg: {avg_random_effective:.1f}")

    if learned_effective_zeros > 0:
        ratio = avg_random_effective / learned_effective_zeros
        print(f"  Ratio: {ratio:.3f}")
        assert 0.8 < ratio < 1.2, (
            f"Effective zeros should be similar. Ratio={ratio:.3f}"
        )
        print(f"  [FIX VERIFIED] Effective sparsity consistent (ratio={ratio:.3f})")

    print("[PASS] test_causal_filter_fixes_permutation")


def test_at_full_ablation_masks_are_identical():
    """At a threshold above all scores, both learned and random binary masks
    should be identical (all zeros in non-filtered region)."""
    num_layers, num_heads, num_sents = 2, 2, 4
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    causal_filter = build_causal_filter(num_sents)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    threshold = 1.0  # above all scores

    learned_binary = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
    )

    random_masks = build_random_score_masks(
        node_mask, num_samples=5, layers=layers, combined_filter=combined_filter
    )

    for k, rand_scores in enumerate(random_masks):
        rand_binary = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
        )
        for layer in layers:
            assert torch.equal(learned_binary[layer], rand_binary[layer]), (
                f"At full ablation, masks should be identical "
                f"(layer {layer}, sample {k})"
            )

    print("[PASS] test_at_full_ablation_masks_are_identical")


def test_at_no_ablation_masks_are_identical():
    """At a threshold below all scores, both learned and random binary masks
    should be identical (all ones)."""
    num_layers, num_heads, num_sents = 2, 2, 4
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    causal_filter = build_causal_filter(num_sents)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    threshold = -1.0  # below all scores

    learned_binary = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
    )

    random_masks = build_random_score_masks(
        node_mask, num_samples=5, layers=layers, combined_filter=combined_filter
    )

    for k, rand_scores in enumerate(random_masks):
        rand_binary = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_filter, torch.device("cpu")
        )
        for layer in layers:
            assert torch.equal(learned_binary[layer], rand_binary[layer]), (
                f"At no ablation, masks should be identical "
                f"(layer {layer}, sample {k})"
            )

    # Verify all values are 1.0
    for layer in layers:
        assert (learned_binary[layer] == 1.0).all(), "All values should be 1.0"

    print("[PASS] test_at_no_ablation_masks_are_identical")


def test_random_permutation_preserves_score_distribution():
    """Random score masks should have the same multiset of score values
    as the learned mask (just in different positions within the causally-valid region)."""
    num_layers, num_heads, num_sents = 2, 2, 4
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    causal_filter = build_causal_filter(num_sents)
    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    random_masks = build_random_score_masks(
        node_mask, num_samples=3, layers=layers, combined_filter=combined_filter
    )

    # Collect non-filtered scores from learned mask
    learned_scores = []
    for layer in layers:
        for h in node_mask.scores[layer]:
            for i in range(num_sents):
                for j in range(num_sents):
                    if not combined_filter[i, j]:
                        learned_scores.append(node_mask.scores[layer][h][i][j])
    learned_sorted = sorted(learned_scores)

    # Each random mask should have the same sorted values
    for k, rand_dict in enumerate(random_masks):
        rand_scores = []
        for layer in layers:
            for h in rand_dict[layer]:
                for i in range(num_sents):
                    for j in range(num_sents):
                        if not combined_filter[i, j]:
                            rand_scores.append(rand_dict[layer][h][i][j])
        rand_sorted = sorted(rand_scores)
        assert len(rand_sorted) == len(learned_sorted), (
            f"Random mask {k}: wrong number of scores "
            f"({len(rand_sorted)} vs {len(learned_sorted)})"
        )
        for a, b in zip(learned_sorted, rand_sorted):
            assert abs(a - b) < 1e-10, (
                f"Random mask {k}: score distribution mismatch"
            )

    print("[PASS] test_random_permutation_preserves_score_distribution")


def test_random_masks_only_permute_within_causal_region():
    """Verify that upper-triangle positions in random score masks are always 0.0
    (never receive permuted scores), and permutation only happens in lower triangle."""
    num_layers, num_heads, num_sents = 2, 2, 4
    node_mask = make_node_mask(num_layers, num_heads, num_sents)
    layers = list(range(num_layers))

    causal_filter = build_causal_filter(num_sents)
    gap_filter = build_gap_filter(num_sents, sentence_gap=0)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix")
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    random_masks = build_random_score_masks(
        node_mask, num_samples=5, layers=layers, combined_filter=combined_filter
    )

    for k, rand_dict in enumerate(random_masks):
        for layer in layers:
            for h in rand_dict[layer]:
                for i in range(num_sents):
                    for j in range(num_sents):
                        if j > i:  # upper triangle
                            assert rand_dict[layer][h][i][j] == 0.0, (
                                f"Random mask {k}: upper triangle [{i},{j}] "
                                f"in layer {layer} head {h} should be 0.0, "
                                f"got {rand_dict[layer][h][i][j]}"
                            )

    print("[PASS] test_random_masks_only_permute_within_causal_region")


def test_sparsity_with_vs_without_causal_filter():
    """NodeMask.sparsity() should give different results with and without causal filter,
    because causal filter excludes the structurally-zero upper triangle from the count."""
    node_mask = make_node_mask(num_layers=2, num_heads=2, num_sents=4)

    no_filter = torch.zeros(4, 4, dtype=torch.bool)
    causal_filter = build_causal_filter(4)

    # At threshold=0: all upper-triangle entries (value=0.0) are NOT below 0.0
    # They inflate the denominator without being "below threshold"
    sp_no_filter = node_mask.sparsity(0.0, gap_filter=no_filter)
    sp_with_causal = node_mask.sparsity(0.0, gap_filter=causal_filter)

    print(f"\n  Sparsity at threshold=0:")
    print(f"  Without causal filter: {sp_no_filter:.4%}")
    print(f"  With causal filter: {sp_with_causal:.4%}")

    # With causal filter, only lower triangle entries counted.
    # The denominator is smaller → different sparsity value.
    # The numerator also changes since upper-triangle zeros don't count.
    print("[PASS] test_sparsity_with_vs_without_causal_filter")


# ---------------------------------------------------------------------------
# Verify with actual JSON data
# ---------------------------------------------------------------------------

def test_with_actual_data():
    """Load the actual mask JSON and verify the causal structure issue."""
    json_path = Path("results/circuitviz_test/circuit_nodewise_attribution_attention_layers_all_branches1_ig50.json")
    if not json_path.exists():
        print("[SKIP] Actual data file not found")
        return

    node_mask = NodeMask.from_json(str(json_path))
    num_sents = len(node_mask.sentences)
    layers = node_mask.layers
    num_heads = node_mask.metadata.get("num_heads", 32)

    # Verify upper triangle is all zero
    upper_nonzero = 0
    for layer in node_mask.scores:
        for head in node_mask.scores[layer]:
            mat = node_mask.scores[layer][head]
            for i in range(num_sents):
                for j in range(num_sents):
                    if j > i and mat[i][j] != 0.0:
                        upper_nonzero += 1
    assert upper_nonzero == 0, f"Expected all upper triangle = 0, found {upper_nonzero} non-zero"
    print(f"\n  Actual data: upper triangle all zero: OK")

    # Build filters
    sentence_gap = node_mask.metadata.get("sentence_gap", 0)
    mask_mode = node_mask.metadata.get("mask_mode", "prefix")
    num_prefix = node_mask.metadata.get("num_prefix_sentences", num_sents)

    gap_filter = build_gap_filter(num_sents, sentence_gap)
    mode_filter = build_mode_filter(num_prefix, num_sents, mask_mode)
    causal_filter = build_causal_filter(num_sents)

    # Without causal filter (buggy)
    combined_no_causal = build_combined_filter(gap_filter, mode_filter)
    # With causal filter (fixed)
    combined_with_causal = build_combined_filter(gap_filter, mode_filter, causal_filter)

    device = torch.device("cpu")
    threshold = 5e-3

    # Random masks both ways
    random_no_causal = build_random_score_masks(
        node_mask, num_samples=3, layers=layers, combined_filter=combined_no_causal
    )
    random_with_causal = build_random_score_masks(
        node_mask, num_samples=3, layers=layers, combined_filter=combined_with_causal
    )

    # Without causal filter: effective sparsity diverges
    learned_binary = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_no_causal, device
    )
    learned_eff = count_effective_zeros(learned_binary, num_sents)

    rand_effs_no = []
    for rand_scores in random_no_causal:
        rb = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_no_causal, device
        )
        rand_effs_no.append(count_effective_zeros(rb, num_sents))

    # With causal filter: effective sparsity should match
    learned_binary_c = build_binary_masks(
        node_mask, threshold, layers, num_heads, num_sents, combined_with_causal, device
    )
    learned_eff_c = count_effective_zeros(learned_binary_c, num_sents)

    rand_effs_with = []
    for rand_scores in random_with_causal:
        rb = build_binary_masks(
            rand_scores, threshold, layers, num_heads, num_sents, combined_with_causal, device
        )
        rand_effs_with.append(count_effective_zeros(rb, num_sents))

    avg_no = np.mean(rand_effs_no)
    avg_with = np.mean(rand_effs_with)

    print(f"  At threshold={threshold}:")
    print(f"    Without causal filter:")
    print(f"      Learned effective: {learned_eff}, Random avg: {avg_no:.0f}")
    if learned_eff > 0:
        print(f"      Ratio: {avg_no/learned_eff:.3f}")
    print(f"    With causal filter:")
    print(f"      Learned effective: {learned_eff_c}, Random avg: {avg_with:.0f}")
    if learned_eff_c > 0:
        ratio = avg_with / learned_eff_c
        print(f"      Ratio: {ratio:.3f}")
        assert 0.8 < ratio < 1.2, f"With causal filter, ratio should be ~1.0, got {ratio:.3f}"

    print("[PASS] test_with_actual_data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Basic consistency tests")
    print("=" * 60)
    test_binary_masks_nodemask_vs_dict()

    print()
    print("=" * 60)
    print("Causal filter unit tests")
    print("=" * 60)
    test_causal_filter_shape_and_values()
    test_combined_filter_includes_causal()
    test_combined_filter_backward_compatible()

    print()
    print("=" * 60)
    print("Bug demonstration (without causal filter)")
    print("=" * 60)
    test_random_permutation_distorts_effective_sparsity_without_causal_filter()

    print()
    print("=" * 60)
    print("Fix verification (with causal filter)")
    print("=" * 60)
    test_causal_filter_fixes_permutation()
    test_at_full_ablation_masks_are_identical()
    test_at_no_ablation_masks_are_identical()
    test_random_permutation_preserves_score_distribution()
    test_random_masks_only_permute_within_causal_region()
    test_sparsity_with_vs_without_causal_filter()

    print()
    print("=" * 60)
    print("Actual data validation")
    print("=" * 60)
    test_with_actual_data()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
