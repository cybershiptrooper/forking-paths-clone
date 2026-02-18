"""Visualize and test expand_sentence_mask_to_tokens.

Usage:
    uv run python tests/test_expand_sentence_mask.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path (same pattern as expts/ scripts)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from utils.circuit_discovery.nodewise_attribution import expand_sentence_mask_to_tokens
from utils.masks import build_gap_filter, apply_gap_filter, build_mode_filter, build_combined_filter


# ---------------------------------------------------------------------------
# Naive reference implementation (readable, uses for-loops)
# ---------------------------------------------------------------------------
def expand_sentence_mask_to_tokens_naive(
    mask: torch.Tensor,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position=None,
) -> torch.Tensor:
    """Expand sentence mask to token mask using explicit loops.

    Meant to be easy to read; not differentiable or fast.
    """
    num_heads, num_sents, _ = mask.shape

    # Determine which sentence each query/key token belongs to
    if cache_position is not None:
        q_sent = token_to_sent[cache_position.long()]
    else:
        q_sent = token_to_sent[:q_len]
    k_sent = token_to_sent[:k_len]

    # Apply gap filter to the sentence-level mask
    effective_mask = apply_gap_filter(mask, gap_filter, fill_value=1.0)

    # Build token mask element-by-element
    token_mask = torch.ones(num_heads, q_len, k_len, dtype=mask.dtype)
    for h in range(num_heads):
        for i in range(q_len):
            for j in range(k_len):
                qs = q_sent[i].item()
                ks = k_sent[j].item()
                if qs == -1 or ks == -1:
                    # Token not in any sentence → passthrough (1.0)
                    token_mask[h, i, j] = 1.0
                else:
                    token_mask[h, i, j] = effective_mask[h, qs, ks]
    return token_mask


# ---------------------------------------------------------------------------
# Setup: build a small toy example
# ---------------------------------------------------------------------------
def build_toy_example():
    """Construct a toy scenario with 3 sentences and 12 tokens.

    Token layout (0-indexed):
      0     : BOS (no sentence, index -1)
      1-3   : sentence 0
      4-7   : sentence 1
      8-11  : sentence 2
    """
    num_heads = 1
    num_sents = 3
    total_seq_len = 20

    # Sentence-level mask: give each head a distinct pattern
    torch.manual_seed(42)
    mask = torch.rand(num_heads, num_sents, num_sents)

    # Token-to-sentence mapping
    token_to_sent = torch.full((total_seq_len,), -1, dtype=torch.long)
    token_to_sent[1:4] = 0  # sentence 0
    token_to_sent[4:8] = 1  # sentence 1
    token_to_sent[8:12] = 2  # sentence 2

    gap_filter = build_gap_filter(num_sents, sentence_gap=1)

    return mask, token_to_sent, gap_filter, total_seq_len, num_heads, num_sents


# ---------------------------------------------------------------------------
# Test: compare vectorized vs naive
# ---------------------------------------------------------------------------
def test_matches_naive():
    mask, token_to_sent, gap_filter, total_seq_len, *_ = build_toy_example()
    q_len = total_seq_len
    k_len = total_seq_len

    result = expand_sentence_mask_to_tokens(mask, token_to_sent, gap_filter, q_len, k_len)
    expected = expand_sentence_mask_to_tokens_naive(
        mask, token_to_sent, gap_filter, q_len, k_len
    )
    assert torch.allclose(
        result, expected, atol=1e-6
    ), f"Mismatch!\nmax diff = {(result - expected).abs().max().item()}"
    print("[PASS] Full-sequence: vectorized matches naive")


def test_matches_naive_with_cache_position():
    """Simulate KV-cached generation where q_len < total_seq_len."""
    mask, token_to_sent, gap_filter, total_seq_len, *_ = build_toy_example()

    # Pretend we're generating token 10 (q_len=1), full cache (k_len=11)
    q_len = 1
    k_len = 11
    cache_position = torch.tensor([10])

    result = expand_sentence_mask_to_tokens(
        mask, token_to_sent, gap_filter, q_len, k_len, cache_position
    )
    expected = expand_sentence_mask_to_tokens_naive(
        mask, token_to_sent, gap_filter, q_len, k_len, cache_position
    )
    assert torch.allclose(
        result, expected, atol=1e-6
    ), f"Mismatch!\nmax diff = {(result - expected).abs().max().item()}"
    print("[PASS] Cache-position: vectorized matches naive")


def test_unassigned_tokens_passthrough():
    """All mask values for BOS (token 0, sent=-1) should be 1.0."""
    mask, token_to_sent, gap_filter, total_seq_len, *_ = build_toy_example()
    q_len = total_seq_len
    k_len = total_seq_len

    result = expand_sentence_mask_to_tokens(mask, token_to_sent, gap_filter, q_len, k_len)
    # Row 0 (BOS as query) should be all 1s
    assert (result[:, 0, :] == 1.0).all(), "BOS query row should be all 1.0"
    # Column 0 (BOS as key) should be all 1s
    assert (result[:, :, 0] == 1.0).all(), "BOS key column should be all 1.0"
    print("[PASS] Unassigned tokens pass through as 1.0")


def test_gap_filter_passthrough():
    """Sentence pairs within the gap should have mask value 1.0."""
    mask, token_to_sent, gap_filter, total_seq_len, *_ = build_toy_example()
    q_len = total_seq_len
    k_len = total_seq_len

    result = expand_sentence_mask_to_tokens(mask, token_to_sent, gap_filter, q_len, k_len)
    # With gap=1, diagonal sentence pairs (same sentence) should be 1.0
    # Tokens 1-3 are sent 0; query in sent 0 attending to key in sent 0 → 1.0
    assert (result[:, 1:4, 1:4] == 1.0).all(), "Same-sentence attention should be 1.0"
    print("[PASS] Gap-filtered (same-sentence) entries are 1.0")


def test_differentiable():
    """Gradients should flow through to the sentence-level mask."""
    mask, token_to_sent, gap_filter, total_seq_len, *_ = build_toy_example()
    mask = mask.requires_grad_(True)
    q_len = total_seq_len
    k_len = total_seq_len

    result = expand_sentence_mask_to_tokens(mask, token_to_sent, gap_filter, q_len, k_len)
    loss = result.sum()
    loss.backward()

    assert mask.grad is not None, "Gradient should not be None"
    # Gap-filtered entries should have zero gradient (they're replaced by constants)
    # Non-gap entries should have non-zero gradient
    non_gap = ~gap_filter
    assert (
        mask.grad[:, non_gap].abs() > 0
    ).any(), "Non-gap entries should receive gradient"
    print("[PASS] Gradients flow to sentence-level mask")


# ---------------------------------------------------------------------------
# Tests: mode filter
# ---------------------------------------------------------------------------
def test_mode_filter_prefix():
    """In 'prefix' mode, only prefix-to-prefix entries are learnable."""
    num_prefix = 3
    num_total = 5  # 3 prefix + 2 generation
    frozen = build_mode_filter(num_prefix, num_total, "prefix")
    # Top-left 3x3 should be False (learnable)
    assert not frozen[:num_prefix, :num_prefix].any(), "prefix→prefix should be learnable"
    # Everything else should be True (frozen)
    assert frozen[num_prefix:, :].all(), "gen rows should be frozen in prefix mode"
    assert frozen[:, num_prefix:].all(), "gen-key cols should be frozen"
    print("[PASS] mode_filter prefix")


def test_mode_filter_generation():
    """In 'generation' mode, only gen-query → prefix-key entries are learnable."""
    num_prefix = 3
    num_total = 5
    frozen = build_mode_filter(num_prefix, num_total, "generation")
    # Bottom-left block (gen rows, prefix cols) should be False (learnable)
    assert not frozen[num_prefix:, :num_prefix].any(), "gen→prefix should be learnable"
    # Prefix rows should be frozen
    assert frozen[:num_prefix, :].all(), "prefix rows should be frozen in generation mode"
    # Gen-key columns should be frozen
    assert frozen[:, num_prefix:].all(), "gen-key cols should be frozen"
    print("[PASS] mode_filter generation")


def test_mode_filter_both():
    """In 'both' mode, all-query → prefix-key entries are learnable."""
    num_prefix = 3
    num_total = 5
    frozen = build_mode_filter(num_prefix, num_total, "both")
    # Entire left block (all rows, prefix cols) should be False (learnable)
    assert not frozen[:, :num_prefix].any(), "all→prefix should be learnable"
    # Gen-key columns should still be frozen
    assert frozen[:, num_prefix:].all(), "gen-key cols should be frozen"
    print("[PASS] mode_filter both")


def test_combined_filter_blocks_gradients():
    """Gradients should only flow through unfrozen entries in combined filter."""
    num_heads = 2
    num_prefix = 3
    num_total = 5  # 3 prefix + 2 generation

    gap_filter = build_gap_filter(num_total, sentence_gap=1)
    mode_filter = build_mode_filter(num_prefix, num_total, "generation")
    combined = build_combined_filter(gap_filter, mode_filter)

    # Build a mask and compute gradients
    mask = torch.rand(num_heads, num_total, num_total, requires_grad=True)
    token_to_sent = torch.full((20,), -1, dtype=torch.long)
    # Prefix sentences: 0-2
    token_to_sent[1:4] = 0
    token_to_sent[4:7] = 1
    token_to_sent[7:10] = 2
    # Generation sentences: 3-4
    token_to_sent[10:14] = 3
    token_to_sent[14:18] = 4

    result = expand_sentence_mask_to_tokens(mask, token_to_sent, combined, 18, 18)
    result.sum().backward()

    # Frozen entries should have zero gradient
    frozen_2d = combined.unsqueeze(0).expand(num_heads, -1, -1)
    assert (mask.grad[frozen_2d] == 0).all(), "Frozen entries should have zero gradient"
    # Unfrozen gen→prefix entries should have non-zero gradient
    unfrozen = ~combined
    assert (mask.grad[:, unfrozen].abs() > 0).any(), "Unfrozen entries should receive gradient"
    print("[PASS] combined_filter blocks gradients correctly")


def test_qsent_indexing_with_generation_sentences():
    """Verify q_sent uses [:q_len] so generation tokens get correct sentence indices."""
    num_sents = 4  # 2 prefix + 2 generation
    total_map_len = 20

    token_to_sent = torch.full((total_map_len,), -1, dtype=torch.long)
    token_to_sent[0:5] = 0   # prefix sent 0
    token_to_sent[5:10] = 1  # prefix sent 1
    token_to_sent[10:15] = 2 # gen sent 0
    token_to_sent[15:20] = 3 # gen sent 1

    mask = torch.rand(1, num_sents, num_sents)
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)

    # Full forward: q_len = k_len = 18 (< total_map_len)
    q_len = 18
    k_len = 18
    result = expand_sentence_mask_to_tokens(mask, token_to_sent, gap_filter, q_len, k_len)

    # Token 12 is in gen sent 2, token 6 is in prefix sent 1
    # result[0, 12, 6] should be mask[0, 2, 1]
    expected_val = mask[0, 2, 1].item()
    actual_val = result[0, 12, 6].item()
    assert abs(actual_val - expected_val) < 1e-6, (
        f"Gen query → prefix key should use mask[0,2,1]={expected_val}, got {actual_val}"
    )
    print("[PASS] q_sent indexing correct for generation sentences")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize():
    mask, token_to_sent, gap_filter, total_seq_len, num_heads, num_sents = (
        build_toy_example()
    )
    q_len = total_seq_len
    k_len = total_seq_len

    token_mask = expand_sentence_mask_to_tokens(
        mask, token_to_sent, gap_filter, q_len, k_len
    )
    naive_mask = expand_sentence_mask_to_tokens_naive(
        mask, token_to_sent, gap_filter, q_len, k_len
    )
    effective_mask = apply_gap_filter(mask, gap_filter, fill_value=1.0)

    # Token labels: anything with sent=-1 is labelled "BOS"
    token_labels = []
    for i in range(total_seq_len):
        s = token_to_sent[i].item()
        if s == -1:
            token_labels.append(f"BOS:t{i}")
        else:
            token_labels.append(f"S{s}:t{i}")
    sent_labels = [f"S{i}" for i in range(num_sents)]

    fig = plt.figure(figsize=(20, 5 * num_heads))
    outer_gs = gridspec.GridSpec(num_heads, 1, figure=fig, hspace=0.4)

    for h in range(num_heads):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer_gs[h], wspace=0.35
        )

        # 1) Raw sentence mask
        ax0 = fig.add_subplot(inner_gs[0])
        im0 = ax0.imshow(mask[h].detach().numpy(), vmin=0, vmax=1, cmap="viridis")
        ax0.set_xticks(range(num_sents))
        ax0.set_xticklabels(sent_labels)
        ax0.set_yticks(range(num_sents))
        ax0.set_yticklabels(sent_labels)
        ax0.set_xlabel("Key sentence")
        ax0.set_ylabel("Query sentence")
        ax0.set_title(f"Head {h}: Raw sentence mask")
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        # 2) Effective mask (after gap filter)
        ax1 = fig.add_subplot(inner_gs[1])
        im1 = ax1.imshow(
            effective_mask[h].detach().numpy(), vmin=0, vmax=1, cmap="viridis"
        )
        ax1.set_xticks(range(num_sents))
        ax1.set_xticklabels(sent_labels)
        ax1.set_yticks(range(num_sents))
        ax1.set_yticklabels(sent_labels)
        ax1.set_xlabel("Key sentence")
        ax1.set_ylabel("Query sentence")
        ax1.set_title(f"Head {h}: After gap filter (gap=1)")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # 3) Naive token mask (for-loop reference)
        ax2 = fig.add_subplot(inner_gs[2])
        im2 = ax2.imshow(naive_mask[h].detach().numpy(), vmin=0, vmax=1, cmap="viridis")
        ax2.set_xticks(range(total_seq_len))
        ax2.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax2.set_yticks(range(total_seq_len))
        ax2.set_yticklabels(token_labels, fontsize=7)
        ax2.set_xlabel("Key token")
        ax2.set_ylabel("Query token")
        ax2.set_title(f"Head {h}: Naive (for-loop)")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # 4) Vectorized token mask
        ax3 = fig.add_subplot(inner_gs[3])
        im3 = ax3.imshow(token_mask[h].detach().numpy(), vmin=0, vmax=1, cmap="viridis")
        ax3.set_xticks(range(total_seq_len))
        ax3.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax3.set_yticks(range(total_seq_len))
        ax3.set_yticklabels(token_labels, fontsize=7)
        ax3.set_xlabel("Key token")
        ax3.set_ylabel("Query token")
        ax3.set_title(f"Head {h}: Vectorized")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

    fig.suptitle(
        "expand_sentence_mask_to_tokens: sentence → token mask expansion",
        fontsize=14,
        y=1.01,
    )
    plt.savefig("tests/rough_expand_mask_viz.png", dpi=150, bbox_inches="tight")
    print(f"Saved visualization to tests/expand_mask_viz.png")
    # plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Tests")
    print("=" * 60)
    test_matches_naive()
    test_matches_naive_with_cache_position()
    test_unassigned_tokens_passthrough()
    test_gap_filter_passthrough()
    test_differentiable()

    print()
    print("=" * 60)
    print("Mode filter tests")
    print("=" * 60)
    test_mode_filter_prefix()
    test_mode_filter_generation()
    test_mode_filter_both()
    test_combined_filter_blocks_gradients()
    test_qsent_indexing_with_generation_sentences()

    print()
    print("=" * 60)
    print("Visualization")
    print("=" * 60)
    visualize()
