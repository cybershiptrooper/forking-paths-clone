"""Tests for build_prompt_filter (freeze_prompt_sentences support).

The prompt filter freezes every mask entry whose query or key sentence is
one of the first N sentences (the question prompt). Combined with the
gap/mode/causal filters it restricts the learnable pool to
reasoning-to-reasoning attention.

Usage:
    uv run pytest tests/test_prompt_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.masks import (
    NodeMask,
    build_causal_filter,
    build_combined_filter,
    build_gap_filter,
    build_mode_filter,
    build_prompt_filter,
)


def test_prompt_filter_freezes_rows_and_columns():
    pf = build_prompt_filter(3, 6)
    assert pf.shape == (6, 6)
    assert pf[:3, :].all()
    assert pf[:, :3].all()
    assert not pf[3:, 3:].any()


def test_prompt_filter_zero_sentences_freezes_nothing():
    pf = build_prompt_filter(0, 5)
    assert not pf.any()


def test_combined_filter_learnable_pool_is_reasoning_only():
    n_prompt, num_sents = 3, 6
    gap = build_gap_filter(num_sents, 1)
    mode = build_mode_filter(num_sents, num_sents, "prefix")
    causal = build_causal_filter(num_sents)
    prompt = build_prompt_filter(n_prompt, num_sents)
    combined = build_combined_filter(gap, mode, causal, prompt)
    learnable = (~combined).nonzero().tolist()
    # Only reasoning-query -> reasoning-key, strictly below diagonal, gap >= 1
    assert learnable == [[4, 3], [5, 3], [5, 4]]


def test_combined_filter_without_prompt_filter_unchanged():
    num_sents = 6
    gap = build_gap_filter(num_sents, 1)
    mode = build_mode_filter(num_sents, num_sents, "prefix")
    causal = build_causal_filter(num_sents)
    legacy = build_combined_filter(gap, mode, causal)
    explicit_none = build_combined_filter(gap, mode, causal, None)
    assert torch.equal(legacy, explicit_none)


def test_node_mask_metadata_builds_prompt_filter():
    num_sents = 6
    nm = NodeMask(
        model_name="m",
        algorithm="a",
        layers=[0],
        sentences=[{"start": i, "end": i} for i in range(num_sents)],
        objective_name="o",
        metadata={
            "sentence_gap": 1,
            "mask_mode": "prefix",
            "num_prefix_sentences": num_sents,
            "num_frozen_prompt_sentences": 3,
            "mask_granularity": "pair",
        },
        scores=[[0.0] * num_sents for _ in range(num_sents)],
    )
    expected = build_combined_filter(
        build_gap_filter(num_sents, 1),
        build_mode_filter(num_sents, num_sents, "prefix"),
        build_causal_filter(num_sents),
        build_prompt_filter(3, num_sents),
    )
    assert torch.equal(nm._build_combined_filter_from_metadata(), expected)


def test_node_mask_metadata_missing_field_means_no_prompt_filter():
    num_sents = 4
    nm = NodeMask(
        model_name="m",
        algorithm="a",
        layers=[0],
        sentences=[{"start": i, "end": i} for i in range(num_sents)],
        objective_name="o",
        metadata={
            "sentence_gap": 1,
            "mask_mode": "prefix",
            "num_prefix_sentences": num_sents,
            "mask_granularity": "pair",
        },
        scores=[[0.0] * num_sents for _ in range(num_sents)],
    )
    expected = build_combined_filter(
        build_gap_filter(num_sents, 1),
        build_mode_filter(num_sents, num_sents, "prefix"),
        build_causal_filter(num_sents),
    )
    assert torch.equal(nm._build_combined_filter_from_metadata(), expected)
