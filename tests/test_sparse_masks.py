"""Tests for sparse NodeMask serialization.

Tests cover:
- H6: Sparse roundtrip (to_json sparse → from_json → scores match)
- H7: Backward compatibility (dense files load correctly)
- Size comparison (sparse < dense)
- All three granularities (head, layer, pair)

Usage:
    uv run python tests/test_sparse_masks.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.masks import NodeMask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node_mask(granularity: str = "head", num_sents: int = 6, num_layers: int = 2,
                   num_heads: int = 4, sentence_gap: int = 1):
    """Create a synthetic NodeMask for testing."""
    import random
    random.seed(42)

    sentences = [
        {"start": i * 10, "end": i * 10 + 9, "text": f"Sentence {i}"}
        for i in range(num_sents)
    ]
    layers = list(range(num_layers))

    metadata = {
        "mask_granularity": granularity,
        "sentence_gap": sentence_gap,
        "mask_mode": "prefix",
        "num_prefix_sentences": num_sents,
        "num_heads": num_heads,
    }

    if granularity == "head":
        scores = {}
        for layer in layers:
            scores[layer] = {}
            for head in range(num_heads):
                matrix = [[0.0] * num_sents for _ in range(num_sents)]
                # Only fill lower triangle (causal) with non-gap entries
                for i in range(num_sents):
                    for j in range(i):
                        if abs(i - j) > sentence_gap:
                            matrix[i][j] = random.uniform(-0.01, 0.01)
                scores[layer][head] = matrix
    elif granularity == "layer":
        scores = {}
        for layer in layers:
            matrix = [[0.0] * num_sents for _ in range(num_sents)]
            for i in range(num_sents):
                for j in range(i):
                    if abs(i - j) > sentence_gap:
                        matrix[i][j] = random.uniform(-0.01, 0.01)
            scores[layer] = matrix
    else:  # "pair"
        scores = [[0.0] * num_sents for _ in range(num_sents)]
        for i in range(num_sents):
            for j in range(i):
                if abs(i - j) > sentence_gap:
                    scores[i][j] = random.uniform(-0.01, 0.01)

    return NodeMask(
        model_name="test-model",
        algorithm="test-algo",
        layers=layers,
        sentences=sentences,
        objective_name="kl_divergence",
        metadata=metadata,
        scores=scores,
    )


def scores_equal(scores_a, scores_b, granularity: str, tol: float = 1e-10,
                  combined_filter=None) -> bool:
    """Check if two score structures are equal at active positions.

    If *combined_filter* is provided, only positions where
    ``combined_filter[i][j] == False`` are compared.  Filtered positions
    may differ (original has 0.0, loaded has fill_value 1.0).
    """
    def _check_matrix(a, b, num_sents):
        for i in range(num_sents):
            for j in range(num_sents):
                if combined_filter is not None and bool(combined_filter[i, j]):
                    continue  # filtered position, skip
                if abs(a[i][j] - b[i][j]) > tol:
                    return False
        return True

    if granularity == "head":
        for layer in scores_a:
            if layer not in scores_b:
                return False
            for head in scores_a[layer]:
                if head not in scores_b[layer]:
                    return False
                ns = len(scores_a[layer][head])
                if not _check_matrix(scores_a[layer][head], scores_b[layer][head], ns):
                    return False
    elif granularity == "layer":
        for layer in scores_a:
            if layer not in scores_b:
                return False
            ns = len(scores_a[layer])
            if not _check_matrix(scores_a[layer], scores_b[layer], ns):
                return False
    else:  # "pair"
        ns = len(scores_a)
        if not _check_matrix(scores_a, scores_b, ns):
            return False
    return True


# ---------------------------------------------------------------------------
# H6: Sparse roundtrip
# ---------------------------------------------------------------------------


def test_sparse_roundtrip_head():
    """Sparse roundtrip for head granularity."""
    mask = make_node_mask(granularity="head")
    cf = mask._build_combined_filter_from_metadata()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mask.to_json(path, sparse=True)
        loaded = NodeMask.from_json(path)

        # Check scores_format is in the JSON
        with open(path) as f:
            data = json.load(f)
        assert data.get("scores_format") == "sparse", "scores_format should be 'sparse'"

        # Verify active positions match
        assert scores_equal(mask.scores, loaded.scores, "head", combined_filter=cf), (
            "Sparse roundtrip failed for head granularity"
        )
        # Verify filtered positions are filled with 1.0 (fill_value)
        for layer in loaded.scores:
            for head in loaded.scores[layer]:
                for i in range(len(loaded.sentences)):
                    # Diagonal should be fill_value (1.0) since it's filtered
                    assert loaded.scores[layer][head][i][i] == 1.0, (
                        f"Diagonal [{i}][{i}] should be 1.0 (filtered)"
                    )
    finally:
        os.unlink(path)

    print("[PASS] test_sparse_roundtrip_head")


def test_sparse_roundtrip_layer():
    """Sparse roundtrip for layer granularity."""
    mask = make_node_mask(granularity="layer")
    cf = mask._build_combined_filter_from_metadata()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mask.to_json(path, sparse=True)
        loaded = NodeMask.from_json(path)
        assert scores_equal(mask.scores, loaded.scores, "layer", combined_filter=cf), (
            "Sparse roundtrip failed for layer granularity"
        )
    finally:
        os.unlink(path)

    print("[PASS] test_sparse_roundtrip_layer")


def test_sparse_roundtrip_pair():
    """Sparse roundtrip for pair granularity."""
    mask = make_node_mask(granularity="pair")
    cf = mask._build_combined_filter_from_metadata()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mask.to_json(path, sparse=True)
        loaded = NodeMask.from_json(path)
        assert scores_equal(mask.scores, loaded.scores, "pair", combined_filter=cf), (
            "Sparse roundtrip failed for pair granularity"
        )
    finally:
        os.unlink(path)

    print("[PASS] test_sparse_roundtrip_pair")


# ---------------------------------------------------------------------------
# H6: Size comparison
# ---------------------------------------------------------------------------


def test_sparse_size_comparison():
    """Report sparse vs dense sizes — sparse saves space only with high filter ratios."""
    mask = make_node_mask(granularity="pair", num_sents=20, sentence_gap=3)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        sparse_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        dense_path = f.name
    try:
        mask.to_json(sparse_path, sparse=True)
        mask.to_json(dense_path, sparse=False)

        sparse_size = os.path.getsize(sparse_path)
        dense_size = os.path.getsize(dense_path)

        ratio = sparse_size / dense_size
        print(f"  Pair (S=20, gap=3): sparse={sparse_size}B, "
              f"dense={dense_size}B, ratio={ratio:.2%}")

        # Both should produce valid, loadable files
        loaded_sparse = NodeMask.from_json(sparse_path)
        loaded_dense = NodeMask.from_json(dense_path)
        cf = mask._build_combined_filter_from_metadata()
        assert scores_equal(
            loaded_sparse.scores, loaded_dense.scores, "pair", combined_filter=cf
        ), "Sparse and dense should produce equivalent scores at active positions"
    finally:
        os.unlink(sparse_path)
        os.unlink(dense_path)

    print("[PASS] test_sparse_size_comparison")


# ---------------------------------------------------------------------------
# H7: Backward compatibility — dense files load correctly
# ---------------------------------------------------------------------------


def test_dense_roundtrip():
    """Dense format (sparse=False) should roundtrip correctly."""
    mask = make_node_mask(granularity="head")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        mask.to_json(path, sparse=False)

        # Verify no scores_format in JSON (legacy)
        with open(path) as f:
            data = json.load(f)
        assert "scores_format" not in data, "Dense format should not have scores_format"

        loaded = NodeMask.from_json(path)
        assert scores_equal(mask.scores, loaded.scores, "head"), (
            "Dense roundtrip failed"
        )
    finally:
        os.unlink(path)

    print("[PASS] test_dense_roundtrip")


def test_dense_backward_compat():
    """A manually constructed dense JSON (simulating legacy file) should load."""
    data = {
        "mask_type": "NodeMask",
        "model_name": "test",
        "algorithm": "test",
        "layers": [0],
        "sentences": [
            {"start": 0, "end": 4, "text": "S0"},
            {"start": 5, "end": 9, "text": "S1"},
            {"start": 10, "end": 14, "text": "S2"},
        ],
        "objective_name": "kl_divergence",
        "metadata": {"mask_granularity": "pair"},
        "scores": [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.3, 0.7, 0.0],
        ],
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(data, f)
        path = f.name
    try:
        loaded = NodeMask.from_json(path)
        assert loaded.scores[1][0] == 0.5
        assert loaded.scores[2][1] == 0.7
        assert loaded.granularity == "pair"
    finally:
        os.unlink(path)

    print("[PASS] test_dense_backward_compat")


def test_existing_mask_file():
    """Try to load an existing mask file from results/ if available."""
    from glob import glob
    masks = glob("results/circuitviz/**/*.json", recursive=True)
    if not masks:
        print("[SKIP] No existing mask files found in results/circuitviz/")
        return

    path = masks[0]
    try:
        loaded = NodeMask.from_json(path)
        assert loaded.scores is not None
        assert len(loaded.sentences) > 0
        print(f"  Loaded {path}: {loaded.granularity}, {len(loaded.layers)} layers, "
              f"{len(loaded.sentences)} sentences")
    except Exception as e:
        print(f"[FAIL] Could not load {path}: {e}")
        raise

    print("[PASS] test_existing_mask_file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Sparse mask tests")
    print("=" * 60)

    print("\n--- H6: Sparse roundtrip ---")
    test_sparse_roundtrip_head()
    test_sparse_roundtrip_layer()
    test_sparse_roundtrip_pair()

    print("\n--- H6: Size comparison ---")
    test_sparse_size_comparison()

    print("\n--- H7: Backward compatibility ---")
    test_dense_roundtrip()
    test_dense_backward_compat()
    test_existing_mask_file()

    print("\n" + "=" * 60)
    print("All sparse mask tests passed!")
    print("=" * 60)
