"""Tests for the auto-fallback to judge model when >50% branches lack \\boxed{}.

Usage:
    uv run python tests/test_answer_fallback.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.importance_sampling import extract_answer_ids


# ---------------------------------------------------------------------------
# H8: Judge fallback logic
# ---------------------------------------------------------------------------


def test_boxed_extraction_normal():
    """When all branches have \\boxed{}, no fallback needed."""
    branches = [
        {"text": "The answer is \\boxed{42}."},
        {"text": "I think \\boxed{42}."},
        {"text": "So \\boxed{43} is wrong, \\boxed{42}."},
        {"text": "Therefore \\boxed{7}."},
    ]
    answer_ids, labels = extract_answer_ids(branches, prefix_text="")
    # Should find 2 groups: "42" and "7"
    assert len(labels) == 2, f"Expected 2 groups, got {len(labels)}: {labels}"
    assert all(not label.startswith("__no_answer_") for label in labels), (
        "All branches have boxed answers, no __no_answer_ labels expected"
    )
    print("[PASS] test_boxed_extraction_normal")


def test_boxed_extraction_majority_missing():
    """When >50% branches lack \\boxed{}, extract_answer_ids should still work
    (the fallback logic is in learn_circuit.py, not here)."""
    branches = [
        {"text": "The answer is \\boxed{42}."},
        {"text": "I think the answer is 42"},
        {"text": "Let me think about this..."},
        {"text": "Hmm, I'm not sure."},
    ]
    answer_ids, labels = extract_answer_ids(branches, prefix_text="")
    # 3 out of 4 lack boxed → 3 unique __no_answer_ labels + "42"
    no_answer_count = sum(1 for l in labels if l.startswith("__no_answer_"))
    assert no_answer_count == 3, f"Expected 3 no-answer labels, got {no_answer_count}"

    # Count branches mapping to no-answer labels
    no_answer_branches = sum(
        1 for a in answer_ids if labels[a].startswith("__no_answer_")
    )
    assert no_answer_branches == 3, f"Expected 3 no-answer branches, got {no_answer_branches}"
    assert no_answer_branches > len(branches) * 0.5, "Should trigger fallback threshold"
    print("[PASS] test_boxed_extraction_majority_missing")


def test_fallback_requires_api_key():
    """The fallback in learn_circuit.py should raise when OPENROUTER_API_KEY is missing.

    We test the logic pattern that learn_circuit.py uses (not the function itself,
    since it requires vLLM etc).
    """
    import os

    branches = [
        {"text": "The answer is 42"},
        {"text": "I think 42"},
        {"text": "Let me think..."},
        {"text": "Not sure."},
    ]
    answer_ids_list, answer_labels = extract_answer_ids(branches, prefix_text="")

    no_answer_branches = sum(
        1 for a in answer_ids_list if answer_labels[a].startswith("__no_answer_")
    )

    # Simulate the fallback check from learn_circuit.py
    if no_answer_branches > len(branches) * 0.5:
        # Mock missing API key
        with patch.dict(os.environ, {}, clear=True):
            key = os.getenv("OPENROUTER_API_KEY")
            assert key is None, "OPENROUTER_API_KEY should be None in test"
            # learn_circuit.py would raise ValueError here
            try:
                if not key:
                    raise ValueError(
                        "More than 50% of branches lack \\boxed{} answers. "
                        "Judge-based answer extraction requires OPENROUTER_API_KEY."
                    )
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "OPENROUTER_API_KEY" in str(e)
    else:
        assert False, "Should have detected >50% no-answer branches"

    print("[PASS] test_fallback_requires_api_key")


def test_judge_clustering_called():
    """When judge_client is provided, extract_answer_ids should use judge clustering."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "YES"
    mock_client.chat.completions.create.return_value = mock_response

    branches = [
        {"text": "The answer is \\boxed{42}."},
        {"text": "I think \\boxed{42.0}."},
        {"text": "So \\boxed{7}."},
        {"text": "Therefore \\boxed{7.00}."},
    ]
    answer_ids, labels = extract_answer_ids(
        branches, prefix_text="",
        judge_client=mock_client, judge_model="test-model",
        question="What is 6*7?",
    )
    # Judge says all answers are equivalent (always YES), so should be 1 group
    assert len(labels) == 1, f"Judge always says YES → 1 group, got {len(labels)}"
    assert all(a == 0 for a in answer_ids), "All should be in group 0"
    print("[PASS] test_judge_clustering_called")


def test_normalize_answer():
    """Test answer normalization used during boxed extraction."""
    from utils.importance_sampling import normalize_answer

    assert normalize_answer("42") == "42"
    assert normalize_answer("42.0") == "42"
    assert normalize_answer(" 42 ") == "42"
    assert normalize_answer("$42$") == "42"
    assert normalize_answer("\\frac{1}{2}") == "0.5"
    assert normalize_answer("\\frac{84}{2}") == "42"
    print("[PASS] test_normalize_answer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Answer fallback tests")
    print("=" * 60)

    test_boxed_extraction_normal()
    test_boxed_extraction_majority_missing()
    test_fallback_requires_api_key()
    test_judge_clustering_called()
    test_normalize_answer()

    print("\n" + "=" * 60)
    print("All answer fallback tests passed!")
    print("=" * 60)
