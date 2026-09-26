"""Answer grading for sampled continuations, shared by the scan, the
bank builder, and the masked-rollout evaluation.

Regex-based (no LLM parser) so the *same* grading applies at bank-build
time and at eval time.  Validate against the collection pipeline's
LLM-parsed labels with ``validate_extractor.py`` before trusting it on a
new dataset.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.rewards import extract_boxed

THINK_END = "</think>"

# Ordered patterns for a multiple-choice letter in the answer section.
_LETTER_PATTERNS = [
    # "answer is (C" / "answer: C" / "Answer – C" — allow a few filler
    # chars (colon, bold markers, parenthesis) between "answer" and letter
    re.compile(r"answer(?:\s+is)?\s*[:\-–]?\s*\*{0,2}\(?\s*([A-E])\b", re.IGNORECASE),
    # "**C)** ..." or "**C.**" bold-leading letter
    re.compile(r"\*\*\(?([A-E])[\)\.\:]"),
    # bare "(C)" as a standalone token
    re.compile(r"(?<![A-Za-z0-9])\(([A-E])\)(?![A-Za-z0-9])"),
]


def extract_letter(text: str, letters: str = "ABCDE") -> Optional[str]:
    """Extract a multiple-choice letter from an answer section.

    Priority: \\boxed{...} content, then the regex patterns above (first
    match wins), then None.
    """
    boxed = extract_boxed(text)
    if boxed:
        b = boxed.strip().strip("()*. ")
        # "\boxed{C}" or "\boxed{C) 16 hrs}"
        if b[:1].upper() in letters and (len(b) == 1 or not b[1:2].isalnum()):
            return b[0].upper()
    for pat in _LETTER_PATTERNS:
        m = pat.search(text)
        if m and m.group(1).upper() in letters:
            return m.group(1).upper()
    return None


def _normalize_free_answer(s: str) -> str:
    s = " ".join(s.split())
    s = s.rstrip(".").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.strip("$")
    return s


def extract_free_answer(text: str) -> Optional[str]:
    """Extract an open-ended (e.g. MATH) answer: \\boxed content only."""
    boxed = extract_boxed(text)
    if boxed is None:
        return None
    return _normalize_free_answer(boxed)


def grade_continuation(
    continuation_text: str,
    correct_letter: Optional[str],
    correct_answer: Optional[str],
    dataset_type: str,
    answer_letters: str = "ABCDE",
) -> dict:
    """Grade one sampled continuation for termination + correctness.

    Returns a dict with:
        terminated:   bool — ``</think>`` appears in the continuation
        answer:       extracted answer (letter or normalized string) or None
        correct:      bool — answer matches gold (False when unparseable)
        cluster_id:   0 = terminated & correct, 1 = terminated & not
                      correct (wrong or unparseable), 2 = not terminated
        grade_method: how the answer was extracted
    """
    terminated = THINK_END in continuation_text
    answer = None
    correct = False
    method = "none"
    if terminated:
        section = continuation_text.split(THINK_END, 1)[1]
        if dataset_type == "multiple choice" and correct_letter:
            answer = extract_letter(section, answer_letters)
            method = "letter_regex" if answer else "unparsed"
            correct = answer is not None and answer == correct_letter.strip().upper()
        else:
            answer = extract_free_answer(section)
            method = "boxed" if answer else "unparsed"
            gold = _normalize_free_answer(correct_answer or "")
            correct = answer is not None and gold != "" and answer == gold
    cluster_id = 0 if (terminated and correct) else (1 if terminated else 2)
    return {
        "terminated": terminated,
        "answer": answer,
        "correct": bool(correct),
        "cluster_id": cluster_id,
        "grade_method": method,
    }
