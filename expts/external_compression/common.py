"""Shared construction code for the external CoT-compression experiment.

Replicates the construction of task 9 ("compressing reasoning traces") from
github.com/Centrattic/cot-proxy-tasks exactly:

- sentence splitting: ``re.split(r'(?<=[.!?])\\s+|(?<=[.!?])$|\\n\\n+', text)``
  (their ``src/tasks/forced_response/utils.py``);
- user message: question + lettered choices + "Answer with just the letter";
- prompt: chat template (system = "You are a helpful assistant.",
  ``add_generation_prompt=True``) + ``"<think>\\n"`` + CoT prefix
  (their ``src/utils/chat_template.py``);
- all prefix reconstructions join sentences with a single space
  (their ``CompressionSpec.full_prefix`` / ``reconstruct``);
- answer probe: CoT prefix + ``" So, the answer is: "`` + ``"</think>\\n"``,
  answer read as the next-token distribution over bare letter tokens,
  renormalised (their ``CompressedCotTask.get_choice_distribution``).

Everything here is CPU-side and deterministic; model-side code lives in the
sibling scripts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# EXTCOMP_* env vars exist for sandboxed smoke tests (small model, scratch
# dirs) — production runs use the defaults.
DATA_DIR = os.environ.get(
    "EXTCOMP_DATA_DIR", os.path.join(REPO_ROOT, "data", "external_compression")
)
SPEC_DIR = os.path.join(DATA_DIR, "question_specs")
ROLLOUT_DIR = os.path.join(DATA_DIR, "rollouts")
RESULTS_DIR = os.environ.get(
    "EXTCOMP_RESULTS_DIR", os.path.join(REPO_ROOT, "results", "external_compression")
)

MODEL_NAME = os.environ.get("EXTCOMP_MODEL", "Qwen/Qwen3-32B")
GEN_TEMPERATURE = 0.7          # their forced_response/task.py DEFAULT_TEMPERATURE
GEN_MAX_TOKENS = 16384         # their forced_response/task.py:296
SYSTEM_MSG = "You are a helpful assistant."
ANCHOR = " So, the answer is: "   # their compressed_cot/task.py:174 (nonempty-prefix case)
PROBE_TAIL = "</think>\n"          # appended after the anchor, before the letter
K_KEEP = 5                        # protected tail: last 5 sentences of the prefix
M_VALUES = [3, 5, 10, 15, 20]
DELETION_KL_THRESHOLD = 0.1
BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 10**9)]
BUCKET_NAMES = ["lt50", "50-100", "100-200", "200plus"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$|\n\n+")


def split_cot_into_sentences(text: str) -> List[str]:
    """Their sentence splitter, verbatim (forced_response/utils.py:17-28)."""
    text = text.strip()
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if s and s.strip()]


def build_user_msg(question_text: str, choices: List[str]) -> str:
    """Their _user_msg (compressed_cot/task.py:263-271), multiple-choice branch."""
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    choices_block = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    labels_str = (
        ", ".join(labels[:-1]) + f", or {labels[-1]}"
        if len(labels) > 2
        else " or ".join(labels)
    )
    return (
        f"{question_text}\n\n{choices_block}\n\n"
        f"Answer with just the letter ({labels_str})."
    )


def build_prompt_str(tokenizer, user_msg: str) -> str:
    """Their build_thinking_prompt with empty cot_prefix: chat template + '<think>\\n'."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    # Qwen3's template must not have opened a think block itself.
    assert "<think>" not in prompt, "chat template already contains <think>"
    return prompt + "<think>\n"


def letter_token_ids(tokenizer, letters: List[str]) -> List[int]:
    """Their _resolve_choice_token_ids: encode(letter)[-1] per bare letter."""
    return [tokenizer.encode(c, add_special_tokens=False)[-1] for c in letters]


def load_spec(question_id: str) -> dict:
    with open(os.path.join(SPEC_DIR, f"{question_id}.json")) as f:
        return json.load(f)


def load_rollout(question_id: str) -> dict:
    with open(os.path.join(ROLLOUT_DIR, f"{question_id}.json")) as f:
        return json.load(f)


def list_spec_ids() -> List[str]:
    return sorted(
        f[: -len(".json")] for f in os.listdir(SPEC_DIR) if f.endswith(".json")
    )


# ---------------------------------------------------------------------------
# Forced-input construction
# ---------------------------------------------------------------------------

@dataclass
class Sent:
    """Token span (inclusive) — mirrors utils.utils.Sentence fields."""
    start: int
    end: int


def forced_text(prompt_str: str, sentences: List[str]) -> str:
    """prompt + ' '.join(sentences) + anchor + '</think>\\n' (their layout)."""
    prefix = prompt_str + " ".join(sentences)
    return prefix + ANCHOR + PROBE_TAIL


def encode_forced(tokenizer, prompt_str: str, sentences: List[str]) -> List[int]:
    """Tokenize the whole forced string at once (their encode of prompt_str)."""
    return tokenizer.encode(
        forced_text(prompt_str, sentences), add_special_tokens=False,
    )


def build_masking_input(
    tokenizer,
    prompt_str: str,
    sentences_n: List[str],
    num_mapped: int,
) -> Tuple[List[int], int, List[Sent]]:
    """Tokenize prompt + N joined sentences + probe; compute sentence spans.

    Returns ``(ids, prefix_len, spans)`` where

    - ``ids`` is the full forced token sequence (prefix + anchor + tail),
      tokenized as ONE string so it matches ``encode_forced`` exactly;
    - ``prefix_len`` is the number of tokens covering
      ``prompt_str + ' '.join(sentences_n)``;
    - ``spans`` = [prompt block span] + spans of the first ``num_mapped``
      sentences.  The remaining sentences (the protected tail) and the probe
      are deliberately NOT in the list — their tokens map to the sentinel in
      the token-level mask expansion, so they are always readable and their
      query rows are never masked (same as the base experiment's context
      sentences and probe suffix).

    Token→sentence assignment: a token belongs to the region containing its
    LAST character, so the single joining space before sentence j+1 (which
    BPE attaches to the following token) lands in sentence j+1.
    """
    text = forced_text(prompt_str, sentences_n)
    prefix_char_len = len(prompt_str) + len(" ".join(sentences_n))

    enc = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Char start of each sentence within `text`.
    starts: List[int] = []
    pos = len(prompt_str)
    for s in sentences_n:
        starts.append(pos)
        pos += len(s) + 1  # +1 for the joining space
    # The space BEFORE sentence j is at starts[j] - 1; assign it to sentence j.
    region_starts = [0] + [max(st - 1, 0) for st in starts]  # region 0 = prompt

    def region_of(char_pos: int) -> int:
        """Index into region_starts: 0 = prompt block, j+1 = sentence j."""
        lo, hi = 0, len(region_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if region_starts[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    prefix_len = 0
    token_region: List[int] = []
    for i, (a, b) in enumerate(offsets):
        last_char = max(b - 1, a)
        token_region.append(region_of(last_char))
        if b <= prefix_char_len:
            prefix_len = i + 1

    # Sanity: no token may straddle the prefix/probe boundary.
    a, b = offsets[prefix_len]
    assert a >= prefix_char_len - 1, (
        f"token {prefix_len} straddles the prefix/probe boundary: ({a},{b}) "
        f"vs prefix_char_len={prefix_char_len}"
    )

    # Build contiguous spans for regions 0..num_mapped (prompt + mapped sents).
    spans: List[Sent] = []
    for r in range(0, num_mapped + 1):
        tok_idx = [i for i, tr in enumerate(token_region) if tr == r and i < prefix_len]
        assert tok_idx, f"region {r} has no tokens"
        assert tok_idx == list(range(tok_idx[0], tok_idx[-1] + 1)), (
            f"region {r} is not contiguous"
        )
        spans.append(Sent(start=tok_idx[0], end=tok_idx[-1]))
    for i in range(1, len(spans)):
        assert spans[i].start == spans[i - 1].end + 1, "spans not adjacent"

    return ids, prefix_len, spans


def all_sentence_token_ranges(
    tokenizer, prompt_str: str, sentences: List[str], extra_text: str = "",
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Token ranges for prompt block + EVERY sentence (+ optional extra text).

    Input text is ``prompt_str + ' '.join(sentences) + extra_text`` (pass
    ``extra_text=' ' + next_sentence`` to append a forced continuation).
    Returns ``(ids, ranges)`` where ranges[0] is the prompt block,
    ranges[j] is sentence j-1, and — if extra_text is nonempty — the last
    range is the extra text.  Ranges are inclusive (start, end) token
    indices, contiguous and covering all tokens.
    """
    text = prompt_str + " ".join(sentences) + extra_text
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    starts: List[int] = []
    pos = len(prompt_str)
    for s in sentences:
        starts.append(pos)
        pos += len(s) + 1
    region_starts = [0] + [max(st - 1, 0) for st in starts]
    if extra_text:
        region_starts.append(len(prompt_str) + len(" ".join(sentences)))

    def region_of(char_pos: int) -> int:
        lo = 0
        for r, st in enumerate(region_starts):
            if st <= char_pos:
                lo = r
        return lo

    token_region = [region_of(max(b - 1, a)) for (a, b) in offsets]
    ranges: List[Tuple[int, int]] = []
    for r in range(len(region_starts)):
        idx = [i for i, tr in enumerate(token_region) if tr == r]
        assert idx and idx == list(range(idx[0], idx[-1] + 1)), (
            f"region {r} empty or non-contiguous"
        )
        ranges.append((idx[0], idx[-1]))
    return ids, ranges


# ---------------------------------------------------------------------------
# KL helpers (renormalised over the letter tokens, matching their softmax
# over found letters)
# ---------------------------------------------------------------------------

def letter_distribution(logits_row, answer_ids: List[int]) -> List[float]:
    """Softmax over the letter-token logits of one (vocab,) logits row."""
    import torch
    row = logits_row.float()
    ans = row[list(answer_ids)]
    return torch.softmax(ans, dim=-1).tolist()


def kl_from_distributions(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    import math
    return sum(
        pi * (math.log(max(pi, eps)) - math.log(max(qi, eps)))
        for pi, qi in zip(p, q)
    )


def bucket_of(num_rankable: int) -> str:
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= num_rankable < hi:
            return name
    raise ValueError(num_rankable)
