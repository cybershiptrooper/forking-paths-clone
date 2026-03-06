"""Reward computation for reward-weighted circuit discovery.

Provides:
- Answer correctness rewards (+1/-1) via LLM judge
- Chain-of-thought length rewards (shorter = positive)
- Answer token position finding for answer-only masking
"""

import os
import re
from typing import Optional

import torch

from utils.cot_analysis import split_tokens_into_sentences


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last ``\\boxed{...}``, handling nested braces."""
    result = None
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            result = text[start : i - 1]
    return result


# ---------------------------------------------------------------------------
# LLM-based answer judging (OpenRouter)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a math answer checker. Given a math problem, the correct answer, and a model's answer, \
determine if the model's answer is mathematically equivalent to the correct answer. \
Equivalent means the same value even if written differently (e.g. 42/5 = 8.4, \\frac{{1}}{{2}} = 0.5).

Problem:
{question}

Correct answer: {ground_truth}

Model's answer: {model_answer}

Is the model's answer correct? Reply with only YES or NO."""


def judge_answer(
    question: str,
    ground_truth: str,
    model_answer: str,
    client,
    model: str = "meta-llama/llama-3.2-3b-instruct",
) -> bool:
    """Judge whether *model_answer* is equivalent to *ground_truth* using an LLM.

    Args:
        client: An ``openai.OpenAI`` compatible client (e.g. OpenRouter).
        model: Model identifier for the judge.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    model_answer=model_answer,
                ),
            }
        ],
        max_tokens=5,
        temperature=0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Correctness rewards
# ---------------------------------------------------------------------------


def compute_correctness_rewards(
    branches: list[dict],
    correct_answer: str,
    question: str,
    prefix_text: str,
    client,
    judge_model: str = "meta-llama/llama-3.2-3b-instruct",
) -> list[float]:
    """Compute +1/-1 rewards based on answer correctness for each branch.

    Args:
        branches: List of dicts with ``"text"`` (continuation text).
        correct_answer: Ground-truth answer (extracted, e.g. from ``\\boxed``).
        question: The original question text.
        prefix_text: Text preceding the branch continuations.
        client: OpenAI-compatible client for judging.
        judge_model: Model to use for judging.

    Returns:
        List of floats, +1.0 for correct, -1.0 for incorrect.
    """
    rewards = []
    for branch in branches:
        full_text = prefix_text + branch["text"]
        branch_answer = extract_boxed(full_text)
        if branch_answer is None:
            # No boxed answer found — treat as incorrect
            rewards.append(-1.0)
            continue
        correct = judge_answer(
            question, correct_answer, branch_answer, client, judge_model
        )
        rewards.append(1.0 if correct else -1.0)
    return rewards


# ---------------------------------------------------------------------------
# CoT length rewards
# ---------------------------------------------------------------------------


def compute_cot_length_rewards(
    branches: list[dict],
    tokenizer,
    min_sentence_length: int = 10,
) -> list[float]:
    """Compute rewards based on CoT length in sentences.

    Reward = mean_sentence_count - branch_sentence_count.
    Positive for shorter-than-average branches, negative for longer.

    Args:
        branches: List of dicts with ``"token_ids"`` (list of ints).
        tokenizer: HuggingFace tokenizer.
        min_sentence_length: Minimum tokens per sentence for splitting.

    Returns:
        List of float rewards.
    """
    sentence_counts = []
    for branch in branches:
        token_ids = torch.tensor(branch["token_ids"])
        sents = split_tokens_into_sentences(
            token_ids, tokenizer, min_sentence_length=min_sentence_length
        )
        sentence_counts.append(len(sents))

    mean_count = sum(sentence_counts) / max(len(sentence_counts), 1)
    return [mean_count - count for count in sentence_counts]


# ---------------------------------------------------------------------------
# Answer token position finding (for answer-only masking)
# ---------------------------------------------------------------------------


def find_answer_token_positions(
    branch_text: str,
    branch_token_ids: list[int],
    tokenizer,
    prefix_len: int,
) -> Optional[torch.Tensor]:
    """Build a position mask covering only the ``\\boxed{...}`` answer tokens.

    Args:
        branch_text: Full text of the continuation (branch only).
        branch_token_ids: Token IDs of the continuation.
        tokenizer: HuggingFace tokenizer.
        prefix_len: Length of the prompt prefix in tokens.

    Returns:
        A ``(1, prefix_len + cont_len)`` position mask tensor with 1s only at
        answer token positions, or None if no boxed answer is found.
    """
    # Find the last \boxed{...} character span
    last_match = None
    for m in re.finditer(r"\\boxed\{", branch_text):
        start = m.start()
        depth, i = 1, m.end()
        while i < len(branch_text) and depth > 0:
            if branch_text[i] == "{":
                depth += 1
            elif branch_text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last_match = (start, i)  # character span of entire \boxed{...}

    if last_match is None:
        return None

    char_start, char_end = last_match

    # Map character span to token positions by decoding token-by-token
    cont_len = len(branch_token_ids)
    full_len = prefix_len + cont_len

    # Decode each continuation token to find cumulative character offsets
    cumulative_chars = 0
    token_char_starts = []
    for tid in branch_token_ids:
        token_char_starts.append(cumulative_chars)
        token_text = tokenizer.decode([tid])
        cumulative_chars += len(token_text)

    # Find tokens that overlap with the answer character span
    mask = torch.zeros(1, full_len)
    for tok_idx, tok_char_start in enumerate(token_char_starts):
        tok_text = tokenizer.decode([branch_token_ids[tok_idx]])
        tok_char_end = tok_char_start + len(tok_text)
        # Check overlap with answer span
        if tok_char_end > char_start and tok_char_start < char_end:
            # Position in full sequence: prefix_len + tok_idx
            # But position_mask convention: mask[i] = 1 means logits[i] matters
            # logits[i] predicts token i+1, so for token at pos p, mask p-1
            abs_pos = prefix_len + tok_idx
            if abs_pos > 0:
                mask[0, abs_pos - 1] = 1.0

    if mask.sum() == 0:
        return None

    return mask
