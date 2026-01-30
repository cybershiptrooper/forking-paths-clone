import pandas as pd
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
import re
import torch

from utils.utils import SENTENCE_DELIMITERS, Sentence


def load_data(
    idx: str, file_template: str = "data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv"
) -> pd.DataFrame:
    """Load CSV data for a given index."""
    df = pd.read_csv(file_template.format(idx=idx))
    return df


def get_final_convergence(df: pd.DataFrame) -> Optional[Tuple[int, str]]:
    """
    Find the timestep when an outcome reaches probability 1.0 and stays there until the end.

    Returns:
        (timestep, outcome) if found, None otherwise
    """
    max_t = df["t"].max()

    # Check if there's an outcome with probability 1.0 at the final timestep
    final_rows = df[df["t"] == max_t]
    final_locked = final_rows[final_rows["outcome_probability"] == 1.0]

    if final_locked.empty:
        return None  # No convergence at the end

    # Get the outcome that's locked at the end
    final_outcome = final_locked["outcome"].iloc[0]

    # Now find the earliest timestep where this outcome became 1.0 and stayed 1.0
    # Filter for this outcome only
    outcome_df = df[df["outcome"] == final_outcome].sort_values("t")

    # Find the earliest timestep from which probability stays 1.0
    # Work backwards from the end to find where it first became 1.0
    timesteps = outcome_df["t"].values
    probs = outcome_df["outcome_probability"].values

    # Find the last index where probability is not 1.0
    convergence_start_idx = 0
    for i in range(len(probs) - 1, -1, -1):
        if probs[i] != 1.0:
            convergence_start_idx = i + 1
            break

    if convergence_start_idx >= len(timesteps):
        return None  # Edge case: never converged

    convergence_t = timesteps[convergence_start_idx]

    return int(convergence_t), final_outcome


def get_convergence_for_index(
    idx: str, file_template: str = "data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv"
) -> Optional[Tuple[int, str]]:
    """
    Load data for index and return the final convergence timestep and outcome.
    """
    df = load_data(idx, file_template)
    return get_final_convergence(df)


def find_token_index(tokenizer, token_to_find, token_ids):
    for i, tok in enumerate(tokenizer.convert_ids_to_tokens(token_ids)):
        if token_to_find in tok:
            return i
    return -1


def make_stats(
    base_data: dict,
    tokenizer: AutoTokenizer,
    file_template: str = "data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv",
) -> dict:
    idxs = [str(i).zfill(2) for i in range(0, 28)]
    stats = {}

    for i in idxs:
        entry = {}
        df = load_data(i, file_template)
        converged_timestep, converged_outcome = get_convergence_for_index(i)
        entry["converged_outcome"] = converged_outcome
        entry["converged_timestep"] = converged_timestep
        entry["base_outcome"] = base_data[int(i)]["clean_answer"]
        entry["correct_outcome"] = base_data[int(i)]["correct_letter"]
        entry["output_length"] = len(base_data[int(i)]["output_token_ids"])
        entry["think_finish_idx"] = find_token_index(
            tokenizer, "</think>", base_data[int(i)]["output_token_ids"]
        )
        entry["is_correct"] = entry["converged_outcome"] == entry["correct_outcome"]
        lowest_timestep = df["t"].min()
        correct_letter = base_data[int(i)]["correct_letter"]
        lowest_timestep_correct_row = df[
            (df["t"] == lowest_timestep) & (df["outcome"] == correct_letter)
        ]
        outcome_probability = lowest_timestep_correct_row["outcome_probability"].values
        outcome_probability = (
            outcome_probability[0] if len(outcome_probability) > 0 else 0
        )
        entry["outcome_probability_at_lowest_timestep"] = outcome_probability
        entry["lowest_timestep"] = lowest_timestep
        entry["dist_from_think_end"] = (
            entry["think_finish_idx"] - entry["converged_timestep"]
        )
        stats[i] = entry
        del df

    return stats


def split_into_sentences(text: str, min_sentence_length: int = 50) -> list[str]:
    """
    Split text into sentences based on periods, question marks, exclamation marks, and newlines.
    Multiple consecutive newlines are treated as a single separator.

    Args:
        text: The text to split into sentences

    Returns:
        List of sentences (non-empty strings)
    """
    if not text:
        return []

    # Normalize newlines: replace one or more newlines with a single newline
    text = re.sub(r"\n+", "\n", text)

    sentences = []
    current_sentence = ""

    i = 0
    while i < len(text):
        char = text[i]

        # Check for sentence-ending punctuation or <think> tag
        if (
            char in SENTENCE_DELIMITERS
            or (text[i : i + 7] == "<think>")
            or (text[i : i + 8] == "</think>")
        ):
            tag_len = 0
            if char in SENTENCE_DELIMITERS:
                current_sentence += char
                tag_len = 1
            elif text[i : i + 7] == "<think>":
                tag_len = 7
                current_sentence += "<think>"
            elif text[i : i + 8] == "</think>":
                tag_len = 8
                current_sentence += "</think>"

            # Check if this is followed by whitespace, newline, or end of string
            next_idx = i + tag_len
            if (
                next_idx >= len(text)
                or text[next_idx] in " \n"
            ):
                if (
                    current_sentence.strip()
                    and len(current_sentence.strip()) >= min_sentence_length
                ):
                    sentences.append(current_sentence.strip())
                current_sentence = ""
                # Skip the whitespace after punctuation or tag
                if next_idx < len(text) and text[next_idx] == " ":
                    i = next_idx  # skip the space
            i += tag_len
        # Check for newline (sentence separator)
        elif char == "\n":
            if (
                current_sentence.strip()
                and len(current_sentence.strip()) >= min_sentence_length
            ):
                sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 1
        else:
            current_sentence += char
            i += 1

    # Add any remaining text as a sentence
    if current_sentence.strip() and len(current_sentence.strip()) >= min_sentence_length:
        sentences.append(current_sentence.strip())

    # Filter out empty sentences
    return [s for s in sentences if s]


def split_tokens_into_sentences(
    token_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    min_sentence_length: int = 10
) -> List[Sentence]:
    """
    Split a token sequence into sentences based on SENTENCE_DELIMITERS.
    
    A sentence boundary is detected when a token's decoded string contains
    any of the delimiters (., !, ?, newline).
    
    Args:
        token_ids: 1D tensor of token IDs
        tokenizer: The tokenizer used to decode tokens
        min_sentence_length: Minimum number of tokens per sentence (default: 10)
        
    Returns:
        List of Sentence namedtuples with (start, end) token indices (inclusive)
        Example: [Sentence(start=0, end=31), Sentence(start=32, end=56), ...]
    """
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze()
    
    token_ids_list = token_ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids_list)
    
    sentences = []
    current_start = 0
    
    for i, token in enumerate(tokens):
        # Decode the token to check for delimiters
        # Some tokenizers use special prefixes like 'Ġ' for space, so decode properly
        decoded_token = tokenizer.decode([token_ids_list[i]])
        
        # Check if any delimiter is in this token
        has_delimiter = any(delim in decoded_token for delim in SENTENCE_DELIMITERS)
        
        # Also check for </think> tag which may span tokens
        is_think_end = "</think>" in decoded_token or token.lower() in ["</think>", "think>"]
        
        if has_delimiter or is_think_end:
            # Check if we have enough tokens for a sentence
            sentence_length = i - current_start + 1
            if sentence_length >= min_sentence_length:
                sentences.append(Sentence(start=current_start, end=i))
                current_start = i + 1
    
    # Handle remaining tokens as the last sentence
    if current_start < len(tokens):
        remaining_length = len(tokens) - current_start
        if remaining_length >= min_sentence_length:
            sentences.append(Sentence(start=current_start, end=len(tokens) - 1))
        elif sentences:
            # Merge with previous sentence if too short
            prev = sentences.pop()
            sentences.append(Sentence(start=prev.start, end=len(tokens) - 1))
        else:
            # Only one short sentence - keep it anyway
            sentences.append(Sentence(start=current_start, end=len(tokens) - 1))
    
    return sentences


def get_sentence_for_token(
    token_idx: int,
    sentences: List[Sentence]
) -> Optional[int]:
    """
    Find which sentence index contains the given token index.
    
    Args:
        token_idx: The token index to search for
        sentences: List of Sentence namedtuples from split_tokens_into_sentences
        
    Returns:
        The sentence index (0-based) containing the token, or None if not found
    """
    for i, sentence in enumerate(sentences):
        if sentence.start <= token_idx <= sentence.end:
            return i
    return None
