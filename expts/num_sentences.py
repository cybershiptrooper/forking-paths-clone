"""
Simple script to print the number of sentences until convergence for an example ID.
"""

import argparse
import json
import torch
from transformers import AutoTokenizer

from utils.cot_analysis import (
    get_convergence_for_index,
    split_tokens_into_sentences,
)
from utils.utils import MODEL_METADATA
from expts.visualize_attention import make_prompt_mcq


def main(
    example_index: int,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    streamlit_folder: str = "data/streamlit",
    dataset_name: str = "gpqa",
):
    """
    Print the number of sentences until convergence for an example.

    Args:
        example_index: Index of example from base_data.json
        model_name: HuggingFace model name
        streamlit_folder: Path to streamlit data folder
        dataset_name: Name of dataset (e.g., 'gpqa')
    """
    # Get model nickname
    model_nickname = MODEL_METADATA[model_name]["nickname"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load example from base_data.json
    base_data_path = (
        f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
    )
    print(f"Loading example {example_index} from {base_data_path}")
    with open(base_data_path) as f:
        base_data = json.load(f)[example_index]

    # Create prompt
    prompt = make_prompt_mcq(base_data, tokenizer)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_length = len(prompt_token_ids)

    # Get convergence index
    idx_str = str(example_index).zfill(2)
    file_template = (
        f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{{idx}}.csv"
    )
    convergence_result = get_convergence_for_index(idx_str, file_template)
    convergence_token_idx = convergence_result[0] if convergence_result else None
    convergence_outcome = convergence_result[1] if convergence_result else None

    if convergence_token_idx is None:
        print(f"Error: No convergence found for example {example_index}")
        return

    print(f"Convergence at token {convergence_token_idx}, outcome: {convergence_outcome}")

    # Prepare generation prefix (prompt + output up to convergence)
    generation_prefix = (
        prompt_token_ids + base_data["output_token_ids"][:convergence_token_idx]
    )
    generation_tokens_tensor = torch.tensor(generation_prefix)

    # Split tokens into sentences
    print("Splitting tokens into sentences...")
    all_sentences = split_tokens_into_sentences(
        generation_tokens_tensor, tokenizer, min_sentence_length=10
    )

    # Filter sentences to only include those up to convergence point
    # (sentences are relative to generation_prefix, so we check against prefix length)
    valid_sentences = [
        sent for sent in all_sentences if sent.end < len(generation_prefix)
    ]

    print(f"\n=== Results ===")
    print(f"Example index: {example_index}")
    print(f"Convergence token index: {convergence_token_idx}")
    print(f"Convergence outcome: {convergence_outcome}")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Generation prefix length: {len(generation_prefix)} tokens")
    print(f"Total sentences found: {len(all_sentences)}")
    print(f"Sentences until convergence: {len(valid_sentences)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print number of sentences until convergence for an example"
    )
    parser.add_argument(
        "--example_index",
        "-idx",
        type=int,
        required=True,
        help="Index of example from base_data.json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--streamlit_folder",
        type=str,
        default="data/streamlit",
        help="Path to streamlit data folder",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gpqa",
        help="Name of dataset (e.g., 'gpqa')",
    )
    args = parser.parse_args()
    main(**vars(args))
