"""
Attention ablation experiment script.

Loads attention visualization results, selects top sentences by mean score,
ablates them during generation, and checks if the model's answer changes.
"""

import argparse
import json
import os
from typing import List, Optional, Union
import glob
import gc

import torch
from vllm import LLM
from transformers.generation.streamers import BaseStreamer
from tqdm import tqdm

from utils.activation_patching_effecient import (
    ablate_sentences,
    load_custom_model_eager,
)
from utils.attention_analysis import select_top_sentences_by_mean_score
from utils.answer_utils import parse_answer
from utils.cot_analysis import get_convergence_for_index, split_tokens_into_sentences
from utils.utils import MODEL_METADATA, Sentence, clear_cuda
from expts.visualize_attention import make_prompt_mcq
import random


def find_results_file(
    results_dir: str,
    example_index: int,
    layer: Optional[int] = None,
    gap: Optional[int] = None,
) -> Optional[str]:
    """
    Find the results.json file for a given example.

    Args:
        results_dir: Base directory for attention visualization results
        example_index: Example index
        layer: Optional layer number to match
        gap: Optional gap value to match

    Returns:
        Path to results.json file, or None if not found
    """
    idx_str = str(example_index).zfill(2)

    # Pattern: example_{idx}_layer_{layer}_gap{gap}/results.json
    if layer is not None and gap is not None:
        pattern = os.path.join(
            results_dir, f"example_{idx_str}_layer_{layer}_gap{gap}", "results.json"
        )
        if os.path.exists(pattern):
            return pattern

    # Try to find any results file for this example
    pattern = os.path.join(results_dir, f"example_{idx_str}_layer_*", "results.json")
    matches = glob.glob(pattern)
    if matches:
        # Prefer files with gap if available
        gap_matches = [m for m in matches if "_gap" in m]
        if gap_matches:
            return gap_matches[0]
        return matches[0]

    return None


def adjust_sentence_indices(
    sentences: List[Sentence],
    prompt_length: int,
    convergence_token_idx: int,
) -> List[Sentence]:
    """
    Adjust sentence token indices for generation context.

    Sentences from results.json are relative to full sequence (prompt + full output).
    When generating from convergence token, we only have (prompt + output up to convergence).

    Args:
        sentences: List of Sentence objects with indices relative to full sequence
        prompt_length: Length of prompt in tokens
        convergence_token_idx: Convergence token index (relative to output start)

    Returns:
        List of Sentence objects adjusted for generation context, with sentences
        beyond convergence point filtered out
    """
    adjusted = []
    full_sequence_length = prompt_length + convergence_token_idx

    for sent in sentences:
        # If sentence is entirely before convergence point, keep it
        if sent.end < full_sequence_length:
            adjusted.append(sent)
        # If sentence starts before convergence but extends beyond, truncate it
        elif sent.start < full_sequence_length:
            adjusted.append(Sentence(start=sent.start, end=full_sequence_length - 1))
        # If sentence is entirely after convergence, skip it

    return adjusted


class ProgressBarStreamer(BaseStreamer):
    """Progress bar streamer for generation."""

    def __init__(self, tokenizer, max_new_tokens):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.pbar = None
        self.generated_tokens = 0

    def __enter__(self):
        self.pbar = tqdm(total=self.max_new_tokens, desc="Generating", unit="token")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()

    def put(self, value):
        if self.pbar:
            # value is a tensor, extract the new token(s)
            if isinstance(value, torch.Tensor):
                value = value.cpu()
                # Handle both 1D and 2D tensors
                if value.dim() == 1:
                    new_tokens = value.tolist()
                else:
                    # Get the last token from each sequence in the batch
                    new_tokens = value[:, -1].tolist()
            else:
                new_tokens = [value] if isinstance(value, int) else value

            num_new = len(new_tokens) if isinstance(new_tokens, list) else 1
            self.generated_tokens += num_new
            self.pbar.update(num_new)
            # Update description with current token
            if self.generated_tokens <= 10 and new_tokens:  # Show first few tokens
                token_ids = (
                    new_tokens[-1:] if isinstance(new_tokens, list) else [new_tokens]
                )
                token_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                self.pbar.set_postfix({"token": repr(token_text[:20])})

    def end(self):
        if self.pbar:
            self.pbar.close()


def parse_layers_arg(layers_str: str) -> Union[int, List[int], str]:
    """
    Parse layers argument from command line.

    Args:
        layers_str: String like "all", "16", or "16,17,18"

    Returns:
        "all", int, or List[int]
    """
    if layers_str.lower() == "all":
        return "all"

    try:
        # Try single int
        return int(layers_str)
    except ValueError:
        # Try comma-separated list
        return [int(x.strip()) for x in layers_str.split(",")]


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    base_model_name: str = "meta-llama/Llama-3.1-8B",  # Not used anymore, kept for compatibility
    example_index: int = 0,
    results_dir: str = "results/attention_viz",
    num_sentences_to_ablate: int = 10,
    layers: str = "all",
    temperature: float = 0.7,
    max_new_tokens: int = 16384,
    output_dir: str = "results/attention_ablation",
    streamlit_folder: str = "data/streamlit",
    dataset_name: str = "gpqa",
    do_not_clip_to_convergence: bool = False,
    random_sentences: bool = False,
    offset_from_convergence: int = 0,
):
    """
    Main function to run attention ablation experiment.

    Args:
        model_name: HuggingFace model name
        base_model_name: Base model name for TransformerLens
        example_index: Index of example from base_data.json
        results_dir: Path to attention_viz results directory
        num_sentences_to_ablate: Number of top sentences to ablate
        layers: Layers to ablate - "all", single int, or comma-separated list
        temperature: Generation temperature
        max_new_tokens: Max tokens to generate
        output_dir: Directory to save ablation results
        streamlit_folder: Path to streamlit data folder
        dataset_name: Name of dataset (e.g., 'gpqa')
        do_not_clip_to_convergence: Whether to clip to convergence token
        random_sentences: Whether to randomly select sentences
        offset_from_convergence: Number of tokens to offset from convergence token to start generation from
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # Parse layers argument
    layers_parsed = parse_layers_arg(layers)

    # Load model with eager attention
    print(f"Loading model: {model_name}")
    model, tokenizer = load_custom_model_eager(
        model_name=model_name,
        device=device,
    )
    model_nickname = MODEL_METADATA[model_name]["nickname"]

    # Load example from base_data.json
    base_data_path = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
    print(f"Loading example {example_index} from {base_data_path}")
    with open(base_data_path) as f:
        base_data = json.load(f)[example_index]

    # Create prompt
    prompt = make_prompt_mcq(base_data, tokenizer)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_length = len(prompt_token_ids)

    # Get convergence index
    idx_str = str(example_index).zfill(2)
    file_template = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{{idx}}.csv"
    convergence_result = get_convergence_for_index(idx_str, file_template)
    convergence_token_idx = convergence_result[0] if convergence_result else None
    convergence_outcome = convergence_result[1] if convergence_result else None

    if convergence_token_idx is None:
        print(f"Error: No convergence found for example {example_index}")
        return

    print(f"Convergence at token {convergence_token_idx}, outcome: {convergence_outcome}")

    # Get base answer
    base_answer = base_data.get("clean_answer", convergence_outcome)
    print(f"Base answer: {base_answer}")

    # Calculate convergence point in full sequence (prompt + output up to convergence)
    convergence_point_full_seq = prompt_length + convergence_token_idx

    if random_sentences:
        # Random sentence selection: split tokens into sentences and randomly select k
        print("Using random sentence selection...")
        print(f"Convergence point in full sequence: token {convergence_point_full_seq}")

        # Prepare generation prefix (prompt + output up to convergence)
        generation_prefix = (
            prompt_token_ids + base_data["output_token_ids"][:convergence_token_idx]
        )
        generation_tokens_tensor = torch.tensor(generation_prefix, device=device)

        # Split tokens into sentences using the function from cot_analysis
        print("Splitting tokens into sentences...")
        all_sentences = split_tokens_into_sentences(
            generation_tokens_tensor, tokenizer, min_sentence_length=10
        )

        # Filter sentences to only include those up to convergence point
        # (sentences are already relative to generation_prefix, so no adjustment needed)
        valid_sentences = [
            sent for sent in all_sentences if sent.end < len(generation_prefix)
        ]

        print(
            f"Found {len(valid_sentences)} valid sentences (out of {len(all_sentences)} total)"
        )

        if len(valid_sentences) == 0:
            print("Error: No valid sentences found for random selection")
            return

        # Randomly select k sentences
        k_actual = min(num_sentences_to_ablate, len(valid_sentences))
        sentences_to_ablate = random.sample(valid_sentences, k_actual)

        print(f"Randomly selected {len(sentences_to_ablate)} sentences to ablate:")
        for i, sent in enumerate(sentences_to_ablate):
            print(f"  {i+1}. tokens {sent.start}-{sent.end}")

        # For random selection, sentences are already relative to generation_prefix
        # so no adjustment needed - they can be used directly
        adjusted_sentences = sentences_to_ablate

    else:
        # Original top-K selection by mean score
        # Load attention visualization results
        print(f"Loading attention visualization results from {results_dir}...")
        results_file = find_results_file(results_dir, example_index)

        if results_file is None:
            print(f"Error: Could not find results.json for example {example_index}")
            return

        print(f"Found results file: {results_file}")
        with open(results_file) as f:
            attention_results = json.load(f)

        # Extract top sentences per head
        top_sentences_per_head = attention_results.get("top_sentences_per_head", {})
        if not top_sentences_per_head:
            print("Error: No top_sentences_per_head found in results.json")
            return

        print(f"Selecting top {num_sentences_to_ablate} sentences by mean score...")
        print(f"Convergence point in full sequence: token {convergence_point_full_seq}")
        sentences_to_ablate = select_top_sentences_by_mean_score(
            top_sentences_per_head,
            k=num_sentences_to_ablate,
            max_token_idx=convergence_point_full_seq,
            clip_to_max=not do_not_clip_to_convergence,
        )

        print(f"Selected {len(sentences_to_ablate)} sentences to ablate:")
        for i, sent in enumerate(sentences_to_ablate):
            print(f"  {i+1}. tokens {sent.start}-{sent.end}")

        # Count sentences until convergence
        all_sentences = attention_results.get("sentences", [])
        sentences_until_convergence = sum(
            1 for s in all_sentences if s["end"] <= convergence_point_full_seq
        )
        print(f"Number of sentences until convergence: {sentences_until_convergence}")

        # Adjust sentence indices for generation context
        # Sentences are relative to full sequence, but we generate from convergence point
        adjusted_sentences = adjust_sentence_indices(
            sentences_to_ablate, prompt_length, convergence_token_idx
        )

    if not random_sentences:
        print(
            f"After adjusting for generation context: {len(adjusted_sentences)} sentences"
        )
        for i, sent in enumerate(adjusted_sentences):
            print(f"  {i+1}. tokens {sent.start}-{sent.end}")

    if len(adjusted_sentences) == 0:
        print(
            "Warning: No sentences to ablate after adjustment. All sentences are beyond convergence point."
        )
        return

    # Prepare generation context (prompt + output up to convergence)
    if not random_sentences:
        generation_prefix = (
            prompt_token_ids
            + base_data["output_token_ids"][
                : convergence_token_idx + offset_from_convergence
            ]
        )
    # For random_sentences, generation_prefix was already created above
    generation_tokens = torch.tensor([generation_prefix], device=device)

    print(f"Generation prefix length: {len(generation_prefix)} tokens")

    # Register ablation hooks
    print(f"Registering ablation hooks for layers: {layers_parsed}...")
    handles = ablate_sentences(model, adjusted_sentences, layers=layers_parsed)

    ablated_output_tokens = None
    new_output_text = None
    try:
        # Generate continuation with ablation
        print("Generating continuation with ablation...")
        streamer = ProgressBarStreamer(tokenizer, max_new_tokens=max_new_tokens)
        with torch.no_grad():
            with streamer:
                ablated_output_tokens = model.generate(
                    input_ids=generation_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_cache=True,
                    streamer=streamer,
                    do_sample=True,
                    # num_return_sequences=10,
                )

        # Extract only the new tokens (after the prefix)
        # The generate method returns full sequence, so extract tokens after prefix length
        new_tokens = ablated_output_tokens[0][len(generation_prefix) :]
        new_output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n=== Ablated Output ===\n{new_output_text}\n")

    finally:
        # Remove hooks
        print("Removing ablation hooks...")
        for handle in handles:
            handle.remove()
        handles.clear()

        # Clean up model and tokenizer to free GPU memory
        print("Freeing GPU memory...")
        del model
        del tokenizer
        if ablated_output_tokens is not None:
            del ablated_output_tokens
        del generation_tokens

        # Force multiple rounds of garbage collection and CUDA cache clearing
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        clear_cuda()

        # Verify memory is freed
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            free_memory = total - reserved
            print(
                f"GPU memory after cleanup - Allocated: {allocated / 1024**3:.2f} GB, "
                f"Reserved: {reserved / 1024**3:.2f} GB, "
                f"Free: {free_memory / 1024**3:.2f} GB"
            )

    # Parse ablated answer
    print("Parsing ablated answer...")
    # Load answer model from config
    with open("config.json") as f:
        config = json.load(f)
    answer_model_name = config["experiment_parameters"]["answer_model"]

    # Check available memory before loading VLLM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        free_memory = total - reserved
        print(
            f"Before loading VLLM - Allocated: {allocated / 1024**3:.2f} GB, "
            f"Reserved: {reserved / 1024**3:.2f} GB, "
            f"Free: {free_memory / 1024**3:.2f} GB"
        )

        # VLLM by default tries to use 90% of GPU memory, so we need at least that much free
        required_memory = total * 0.9
        if free_memory < required_memory:
            print(
                f"Warning: Free memory ({free_memory / 1024**3:.2f} GB) is less than "
                f"VLLM's default requirement ({required_memory / 1024**3:.2f} GB). "
                f"Trying to load with reduced GPU memory utilization..."
            )
            # Reduce GPU memory utilization to 80% instead of default 90%
            answer_llm = LLM(
                model=answer_model_name, dtype="bfloat16", gpu_memory_utilization=0.8
            )
        else:
            answer_llm = LLM(model=answer_model_name, dtype="bfloat16")
    else:
        answer_llm = LLM(model=answer_model_name, dtype="bfloat16")

    # Prepare data for parsing
    ablated_datapoint = {
        "output_text": new_output_text,
        "question": base_data["question"],
        "all_letters": base_data["all_letters"],
        "all_answers": base_data["all_answers"],
        "dataset_type": "multiple choice",
    }

    parsed_results = parse_answer(answer_llm, [ablated_datapoint])
    ablated_answer = parsed_results[0]["clean_answer"]

    print(f"Ablated answer: {ablated_answer}")

    # Compare answers
    answer_changed = base_answer != ablated_answer
    print("\n=== Result ===")
    print(f"Base answer: {base_answer}")
    print(f"Ablated answer: {ablated_answer}")
    print(f"Answer changed: {answer_changed}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model_name": model_name,
        "example_index": example_index,
        "layers_ablated": str(layers_parsed),
        "num_sentences_ablated": len(adjusted_sentences),
        "sentences_ablated": [
            {"start": s.start, "end": s.end} for s in adjusted_sentences
        ],
        "original_sentences": [
            {"start": s.start, "end": s.end} for s in sentences_to_ablate
        ],
        "convergence_token_idx": convergence_token_idx,
        "convergence_outcome": convergence_outcome,
        "base_answer": base_answer,
        "ablated_answer": ablated_answer,
        "answer_changed": answer_changed,
        "random_sentences": random_sentences,
        "generation_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "ablated_output_text": new_output_text,
    }

    output_file = os.path.join(output_dir, f"ablation_example_{idx_str}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention ablation experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
    )
    parser.add_argument("--example_index", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results/attention_viz")
    parser.add_argument("--num_sentences_to_ablate", type=int, default=10)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layers to ablate: "all", single int, or comma-separated list (e.g., "16,17,18")',
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--output_dir", type=str, default="results/attention_ablation")
    parser.add_argument("--streamlit_folder", type=str, default="data/streamlit")
    parser.add_argument("--dataset_name", type=str, default="gpqa")
    parser.add_argument("--do_not_clip_to_convergence", action="store_true")
    parser.add_argument(
        "--random_sentences",
        action="store_true",
        help="Randomly select k sentences instead of selecting top k by mean score",
    )
    parser.add_argument("--offset_from_convergence", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
