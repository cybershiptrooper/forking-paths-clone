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

from utils.activation_patching_controlled import (
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
    """Progress bar streamer for generation with support for multiple sequences."""

    def __init__(self, tokenizer, max_new_tokens, num_return_sequences=1):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.pbar = None
        self.generated_tokens = 0
        self.tokens_per_sequence = [0] * num_return_sequences

    def __enter__(self):
        total_tokens = self.max_new_tokens * self.num_return_sequences
        desc = (
            f"Generating {self.num_return_sequences} sequences"
            if self.num_return_sequences > 1
            else "Generating"
        )
        self.pbar = tqdm(total=total_tokens, desc=desc, unit="token")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()

    def put(self, value):
        if self.pbar:
            # value is a tensor containing tokens from the current generation step
            # For num_return_sequences > 1, shape is [num_return_sequences, ...]
            # For num_return_sequences = 1, shape is [seq_len] or [1]
            # Each call to put() represents one generation step
            if isinstance(value, torch.Tensor):
                value = value.cpu()
                # Handle both 1D and 2D tensors
                if value.dim() == 1:
                    # Single sequence: typically one token per step
                    # If we get multiple tokens, it's the full sequence so far
                    new_tokens = value.tolist()
                    # For streaming, we typically get one new token at a time
                    num_new = 1 if len(new_tokens) == 1 else len(new_tokens)
                    # Take the last token for display
                    display_token = new_tokens[-1] if new_tokens else None
                else:
                    # Multiple sequences: shape [num_return_sequences, current_length]
                    # Get the last token from each sequence (newly generated tokens)
                    new_tokens = value[:, -1].tolist()
                    num_new = len(new_tokens)
                    display_token = new_tokens[0] if new_tokens else None

                    # Update per-sequence token counts
                    for i in range(min(len(new_tokens), len(self.tokens_per_sequence))):
                        self.tokens_per_sequence[i] += 1
            else:
                new_tokens = [value] if isinstance(value, int) else value
                num_new = len(new_tokens) if isinstance(new_tokens, list) else 1
                display_token = (
                    new_tokens[-1]
                    if isinstance(new_tokens, list) and new_tokens
                    else (new_tokens if isinstance(new_tokens, int) else None)
                )

            self.generated_tokens += num_new
            self.pbar.update(num_new)

            # Update description with current token and sequence info
            if self.generated_tokens <= 10 and display_token is not None:
                token_text = self.tokenizer.decode(
                    [display_token], skip_special_tokens=True
                )

                if self.num_return_sequences > 1:
                    # Show which sequences are being generated
                    active_sequences = sum(
                        1 for count in self.tokens_per_sequence if count > 0
                    )
                    self.pbar.set_postfix(
                        {
                            "token": repr(token_text[:20]),
                            "sequences": f"{active_sequences}/{self.num_return_sequences}",
                        }
                    )
                else:
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
    num_return_sequences: int = 20,
    batch_size: int = 8,
):
    """
    Main function to run attention ablation experiment.

    Args:
        model_name: HuggingFace model name
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
        num_return_sequences: Number of parallel completions to generate
        batch_size: Number of sequences to generate per batch (to avoid GPU OOM)
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

    # Define source sentences (queries that should not see the targets)
    # We want to prevent generated tokens from attending to the ablated sentences
    # The generated tokens start at len(generation_prefix) and we'll ablate for the first 200 tokens
    ablation_length = 200
    source_sentences = [
        Sentence(
            start=len(generation_prefix),
            end=len(generation_prefix) + ablation_length
        )
    ]

    print(f"\nAblation configuration:")
    print(f"  Target sentences (keys to hide): {len(adjusted_sentences)} sentences")
    for i, sent in enumerate(adjusted_sentences):
        print(f"    {i+1}. tokens {sent.start}-{sent.end}")
    print(f"  Source sentences (queries to block): {source_sentences}")

    # Register ablation hooks
    print(f"\nRegistering ablation hooks for layers: {layers_parsed}...")
    handles = ablate_sentences(
        model,
        sentences_to_ablate=adjusted_sentences,  # TARGETS (Keys)
        ablate_from_sentences=source_sentences,  # SOURCES (Queries)
        layers=layers_parsed,
    )

    new_output_texts = []
    try:
        # Generate continuation with ablation in batches to avoid GPU OOM
        # Ensure batch_size doesn't exceed num_return_sequences
        effective_batch_size = min(batch_size, num_return_sequences)
        print(
            f"Generating {num_return_sequences} continuations with ablation (batch_size={effective_batch_size})..."
        )

        # Calculate number of batches needed
        num_batches = (
            num_return_sequences + effective_batch_size - 1
        ) // effective_batch_size

        # Use tqdm for batch progress
        batch_pbar = tqdm(
            total=num_batches,
            desc=f"Batches ({num_return_sequences} sequences)",
            unit="batch",
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * effective_batch_size
            batch_end = min(batch_start + effective_batch_size, num_return_sequences)
            current_batch_size = batch_end - batch_start

            batch_pbar.set_postfix(
                {
                    "batch": f"{batch_idx + 1}/{num_batches}",
                    "sequences": f"{len(new_output_texts)}/{num_return_sequences}",
                }
            )

            # Create streamer for this batch
            streamer = ProgressBarStreamer(
                tokenizer,
                max_new_tokens=max_new_tokens,
                num_return_sequences=current_batch_size,
            )

            with torch.no_grad():
                with streamer:
                    ablated_output_tokens = model.generate(
                        input_ids=generation_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        use_cache=True,
                        streamer=streamer,
                        do_sample=True,
                        num_return_sequences=current_batch_size,
                    )

            # Extract only the new tokens (after the prefix) for all sequences in this batch
            # The generate method returns full sequence, so extract tokens after prefix length
            if ablated_output_tokens.dim() == 1:
                # Single sequence returned as 1D tensor
                new_tokens = ablated_output_tokens[len(generation_prefix) :]
                new_output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                new_output_texts.append(new_output_text)
                if batch_idx == 0:
                    print(f"\n=== Ablated Output ===\n{new_output_text}\n")
            else:
                # Multiple sequences returned as 2D tensor [current_batch_size, sequence_length]
                for i in range(ablated_output_tokens.shape[0]):
                    new_tokens = ablated_output_tokens[i][len(generation_prefix) :]
                    new_output_text = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )
                    new_output_texts.append(new_output_text)
                    global_idx = batch_start + i
                    if global_idx < 3 or global_idx == num_return_sequences - 1:
                        print(
                            f"\n=== Ablated Output {global_idx + 1}/{num_return_sequences} ===\n{new_output_text}\n"
                        )
                    elif global_idx == 3:
                        print(
                            f"\n... (showing first 3, hiding {num_return_sequences - 4} more) ...\n"
                        )

            # Clear GPU memory after each batch
            del ablated_output_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update batch progress bar
            batch_pbar.update(1)

        batch_pbar.close()

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

    # Parse ablated answers for all sequences
    num_sequences = len(new_output_texts)
    print(f"Parsing ablated answers for {num_sequences} sequences...")
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

    # Prepare data for parsing all sequences
    ablated_datapoints = [
        {
            "output_text": new_output_text,
            "question": base_data["question"],
            "all_letters": base_data["all_letters"],
            "all_answers": base_data["all_answers"],
            "dataset_type": "multiple choice",
        }
        for new_output_text in new_output_texts
    ]

    parsed_results = parse_answer(answer_llm, ablated_datapoints)
    ablated_answers = [result["clean_answer"] for result in parsed_results]

    print("\n=== Parsed Answers ===")
    for i, answer in enumerate(ablated_answers):
        print(f"Sequence {i+1}: {answer}")

    # Compare answers - count how many changed
    answer_changed_counts = {
        "unchanged": 0,
        "changed": 0,
    }
    for ablated_answer in ablated_answers:
        if base_answer != ablated_answer:
            answer_changed_counts["changed"] += 1
        else:
            answer_changed_counts["unchanged"] += 1

    print("\n=== Result ===")
    print(f"Base answer: {base_answer}")
    print(f"Answer changed: {answer_changed_counts['changed']}/{num_sequences} sequences")
    print(
        f"Answer unchanged: {answer_changed_counts['unchanged']}/{num_sequences} sequences"
    )

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
        "ablated_answers": ablated_answers,
        "answer_changed_counts": answer_changed_counts,
        "offset_from_convergence": offset_from_convergence,
        "num_sentences_to_ablate": num_sentences_to_ablate,
        "random_sentences": random_sentences,
        "generation_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "batch_size": batch_size,
        },
        "ablated_output_texts": new_output_texts,
    }
    random_str = "random" if random_sentences else "top_k"
    output_file = os.path.join(
        output_dir,
        f"test_controlled_ablation_example_{idx_str}_offset{offset_from_convergence}_num_sentences{num_sentences_to_ablate}_{random_str}.json",
    )
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
    parser.add_argument("--example_index", "-idx", type=int, default=0)
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
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=16,
        help="Number of parallel completions to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sequences to generate per batch (to avoid GPU OOM). Lower values use less memory.",
    )
    args = parser.parse_args()
    main(**vars(args))
