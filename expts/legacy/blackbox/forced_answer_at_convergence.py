"""
Script to force an answer out of the LLM after the convergence timestep.

This script:
1. Detects convergence using cot_analysis functions
2. Forms prompts using the MCQ format
3. Appends the model's CoT prefix up to convergence (plus optional extra tokens)
4. Forces an answer with a specific suffix
5. Gets probabilities for A, B, C, D
6. Reports everything in a stats dataframe
"""

import argparse
import glob
import json
import math
import os
import re
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.cot_analysis import (
    get_convergence_for_index,
    find_token_index,
    load_data,
)
from utils.prompt_utils import get_cot_prompt


def get_available_indices(file_template: str) -> List[int]:
    """
    Get list of available indices by listing CSV files that match the template.

    Args:
        file_template: Template string like "data/path/{idx}.csv"

    Returns:
        Sorted list of integer indices found
    """
    # Convert template to glob pattern
    # e.g., "data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv" -> "data/streamlit/deepseek_llama_8b/gpqa/*.csv"
    glob_pattern = file_template.replace("{idx}", "*")
    csv_files = glob.glob(glob_pattern)

    # Extract indices from filenames
    indices = []
    # Get the basename pattern to extract the index
    # e.g., for template "path/{idx}.csv", we want to match "path/00.csv" and extract "00"
    template_basename = os.path.basename(file_template)
    # Create regex pattern from template basename
    # Replace {idx} with a capture group for digits
    pattern = template_basename.replace("{idx}", r"(\d+)")
    regex = re.compile(pattern)

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        match = regex.match(basename)
        if match:
            idx = int(match.group(1))
            indices.append(idx)

    return sorted(indices)


def make_prompt_mcq(base_data_dict: dict, tokenizer: AutoTokenizer) -> str:
    """Form the MCQ prompt (from probing_classifier.py)"""
    question = base_data_dict["question"]
    option_choices = base_data_dict["all_answers"]
    letter_choices = base_data_dict["all_letters"]
    formatted_question = f"{question}\n\nChoices:\n" + "\n".join(
        f"{letter}) {option}" for letter, option in zip(letter_choices, option_choices)
    )
    prompt_str = get_cot_prompt(tokenizer, formatted_question, multiple_choice=True)
    return prompt_str


def get_forced_answer_logprobs(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt_token_ids: list,
    cot_prefix_token_ids: list,
    forced_suffix: str = "\n</think>\n\nTherefore, the final answer is (",
) -> dict:
    """
    Get logprobs for A, B, C, D by forcing the model to answer.

    Returns dict mapping letter -> logprob
    """
    # Tokenize the forced suffix
    forced_suffix_token_ids = tokenizer.encode(forced_suffix, add_special_tokens=False)

    # Combine all token ids
    full_prompt_token_ids = (
        prompt_token_ids + cot_prefix_token_ids + forced_suffix_token_ids
    )

    # Get logprobs for the next token
    sampling_params = SamplingParams(
        max_tokens=1,
        logprobs=20,  # Get enough logprobs to capture A, B, C, D
        temperature=0.0,
    )

    outputs = llm.generate([{"prompt_token_ids": full_prompt_token_ids}], sampling_params)

    # Extract probabilities for A, B, C, D
    logprobs_dict = {
        "A": float("-inf"),
        "B": float("-inf"),
        "C": float("-inf"),
        "D": float("-inf"),
    }

    if outputs[0].outputs[0].logprobs:
        first_token_logprobs = outputs[0].outputs[0].logprobs[0]
        for token_id, logprob_info in first_token_logprobs.items():
            decoded = tokenizer.decode([token_id]).strip()
            if decoded in ["A", "B", "C", "D"]:
                logprobs_dict[decoded] = logprob_info.logprob

    return logprobs_dict


def get_forking_path_probs(idx: str, file_template: str) -> dict:
    """Get outcome probabilities at the lowest timestep from forking paths data."""
    df = load_data(idx, file_template)
    lowest_timestep = df["t"].min()

    outcome_probs = {}
    for letter in ["A", "B", "C", "D"]:
        row = df[(df["t"] == lowest_timestep) & (df["outcome"] == letter)]
        outcome_probs[letter] = (
            row["outcome_probability"].values[0] if len(row) > 0 else 0.0
        )

    return outcome_probs


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    base_data_path: str = "data/streamlit/deepseek_llama_8b/gpqa/base_data.json",
    file_template: str = "data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv",
    extra_tokens: int = 0,
    output_path: str = "forced_answer_stats.csv",
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
):
    """
    Main function to force answer extraction at convergence point.

    Args:
        model_name: HuggingFace model name
        base_data_path: Path to base_data.json
        file_template: Template for CSV files with {idx} placeholder
        extra_tokens: Extra tokens to include after convergence (default 0)
        output_path: Path to save the output CSV
        start_index: Start index for processing (default 0)
        end_index: End index for processing (default len(base_data))
    """
    print(f"Loading base data from {base_data_path}...")
    with open(base_data_path) as f:
        base_data = json.load(f)

    # Get available indices from CSV files
    available_indices = get_available_indices(file_template)
    print(f"Found {len(available_indices)} CSV files: {available_indices}")

    # Filter indices based on start_index and end_index
    if start_index is not None:
        available_indices = [i for i in available_indices if i >= start_index]
    if end_index is not None:
        available_indices = [i for i in available_indices if i < end_index]

    # Also filter to only include indices that exist in base_data
    available_indices = [i for i in available_indices if i < len(base_data)]
    print(f"Processing {len(available_indices)} indices: {available_indices}")

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading LLM {model_name}...")
    llm = LLM(model=model_name, dtype="auto")

    results = []
    skipped = []

    for i in available_indices:
        idx = str(i).zfill(2)
        print(f"\nProcessing index {idx}...")

        # Get convergence info
        convergence_result = get_convergence_for_index(idx, file_template)

        if convergence_result is None:
            print(f"  Skipping {idx}: no convergence found")
            skipped.append({"idx": idx, "reason": "no convergence"})
            continue

        converged_timestep, converged_outcome = convergence_result

        # Get think finish index
        output_token_ids = base_data[i]["output_token_ids"]
        think_finish_idx = find_token_index(tokenizer, "</think>", output_token_ids)

        if think_finish_idx == -1:
            print(f"  Skipping {idx}: no </think> token found")
            skipped.append({"idx": idx, "reason": "no </think> token"})
            continue

        # Calculate prefix length
        prefix_length = converged_timestep + extra_tokens

        # Calculate distance from think end
        dist_from_think_end = think_finish_idx - converged_timestep

        # Check skip conditions
        if converged_timestep > think_finish_idx:
            print(
                f"  Skipping {idx}: convergence ({converged_timestep}) is after think token ({think_finish_idx})"
            )
            skipped.append(
                {
                    "idx": idx,
                    "reason": f"convergence after think token: {converged_timestep} > {think_finish_idx}",
                }
            )
            continue

        if prefix_length > think_finish_idx:
            print(
                f"  Skipping {idx}: prefix_length ({prefix_length}) exceeds think_finish_idx ({think_finish_idx})"
            )
            skipped.append(
                {
                    "idx": idx,
                    "reason": f"prefix_length > think_finish_idx: {prefix_length} > {think_finish_idx}",
                }
            )
            continue

        # Form the prompt
        prompt_str = make_prompt_mcq(base_data[i], tokenizer)
        prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Get the CoT prefix up to convergence + extra_tokens
        cot_prefix_token_ids = output_token_ids[:prefix_length]

        # Get forced answer logprobs
        print("  Getting forced answer logprobs...")
        logprobs_dict = get_forced_answer_logprobs(
            llm, tokenizer, prompt_token_ids, cot_prefix_token_ids
        )

        # Get forking path probabilities
        forking_probs = get_forking_path_probs(idx, file_template)

        # Convert logprobs to probabilities for comparison
        forced_probs = {}
        for letter in ["A", "B", "C", "D"]:
            if logprobs_dict[letter] != float("-inf"):
                forced_probs[letter] = math.exp(logprobs_dict[letter])
            else:
                forced_probs[letter] = 0.0

        # Build result entry
        result = {
            "idx": idx,
            "original_answer": base_data[i]["clean_answer"],
            "correct_answer": base_data[i]["correct_letter"],
            "converged_outcome": converged_outcome,
            "converged_timestep": converged_timestep,
            "think_finish_idx": think_finish_idx,
            "dist_from_think_end": dist_from_think_end,
            "extra_tokens": extra_tokens,
            "prefix_length": prefix_length,
            # Forking path probabilities
            "forking_prob_A": forking_probs["A"],
            "forking_prob_B": forking_probs["B"],
            "forking_prob_C": forking_probs["C"],
            "forking_prob_D": forking_probs["D"],
            # Forced answer logprobs
            "forced_logprob_A": logprobs_dict["A"],
            "forced_logprob_B": logprobs_dict["B"],
            "forced_logprob_C": logprobs_dict["C"],
            "forced_logprob_D": logprobs_dict["D"],
            # Forced answer probabilities (for easier comparison)
            "forced_prob_A": forced_probs["A"],
            "forced_prob_B": forced_probs["B"],
            "forced_prob_C": forced_probs["C"],
            "forced_prob_D": forced_probs["D"],
        }

        results.append(result)

        # Get the forced answer (max prob)
        forced_answer = max(forced_probs, key=forced_probs.get)
        forking_answer = max(forking_probs, key=forking_probs.get)

        print(
            f"  Done: converged={converged_outcome}, correct={base_data[i]['correct_letter']}, "
            f"forced={forced_answer} (p={forced_probs[forced_answer]:.4f}), "
            f"forking={forking_answer} (p={forking_probs[forking_answer]:.4f})"
        )

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    if len(df_results) > 0:
        # Select columns for display
        display_cols = [
            "idx",
            "correct_answer",
            "converged_outcome",
            "original_answer",
            "forking_prob_A",
            "forking_prob_B",
            "forking_prob_C",
            "forking_prob_D",
            "forced_prob_A",
            "forced_prob_B",
            "forced_prob_C",
            "forced_prob_D",
        ]
        print(df_results[display_cols].to_string(index=False))

        # Calculate agreement statistics
        df_results["forced_answer"] = (
            df_results[
                ["forced_prob_A", "forced_prob_B", "forced_prob_C", "forced_prob_D"]
            ]
            .idxmax(axis=1)
            .str[-1]
        )
        df_results["forking_answer"] = (
            df_results[
                ["forking_prob_A", "forking_prob_B", "forking_prob_C", "forking_prob_D"]
            ]
            .idxmax(axis=1)
            .str[-1]
        )

        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS:")
        print("-" * 80)

        n_total = len(df_results)
        n_forced_correct = (
            df_results["forced_answer"] == df_results["correct_answer"]
        ).sum()
        n_forking_correct = (
            df_results["forking_answer"] == df_results["correct_answer"]
        ).sum()
        n_converged_correct = (
            df_results["converged_outcome"] == df_results["correct_answer"]
        ).sum()
        n_agreement = (df_results["forced_answer"] == df_results["forking_answer"]).sum()

        print(f"Total samples: {n_total}")
        print(
            f"Forced answer accuracy: {n_forced_correct}/{n_total} ({100*n_forced_correct/n_total:.1f}%)"
        )
        print(
            f"Forking path accuracy: {n_forking_correct}/{n_total} ({100*n_forking_correct/n_total:.1f}%)"
        )
        print(
            f"Converged answer accuracy: {n_converged_correct}/{n_total} ({100*n_converged_correct/n_total:.1f}%)"
        )
        print(
            f"Forced-Forking agreement: {n_agreement}/{n_total} ({100*n_agreement/n_total:.1f}%)"
        )
    else:
        print("No results to display.")

    if len(skipped) > 0:
        print("\n" + "-" * 80)
        print(f"SKIPPED ({len(skipped)} samples):")
        print("-" * 80)
        for s in skipped:
            print(f"  {s['idx']}: {s['reason']}")

    # Save results
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

    # Create scatter plot: forced answer probability vs distance from think end
    if len(df_results) > 0:
        # Calculate max forced probability for each sample
        df_results["max_forced_prob"] = df_results[
            ["forced_prob_A", "forced_prob_B", "forced_prob_C", "forced_prob_D"]
        ].max(axis=1)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color by correctness
        correct_mask = df_results["forced_answer"] == df_results["correct_answer"]

        ax.scatter(
            df_results.loc[correct_mask, "dist_from_think_end"],
            df_results.loc[correct_mask, "max_forced_prob"],
            alpha=0.6,
            label="Correct",
            color="green",
            s=50,
        )
        ax.scatter(
            df_results.loc[~correct_mask, "dist_from_think_end"],
            df_results.loc[~correct_mask, "max_forced_prob"],
            alpha=0.6,
            label="Incorrect",
            color="red",
            s=50,
        )

        ax.set_xlabel("Distance from Think End (tokens)", fontsize=12)
        ax.set_ylabel("Max Forced Answer Probability", fontsize=12)
        ax.set_title("Forced Answer Probability vs Distance from Think End", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_path = output_path.replace(".csv", "_scatter_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter plot to {plot_path}")
        plt.close()

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Force answer extraction at convergence point"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        default="data/streamlit/deepseek_llama_8b/gpqa/base_data.json",
        help="Path to base_data.json",
    )
    parser.add_argument(
        "--file_template",
        type=str,
        default="data/streamlit/deepseek_llama_8b/gpqa/{idx}.csv",
        help="Template for CSV files with {idx} placeholder",
    )
    parser.add_argument(
        "--extra_tokens",
        type=int,
        default=0,
        help="Extra tokens to include after convergence (default 0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="forced_answer_stats.csv",
        help="Path to save the output CSV",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index for processing (default 0)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing (default len(base_data))",
    )

    args = parser.parse_args()
    main(**vars(args))
