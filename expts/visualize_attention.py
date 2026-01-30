"""
Attention pattern visualization script.

Extracts attention patterns from a language model, aggregates by sentences,
computes vertical scores, and generates visualizations with convergence markers.
"""

import argparse
import json
import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.activation_caching import get_attention_patterns
from utils.attention_analysis import (
    aggregate_attention_by_sentences,
    compute_vertical_scores,
    get_top_kurtosis_heads,
    get_top_k_sentences_per_head,
    zero_diagonal,
    apply_gap_filter,
)
from utils.cot_analysis import (
    get_convergence_for_index,
    split_tokens_into_sentences,
    get_sentence_for_token,
)
from utils.plotting import plot_vertical_attention_scores, plot_attention_matrix
from utils.prompt_utils import get_cot_prompt
from utils.utils import MODEL_METADATA


def make_prompt_mcq(base_data_dict: dict, tokenizer: AutoTokenizer) -> str:
    """Create a multiple choice question prompt from base data."""
    question = base_data_dict["question"]
    option_choices = base_data_dict["all_answers"]
    letter_choices = base_data_dict["all_letters"]
    formatted_question = f"{question}\n\nChoices:\n" + "\n".join(
        f"{letter}) {option}" for letter, option in zip(letter_choices, option_choices)
    )
    prompt_str = get_cot_prompt(tokenizer, formatted_question, multiple_choice=True)
    return prompt_str


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    example_index: int = 0,
    layer: int = 36,
    heads: Optional[List[int]] = None,
    top_k_sentences: int = 5,
    top_k_heads: int = 5,
    output_dir: str = "results/attention_viz",
    streamlit_folder: str = "data/streamlit",
    dataset_name: str = "gpqa",
    include_first_sentence: bool = False,
    gap: Optional[int] = None,
):
    """
    Main function to visualize attention patterns.

    Args:
        model_name: HuggingFace model name
        example_index: Index of example from base_data.json
        layer: Layer index to extract attention from
        heads: Optional list of specific heads to analyze. If None, analyze all.
        top_k_sentences: Number of top sentences to store per head
        top_k_heads: Number of top heads (by kurtosis) to plot individual matrices for
        output_dir: Directory to save outputs
        streamlit_folder: Path to streamlit data folder
        dataset_name: Name of dataset (e.g., 'gpqa')
        include_first_sentence: If True, include the first sentence in analysis
        gap: If provided, only consider sentence pairs that are at least this many sentences apart
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # Required to get attention weights (flash attention doesn't return them)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_nickname = MODEL_METADATA[model_name]['nickname']

    # Load example from base_data.json
    base_data_path = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
    print(f"Loading example {example_index} from {base_data_path}")
    with open(base_data_path) as f:
        base_data = json.load(f)[example_index]

    # Create prompt
    prompt = make_prompt_mcq(base_data, tokenizer)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_token_ids = torch.tensor(
        prompt_token_ids + base_data["output_token_ids"], 
        device=model.device
    ).unsqueeze(0)

    # Get convergence index
    idx_str = str(example_index).zfill(2)
    file_template = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{{idx}}.csv"
    convergence_result = get_convergence_for_index(idx_str, file_template)
    convergence_token_idx = convergence_result[0] if convergence_result else None
    convergence_outcome = convergence_result[1] if convergence_result else None
    print(f"Convergence at token {convergence_token_idx}, outcome: {convergence_outcome}")

    # Extract attention patterns
    print(f"Extracting attention patterns from layer {layer}...")
    inputs = {"input_ids": full_token_ids}
    attention = get_attention_patterns(model, inputs, layer, heads)
    print(f"Attention shape: {attention.shape}")  # (num_heads, seq_len, seq_len)

    # Split into sentences
    print("Splitting tokens into sentences...")
    sentences = split_tokens_into_sentences(full_token_ids.squeeze(), tokenizer, min_sentence_length=10)
    print(f"Found {len(sentences)} sentences")

    # Optionally exclude first sentence (e.g., to skip prompt/system message)
    sentence_offset = 0
    if not include_first_sentence and len(sentences) > 1:
        print("Excluding first sentence from analysis")
        sentences = sentences[1:]
        sentence_offset = 1
        print(f"Analyzing {len(sentences)} sentences (after exclusion)")

    # Find which sentence contains convergence token
    convergence_sentence_idx = None
    if convergence_token_idx is not None:
        # Adjust for prompt offset since convergence_token_idx is relative to output
        adjusted_convergence_idx = len(prompt_token_ids) + convergence_token_idx
        convergence_sentence_idx = get_sentence_for_token(adjusted_convergence_idx, sentences)
        print(f"Convergence is in sentence {convergence_sentence_idx}")

    # Compute vertical scores at token level
    # print("Computing vertical scores...")
    # vertical_scores = compute_vertical_scores(attention)  # (num_heads, seq_len)

    # Aggregate attention by sentences
    print("Aggregating attention by sentences...")
    sentence_attention = aggregate_attention_by_sentences(attention, sentences, aggregation='mean')
    print(f"Sentence attention shape: {sentence_attention.shape}")  # (num_heads, num_sentences, num_sentences)

    # Zero out diagonal (self-attention of sentences to themselves)
    sentence_attention = zero_diagonal(sentence_attention)

    # Apply gap filter if specified (only consider pairs at least 'gap' sentences apart)
    if gap is not None:
        print(
            f"Applying gap filter: only considering pairs at least {gap} sentences apart"
        )
        sentence_attention = apply_gap_filter(sentence_attention, gap)

    # Compute vertical scores at sentence level
    sentence_vertical_scores = compute_vertical_scores(sentence_attention)  # (num_heads, num_sentences)

    # Kurtosis analysis
    print("Computing kurtosis for each head...")
    top_heads_kurtosis = get_top_kurtosis_heads(sentence_vertical_scores, k=top_k_heads)
    print(f"Top {top_k_heads} heads by kurtosis:")
    for head_idx, kurtosis in top_heads_kurtosis:
        print(f"  Head {head_idx}: kurtosis = {kurtosis:.4f}")

    # Get top-k sentences per head
    print(f"\nTop {top_k_sentences} sentences per head (by vertical score):")
    top_sentences = get_top_k_sentences_per_head(
        sentence_vertical_scores, 
        sentences, 
        k=top_k_sentences
    )

    # Print top sentences for high-kurtosis heads
    for head_idx, _ in top_heads_kurtosis:
        print(f"\n  Head {head_idx}:")
        for sent_idx, score, sentence in top_sentences[head_idx]:
            print(f"    Sentence {sent_idx} (tokens {sentence.start}-{sentence.end}): score={score:.6f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    gap_suffix = f"_gap{gap}" if gap is not None else ""
    example_output_dir = os.path.join(
        output_dir, f"example_{idx_str}_layer_{layer}{gap_suffix}"
    )
    os.makedirs(example_output_dir, exist_ok=True)

    # Generate and save plots
    print(f"\nGenerating plots in {example_output_dir}...")

    # Plot vertical scores for all heads
    fig, ax = plot_vertical_attention_scores(
        sentence_vertical_scores,
        head_to_highlight=top_heads_kurtosis[0][0] if top_heads_kurtosis else None,
        layer=layer,
        convergence_sentence_idx=convergence_sentence_idx
    )
    fig.savefig(os.path.join(example_output_dir, "vertical_scores_all_heads.png"), dpi=150, bbox_inches='tight')
    print("  Saved: vertical_scores_all_heads.png")

    # Plot attention matrices for top kurtosis heads
    for head_idx, kurtosis in top_heads_kurtosis:
        fig, ax = plot_attention_matrix(
            sentence_attention[head_idx],
            head_idx=head_idx,
            layer=layer,
            convergence_sentence_idx=convergence_sentence_idx
        )
        filename = f"attention_matrix_head_{head_idx}.png"
        fig.savefig(os.path.join(example_output_dir, filename), dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")

    # Save results to JSON
    results = {
        "model_name": model_name,
        "example_index": example_index,
        "layer": layer,
        "num_sentences": len(sentences),
        "include_first_sentence": include_first_sentence,
        "sentence_offset": sentence_offset,
        "gap": gap,
        "convergence_token_idx": convergence_token_idx,
        "convergence_sentence_idx": convergence_sentence_idx,
        "convergence_outcome": convergence_outcome,
        "top_kurtosis_heads": [
            {"head": h, "kurtosis": float(k)} for h, k in top_heads_kurtosis
        ],
        "top_sentences_per_head": {
            str(h): [
                {
                    "sentence_idx": s[0],
                    "score": float(s[1]),
                    "token_start": s[2].start,
                    "token_end": s[2].end,
                }
                for s in lst
            ]
            for h, lst in top_sentences.items()
            if h in [x[0] for x in top_heads_kurtosis]  # Only save for top heads
        },
        "sentences": [
            {"idx": i, "start": s.start, "end": s.end} for i, s in enumerate(sentences)
        ],
    }

    results_path = os.path.join(example_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results.json")

    print("\nDone!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention patterns")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--example_index", type=int, default=0)
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--heads", type=int, nargs="+", default=None, 
                        help="Specific heads to analyze (default: all)")
    parser.add_argument("--top_k_sentences", type=int, default=5)
    parser.add_argument("--top_k_heads", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/attention_viz")
    parser.add_argument("--streamlit_folder", type=str, default="data/streamlit")
    parser.add_argument("--dataset_name", type=str, default="gpqa")
    parser.add_argument("--include_first_sentence", action="store_true",
                        help="Include first sentence in analysis (useful to include prompt)")
    parser.add_argument(
        "--gap",
        type=int,
        default=4,
        help="Only consider sentence pairs at least this many sentences apart (e.g., 4)",
    )

    args = parser.parse_args()
    main(**vars(args))
