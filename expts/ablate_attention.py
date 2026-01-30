"""
Attention ablation experiment script.

Loads attention visualization results, selects top sentences to ablate,
runs model generation with ablation, and checks if the final answer changes.
"""

import argparse
import json
import os
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer
from vllm import LLM

from utils.activation_patching import ablate_sentences, get_model
from utils.attention_analysis import select_top_sentences_by_mean_score
from utils.answer_utils import parse_answer
from utils.cot_analysis import get_convergence_for_index
from utils.prompt_utils import get_cot_prompt
from utils.utils import MODEL_METADATA, Sentence


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


def adjust_sentence_indices_for_context(
    sentences: List[Sentence],
    convergence_token_idx: int,
    prompt_length: int
) -> List[Sentence]:
    """
    Adjust sentence token indices for generation context.
    
    Sentences from results.json are relative to full sequence (prompt + full output).
    When generating from convergence token, we need to:
    - Filter out sentences that start after convergence token (not in context)
    - Keep sentences entirely before convergence token as-is
    - Clip sentences that span convergence token (adjust end to convergence)
    
    Args:
        sentences: List of Sentence objects with indices relative to full sequence
        convergence_token_idx: Token index where convergence occurs (relative to output)
        prompt_length: Length of prompt in tokens
        
    Returns:
        List of adjusted Sentence objects that exist in the generation context
    """
    adjusted = []
    full_sequence_convergence_idx = prompt_length + convergence_token_idx
    
    for sentence in sentences:
        # Check if sentence is entirely before convergence
        if sentence.end < full_sequence_convergence_idx:
            # Sentence is entirely in context, keep as-is
            adjusted.append(sentence)
        elif sentence.start < full_sequence_convergence_idx:
            # Sentence spans convergence, clip it
            clipped = Sentence(
                start=sentence.start,
                end=full_sequence_convergence_idx - 1
            )
            adjusted.append(clipped)
        # If sentence.start >= full_sequence_convergence_idx, skip it (not in context)
    
    return adjusted


def parse_layers_arg(layers_str: str) -> Union[int, List[int], str]:
    """
    Parse layers argument from command line.
    
    Args:
        layers_str: "all", single int, or comma-separated list of ints
        
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
    model_name: Optional[str] = None,
    example_index: int = 0,
    results_dir: str = "results/attention_viz",
    num_sentences_to_ablate: int = 10,
    layers: str = "all",
    max_new_tokens: int = 10000,
    output_dir: str = "results/attention_ablation",
    streamlit_folder: str = "data/streamlit",
    dataset_name: str = "gpqa",
    config: Optional[dict] = None,
):
    """
    Main function to run attention ablation experiment.
    
    Args:
        model_name: Model name (if None, will try to get from results.json)
        example_index: Index of example from base_data.json
        results_dir: Path to attention_viz results directory
        num_sentences_to_ablate: Number of top sentences to ablate
        layers: Layers to ablate - "all", single int, or comma-separated list
        max_new_tokens: Max tokens to generate
        output_dir: Directory to save ablation results
        streamlit_folder: Path to streamlit data folder
        dataset_name: Name of dataset (e.g., 'gpqa')
    """
    # Parse layers argument
    layers_parsed = parse_layers_arg(layers)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name or 'from results'}")
    model_nickname = MODEL_METADATA.get(model_name, {}).get('nickname') if model_name else None
    
    # Load example from base_data.json
    if model_nickname is None and model_name:
        # Try to infer nickname
        for full_name, metadata in MODEL_METADATA.items():
            if full_name == model_name:
                model_nickname = metadata['nickname']
                break
    
    if model_nickname is None:
        raise ValueError(f"Could not determine model nickname for {model_name}. Please check MODEL_METADATA.")
    
    base_data_path = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
    print(f"Loading example {example_index} from {base_data_path}")
    with open(base_data_path) as f:
        base_data = json.load(f)[example_index]
    
    # Get model name from results if not provided
    if model_name is None:
        # Try to find a results file to get model name
        idx_str = str(example_index).zfill(2)
        # Look for any results file for this example
        example_dirs = [d for d in os.listdir(results_dir) if d.startswith(f"example_{idx_str}_")]
        if example_dirs:
            results_path = os.path.join(results_dir, example_dirs[0], "results.json")
            with open(results_path) as f:
                results_data = json.load(f)
                model_name = results_data.get("model_name")
                if model_name:
                    print(f"Using model name from results: {model_name}")
                    model_nickname = MODEL_METADATA[model_name]['nickname']
    
    if model_name is None:
        raise ValueError("Could not determine model_name. Please provide --model_name or ensure results.json exists.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create prompt
    prompt = make_prompt_mcq(base_data, tokenizer)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_length = len(prompt_token_ids)
    
    # Get convergence index
    idx_str = str(example_index).zfill(2)
    file_template = f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{{idx}}.csv"
    convergence_result = get_convergence_for_index(idx_str, file_template)
    if convergence_result is None:
        raise ValueError(f"No convergence found for example {example_index}")
    convergence_token_idx = convergence_result[0]
    convergence_outcome = convergence_result[1]
    print(f"Convergence at token {convergence_token_idx}, outcome: {convergence_outcome}")
    
    # Get base answer
    base_answer = base_data.get("clean_answer", convergence_outcome)
    print(f"Base answer: {base_answer}")
    
    # Load attention visualization results
    # Find results directory for this example
    example_dirs = [d for d in os.listdir(results_dir) if d.startswith(f"example_{idx_str}_")]
    if not example_dirs:
        raise ValueError(f"No attention visualization results found for example {example_index} in {results_dir}")
    
    # Use the first matching directory (or could add logic to select specific layer/gap)
    results_subdir = example_dirs[0]
    results_path = os.path.join(results_dir, results_subdir, "results.json")
    print(f"Loading attention results from {results_path}")
    with open(results_path) as f:
        attention_results = json.load(f)
    
    # Extract top sentences per head
    top_sentences_per_head = attention_results.get("top_sentences_per_head", {})
    if not top_sentences_per_head:
        raise ValueError(f"No top_sentences_per_head found in results.json")
    
    # Select top K sentences by mean score
    print(f"Selecting top {num_sentences_to_ablate} sentences by mean score...")
    sentences_to_ablate = select_top_sentences_by_mean_score(
        top_sentences_per_head,
        k=num_sentences_to_ablate
    )
    print(f"Selected {len(sentences_to_ablate)} sentences to ablate")
    for i, sent in enumerate(sentences_to_ablate):
        print(f"  {i+1}. Sentence: tokens {sent.start}-{sent.end}")
    
    # Adjust sentence indices for generation context
    adjusted_sentences = adjust_sentence_indices_for_context(
        sentences_to_ablate,
        convergence_token_idx,
        prompt_length
    )
    print(f"After adjusting for context: {len(adjusted_sentences)} sentences remain")
    for i, sent in enumerate(adjusted_sentences):
        print(f"  {i+1}. Sentence: tokens {sent.start}-{sent.end}")
    
    if not adjusted_sentences:
        print("Warning: No sentences to ablate after adjusting for context!")
        return
    
    # Prepare generation context (prompt up to convergence token)
    output_token_ids = base_data["output_token_ids"]
    prefix_token_ids = prompt_token_ids + output_token_ids[:convergence_token_idx]
    prompt_str = tokenizer.decode(prefix_token_ids, skip_special_tokens=False)
    
    print(f"\nGenerating with ablation from convergence token...")
    print(f"Prompt length: {len(prefix_token_ids)} tokens")
    
    # Run ablation experiment using existing function
    nnsight_model = get_model(model_name)
    ablate_sentences(
        nnsight_model,
        adjusted_sentences,
        layers=layers_parsed,
        prompt=prompt_str,
        max_new_tokens=max_new_tokens
    )
    
    # Retrieve output from nnsight model
    # Based on user example: model.generator.output.save()
    generated_text = None
    try:
        if hasattr(nnsight_model, 'generator') and nnsight_model.generator is not None:
            # Get the output tokens
            output_tokens = nnsight_model.generator.output.save()
            
            # Decode the generated tokens
            if hasattr(output_tokens, 'tolist'):
                token_list = output_tokens.tolist()
            elif isinstance(output_tokens, list):
                token_list = output_tokens
            elif hasattr(output_tokens, 'cpu'):
                # If it's a tensor
                token_list = output_tokens.cpu().tolist()
            else:
                # Try to convert to list
                token_list = list(output_tokens) if hasattr(output_tokens, '__iter__') else [output_tokens]
            
            # Decode tokens to text
            generated_text = tokenizer.decode(token_list, skip_special_tokens=True)
        else:
            print("Warning: model.generator not available. Output may not be accessible.")
            generated_text = None
    except Exception as e:
        print(f"Error retrieving output: {e}")
        import traceback
        traceback.print_exc()
        generated_text = None
    
    if generated_text is None:
        print("Warning: Could not retrieve generated text. The ablation may have run, but output is not accessible.")
        print("You may need to check nnsight documentation for output retrieval.")
        generated_text = ""
    
    print(f"\nGenerated text (first 500 chars):\n{generated_text[:500]}...")
    
    # Parse ablated answer
    # Need to use vLLM for answer parsing
    print("\nParsing ablated answer...")
    if config is None:
        with open('config.json') as f:
            config = json.load(f)
    answer_model_name = config.get('experiment_parameters', {}).get('answer_model', 'meta-llama/Llama-3.2-1B-Instruct')
    answer_llm = LLM(model=answer_model_name)
    
    # Prepare data for parsing
    ablated_datapoint = {
        "output_text": generated_text,
        "question": base_data["question"],
        "all_letters": base_data["all_letters"],
        "all_answers": base_data["all_answers"],
        "dataset_type": "multiple choice"
    }
    
    parsed_results = parse_answer(answer_llm, [ablated_datapoint])
    ablated_answer = parsed_results[0]["clean_answer"] if parsed_results else None
    
    print(f"Ablated answer: {ablated_answer}")
    
    # Compare answers
    answer_changed = (base_answer != ablated_answer) if ablated_answer else None
    
    print(f"\n{'='*60}")
    print(f"Base answer: {base_answer}")
    print(f"Ablated answer: {ablated_answer}")
    print(f"Answer changed: {answer_changed}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model_name": model_name,
        "example_index": example_index,
        "layers_ablated": str(layers_parsed),
        "num_sentences_to_ablate": num_sentences_to_ablate,
        "convergence_token_idx": convergence_token_idx,
        "convergence_outcome": convergence_outcome,
        "sentences_ablated_original": [
            {"start": s.start, "end": s.end} for s in sentences_to_ablate
        ],
        "sentences_ablated_adjusted": [
            {"start": s.start, "end": s.end} for s in adjusted_sentences
        ],
        "base_answer": base_answer,
        "ablated_answer": ablated_answer,
        "answer_changed": answer_changed,
        "generated_text": generated_text[:10000] if generated_text else None,  # Limit length
    }
    
    output_filename = f"ablation_example_{idx_str}_layers_{layers}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention ablation experiment")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (if None, will try to get from results.json)")
    parser.add_argument("--example_index", type=int, default=0,
                        help="Example index from base_data.json")
    parser.add_argument("--results_dir", type=str, default="results/attention_viz",
                        help="Path to attention_viz results directory")
    parser.add_argument("--num_sentences_to_ablate", type=int, default=10,
                        help="Number of top sentences to ablate")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to ablate: 'all', single int, or comma-separated list")
    parser.add_argument("--max_new_tokens", type=int, default=10000,
                        help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default="results/attention_ablation",
                        help="Directory to save ablation results")
    parser.add_argument("--streamlit_folder", type=str, default="data/streamlit",
                        help="Path to streamlit data folder")
    parser.add_argument("--dataset_name", type=str, default="gpqa",
                        help="Dataset name")
    
    args = parser.parse_args()
    
    # Load config for answer model
    import json as json_module
    with open('config.json') as f:
        config = json_module.load(f)
    
    main(**vars(args))
