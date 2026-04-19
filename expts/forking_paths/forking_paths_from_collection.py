"""Generate forking paths from a specific prompt in a collection JSON.

Thin wrapper around the forking_paths pipeline that:
- Loads a prompt from a collection JSON (e.g. math_filtered.json) by index
- Uses split_tokens_into_sentences for stump identification (same boundaries
  as circuit discovery)
- Keeps the same structure and defaults as forking_paths.py

Usage:
    uv run python -m expts.forking_paths.forking_paths_from_collection \
        --data_path data/collection/deepseek_llama_8b/math_filtered.json \
        --prompt_index 2 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

import argparse
import json
import os
import random
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.cot_analysis import split_tokens_into_sentences
from utils.utils import MODEL_METADATA, set_seed, clear_cuda


def collect_stumps_from_sentences(
    tokenizer,
    prompt_token_ids: List[int],
    output_token_ids: List[int],
    min_sentence_length: int = 10,
):
    """Collect stumps (forking points) using split_tokens_into_sentences.

    This ensures sentence boundaries match those used by circuit discovery.

    Returns list of stump dicts compatible with generate_branches().
    """
    token_ids_tensor = torch.tensor(output_token_ids)
    sentences = split_tokens_into_sentences(
        token_ids_tensor, tokenizer, min_sentence_length=min_sentence_length
    )

    stumps = []
    for s in sentences:
        stump_token_ids = output_token_ids[: s.end + 1]
        stumps.append({
            "stump_token_ids": stump_token_ids,
            "prompt_and_stump_token_ids": prompt_token_ids + stump_token_ids,
            "t": s.end + 1,
        })

    return stumps


def generate_branches(
    llm: LLM,
    stumps: List,
    num_branches: int,
    max_new_tokens: int,
    temperature: float,
):
    """Starting at each stump, sample continuations (branches).

    Identical to forking_paths.generate_branches.
    """
    sampling_params = SamplingParams(
        n=num_branches,
        temperature=temperature,
        logprobs=0,
        max_tokens=max_new_tokens,
    )
    llm_inputs = [
        {"prompt_token_ids": stump["prompt_and_stump_token_ids"]}
        for stump in stumps
    ]
    branch_outputs = llm.generate(llm_inputs, sampling_params)

    branch_results = []
    for i in range(len(branch_outputs)):
        stump = stumps[i]
        for branch_output in branch_outputs[i].outputs:
            output_token_ids = list(stump["stump_token_ids"]) + list(
                branch_output.token_ids
            )
            output_text = llm.get_tokenizer().decode(
                output_token_ids, skip_special_tokens=True
            )
            branch_results.append({
                "t": stump["t"],
                "output_text": output_text,
                "post_stump_output_text": branch_output.text,
                "finish_reason": branch_output.finish_reason,
                "output_length": len(branch_output.token_ids),
                "cumulative_logprob": branch_output.cumulative_logprob,
                "norm_cumulative_logprob": branch_output.cumulative_logprob
                * (1 / max(1, len(branch_output.token_ids))),
            })

    return branch_results


def main(
    data_path: str,
    prompt_index: int,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    num_branches: int = 30,
    max_new_tokens: int = 16384,
    temperature: float = 0.6,
    enable_prefix_caching: bool = True,
    quantization: Optional[str] = None,
    min_sentence_length: int = 10,
    seed: int = 42,
    output_dir: Optional[str] = None,
    only_parse_answers: bool = False,
):
    set_seed(seed)

    # Load the collection record
    with open(data_path) as f:
        dataset = json.load(f)

    if prompt_index >= len(dataset):
        raise ValueError(
            f"prompt_index {prompt_index} out of range "
            f"(dataset has {len(dataset)} records)"
        )

    record = dataset[prompt_index]
    prompt_id = record["prompt_id"]
    print(f"Loaded prompt_index={prompt_index} (prompt_id={prompt_id})")
    print(f"  Question: {record['question'][:120]}...")

    if record["finish_reason"] != "stop":
        print(f"Base answer was cut short (finish_reason={record['finish_reason']}), skipping")
        return

    # Output directory: default to data/forking_paths/<model>/<dataset>/
    if output_dir is None:
        with open("config.json") as f:
            config = json.load(f)
        model_nickname = MODEL_METADATA[model_name]["nickname"]
        ds_name = record.get("dataset_name", "unknown").lower()
        output_dir = os.path.join(
            config["save_locations"]["forking_paths_folder"],
            model_nickname,
            ds_name,
        )
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{prompt_index:02d}.json")

    if not only_parse_answers:
        if os.path.exists(result_path):
            print(f"Results already exist at {result_path}, skipping generation")
        else:
            # Load vLLM
            base_llm = LLM(
                model=model_name,
                dtype="auto",
                enable_prefix_caching=enable_prefix_caching,
                quantization=quantization,
            )

            print(f"\nBase path:\n{record['output_text'][:300]}...\n")

            # Collect stumps using the same sentence splitter as circuit discovery
            stumps = collect_stumps_from_sentences(
                base_llm.get_tokenizer(),
                record["prompt_token_ids"],
                record["output_token_ids"],
                min_sentence_length=min_sentence_length,
            )
            print(f"Number of stumps: {len(stumps)}")

            if not stumps:
                print("No stumps found, skipping")
                del base_llm
                clear_cuda()
                return

            random_stump = random.choice(stumps)
            print(f"Random stump (t = {random_stump['t']}):")
            print(
                base_llm.get_tokenizer().decode(
                    random_stump["stump_token_ids"], skip_special_tokens=True
                )[:200]
            )
            print("-" * 30)

            # Generate branches
            branches = generate_branches(
                base_llm,
                stumps,
                num_branches=num_branches,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            with open(result_path, "w") as f:
                json.dump(branches, f, indent=2)
            print(f"Saved {len(branches)} branches to {result_path}")

            del base_llm
            clear_cuda()

    # Parse answers
    print("\nParsing final answers...")
    with open("config.json") as f:
        config = json.load(f)
    answer_model_name = config["experiment_parameters"]["answer_model"]

    if not os.path.exists(result_path):
        print(f"No results at {result_path}, nothing to parse")
        return

    with open(result_path) as f:
        branches = json.load(f)

    answer_llm = LLM(model=answer_model_name, dtype="bfloat16")

    branch_dataset = [
        {
            "dataset_type": record.get("dataset_type", "open ended"),
            "question": record["question"],
            "all_answers": record.get("all_answers", []),
            "all_letters": record.get("all_letters", []),
            **branch,
        }
        for branch in branches
    ]

    parse_results = parse_answer(answer_llm, branch_dataset)

    with open(result_path, "w") as f:
        json.dump(parse_results, f, indent=2)
    print(f"Saved parsed results to {result_path}")

    # Save metadata alongside results
    meta_path = os.path.join(output_dir, f"{prompt_index:02d}_meta.json")
    meta = {
        "prompt_index": prompt_index,
        "prompt_id": prompt_id,
        "data_path": data_path,
        "question": record["question"],
        "correct_answer": record.get("correct_answer"),
        "model_name": model_name,
        "num_branches": num_branches,
        "temperature": temperature,
        "seed": seed,
        "min_sentence_length": min_sentence_length,
        "num_stumps": len(stumps) if "stumps" in dir() else None,
        "num_branches_generated": len(branches),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    del answer_llm
    clear_cuda()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate forking paths from a collection JSON prompt"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to collection JSON (e.g. data/collection/deepseek_llama_8b/math_filtered.json)",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        required=True,
        help="Index into the JSON array (e.g. 2 for the 3rd record)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument("--num_branches", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_sentence_length", type=int, default=10)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--only_parse_answers",
        action="store_true",
        help="Only parse answers from existing results (skip generation)",
    )

    args = parser.parse_args()
    main(**vars(args))
