import os
import json
import random
from typing import List, Optional, Literal
from collections import Counter
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.utils import MODEL_METADATA, clear_cuda, set_seed
from utils.prompt_utils import get_cot_prompt, get_alignment_prompt
from utils.data_utils import DATASET_TO_FORMAT


def load_data(
    tokenizer: PreTrainedTokenizer,
    dataset: dict,
    n: int = 100,
    shuffle: bool = True,
    start_index: int = 0,
    math_min_level: int = 3,
):
    """
    Load given dataset, using the provided tokenizer to wrap the question in an LLM prompt.

    tokenizer : transformers.PreTrainedTokenizer
        tokenizer for model to analyze
    dataset : dict
        name, source, hf_name, hf_split, type (from datasets metadata)
    n : int
        number of examples to load
    shuffle : True
        whether or not to randomly sample the n examples or just take the first n
    start_index : int
        offset into the (shuffled) dataset to begin sampling from. Use this together
        with `n` to extend an earlier collection without re-doing the same examples.
        prompt_id of each returned datapoint will be `start_index + idx` so ids do
        not collide with earlier runs.
    math_min_level : int
        minimum hendrycks_math difficulty level to keep (only applies to MATH_open)

    Returns
    list[dict]
        loaded dataset with question, correct_letter, correct_answer, all_letters, all_answers, type, and prompt
    """
    # load dataset
    print(f"Loading {dataset['name']} dataset...")
    if dataset["name"] == "MATH_open":
        # EleutherAI/hendrycks_math has 7 separate configs; load and concatenate all
        math_configs = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        hf_dataset = concatenate_datasets([
            load_dataset(dataset["source"], cfg, split=dataset["hf_split"])
            for cfg in math_configs
        ])
        # Keep only problems with difficulty level >= math_min_level
        hf_dataset = hf_dataset.filter(
            lambda x: int(x["level"].split()[-1]) >= math_min_level
        )
    elif dataset["hf"]:
        hf_dataset = load_dataset(
            dataset["source"], dataset["hf_name"], split=dataset["hf_split"]
        )
    else:
        with open(f"{dataset['source']}/test.jsonl") as f:
            json_dataset = [json.loads(line) for line in f]
        # ignore weird Task_id in PythonIO task
        json_dataset = [
            {k: v for k, v in d.items() if k != "Task_id"} for d in json_dataset
        ]
        hf_dataset = Dataset.from_list(json_dataset)

    # optionally shuffle dataset
    if shuffle:
        hf_dataset = hf_dataset.shuffle(seed=42)

    # sample `n` prompts from the dataset
    # do stratified sampling for alignment datasets
    if dataset["name"] == "WildJailBreak":
        dataset_refuse = hf_dataset.filter(
            lambda x: x["data_type"] == "adversarial_harmful"
        ).select(range(n // 2))
        dataset_comply = hf_dataset.filter(
            lambda x: x["data_type"] != "adversarial_harmful"
        ).select(range(n // 2))
        hf_dataset = concatenate_datasets([dataset_refuse, dataset_comply])
    elif dataset["name"] == "Just-Eval":
        dataset_refuse = hf_dataset.filter(lambda x: x["category"] == "safety").select(
            range(n // 2)
        )
        dataset_comply = hf_dataset.filter(lambda x: x["category"] != "safety").select(
            range(n // 2)
        )
        hf_dataset = concatenate_datasets([dataset_refuse, dataset_comply])
    else:
        end_index = min(start_index + n, len(hf_dataset))
        if start_index >= len(hf_dataset):
            raise ValueError(
                f"start_index={start_index} is past the end of the {dataset['name']} "
                f"dataset (len={len(hf_dataset)} after filtering)"
            )
        hf_dataset = hf_dataset.select(range(start_index, end_index))
        if end_index - start_index < n:
            print(
                f"Warning: only {end_index - start_index} examples available from "
                f"{dataset['name']} starting at index {start_index} (requested {n})"
            )

    data = []
    for idx, example in enumerate(hf_dataset):
        datapoint = DATASET_TO_FORMAT[dataset["name"]](example)
        if dataset["type"] == "alignment":
            prompt_str = get_alignment_prompt(
                tokenizer, datapoint["question_with_choices"], alignment_type=None
            )
        else:
            is_mc = dataset["type"] == "multiple choice"
            prompt_str = get_cot_prompt(
                tokenizer, datapoint["question_with_choices"], multiple_choice=is_mc
            )

        data.append(
            {
                **datapoint,  # pass down info about datapoint
                "prompt_id": start_index + idx,  # track prompt index for re-mapping
                "dataset_name": dataset["name"],
                "dataset_type": dataset["type"],
                "prompt": prompt_str,
            }
        )

    return data


def generate_all_paths_batched(
    llm: LLM,
    dataset: List[dict],
    num_paths: int = 10,
    max_new_tokens: int = 10000,
    temperature: float = 0.6,
    batch_size: int = 32,
    return_logprobs: bool = True,
):
    """
    Generate num_paths samples per prompt using temperature sampling (no greedy).
    Processes prompts in batches, generating all paths per prompt in one call.

    llm : vllm.LLM
        vLLM model for generation.
    dataset : list[dict]
        entries from load_data function
    num_paths : int
        number of paths to generate per prompt
    max_new_tokens : int
        maximum length of generated paths
    temperature : float
        sampling temperature (default 0.6)
    batch_size : int
        number of prompts to process in parallel
    return_logprobs : bool
        whether to return the log probabilities

    Returns
    list[dict]
        list of all generated paths with rollout_id, prompt_id, and metadata
    """
    all_results = []
    sampling_params = SamplingParams(
        n=num_paths,  # Generate all paths per prompt in one call
        temperature=temperature,
        max_tokens=max_new_tokens,
        logprobs=0 if return_logprobs else None,
    )

    # Batch over prompts (not prompt × rollout pairs)
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        prompts = [datapoint["prompt"] for datapoint in batch]
        outputs = llm.generate(prompts, sampling_params)

        for datapoint, output in zip(batch, outputs):
            prompt_id = datapoint["prompt_id"]
            # Each output has num_paths completions
            for rollout_id, completion in enumerate(output.outputs):
                result = {
                    "prompt_id": prompt_id,
                    "rollout_id": rollout_id,
                    "output_text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "prompt_token_ids": output.prompt_token_ids,
                    "output_token_ids": completion.token_ids,
                    **datapoint,
                }

                if return_logprobs and completion.logprobs:
                    result["output_logprobs"] = [
                        completion.logprobs[t][completion.token_ids[t]].logprob
                        for t in range(len(completion.token_ids))
                    ]

                all_results.append(result)

    return all_results


def select_base_answer(
    paths_for_prompt: List[dict],
    base_answer_type: Literal["correct", "incorrect", "mode"],
    correct_answer: Optional[str] = None,
) -> Optional[dict]:
    """
    Select a path to use as the "base" answer based on the selection strategy.
    Excludes paths that hit the token limit (finish_reason == 'length') from base selection.

    paths_for_prompt : list[dict]
        all generated paths for a single prompt (with parsed answers)
    base_answer_type : str
        'correct': select a random path with the correct answer
        'incorrect': select a random path with an incorrect answer
        'mode': select a random path with the most common answer
    correct_answer : str, optional
        the ground truth correct answer (needed for 'correct'/'incorrect' modes)

    Returns
    dict or None
        the selected path to use as base, with 'base': True added
        returns None if no valid candidates (all paths hit token limit)
    """
    # Filter out paths that hit the token limit for base selection
    valid_paths = [p for p in paths_for_prompt if p["finish_reason"] != "length"]

    if not valid_paths:
        # All paths hit the token limit
        return None

    answers = [p["clean_answer"] for p in valid_paths]

    if base_answer_type == "mode":
        # Find the most common answer among valid paths
        answer_counts = Counter(answers)
        mode_answer = answer_counts.most_common(1)[0][0]
        # Get all valid paths with the mode answer and pick one randomly
        candidates = [p for p in valid_paths if p["clean_answer"] == mode_answer]

    elif base_answer_type == "correct":
        # Get all valid paths with the correct answer
        candidates = [p for p in valid_paths if p["clean_answer"] == correct_answer]
        if not candidates:
            # Fallback to mode if no correct answers found
            print(
                f"Warning: No correct answers found for prompt_id={paths_for_prompt[0]['prompt_id']}, falling back to mode"
            )
            answer_counts = Counter(answers)
            mode_answer = answer_counts.most_common(1)[0][0]
            candidates = [p for p in valid_paths if p["clean_answer"] == mode_answer]

    elif base_answer_type == "incorrect":
        # Get all valid paths with incorrect answers
        candidates = [p for p in valid_paths if p["clean_answer"] != correct_answer]
        if not candidates:
            # Fallback to mode if all answers are correct
            print(
                f"Warning: All answers correct for prompt_id={paths_for_prompt[0]['prompt_id']}, falling back to mode"
            )
            answer_counts = Counter(answers)
            mode_answer = answer_counts.most_common(1)[0][0]
            candidates = [p for p in valid_paths if p["clean_answer"] == mode_answer]
    else:
        raise ValueError(f"Unknown base_answer_type: {base_answer_type}")

    # Randomly select one of the candidates
    selected = random.choice(candidates)
    selected_with_base = {**selected, "base": True}

    return selected_with_base


def aggregate_results(
    parsed_results: List[dict],
    num_paths: int,
    base_answer_type: Literal["correct", "incorrect", "mode"],
    return_alternate_texts: bool = True,
):
    """
    Aggregate all paths, select base answer, and compute uncertainty metrics.

    parsed_results : list[dict]
        all generated and parsed paths
    num_paths : int
        number of paths per prompt
    base_answer_type : str
        strategy for selecting base answer
    return_alternate_texts : bool
        whether to return texts of alternate paths

    Returns
    list[dict]
        aggregated results sorted by uncertainty
    """
    # Group results by prompt_id
    from collections import defaultdict

    paths_by_prompt = defaultdict(list)
    for result in parsed_results:
        paths_by_prompt[result["prompt_id"]].append(result)

    aggregated_results = []
    skipped_prompts = 0

    for prompt_id in sorted(paths_by_prompt.keys()):
        paths = paths_by_prompt[prompt_id]

        # Get correct answer if available (for correct/incorrect selection)
        correct_answer = paths[0].get("correct_letter") or paths[0].get("correct_answer")

        # Select the base answer (excludes paths that hit token limit)
        base_data = select_base_answer(paths, base_answer_type, correct_answer)

        # Skip this prompt if no valid base answer (all paths hit token limit)
        if base_data is None:
            print(
                f"Skipping prompt_id={prompt_id}: all {len(paths)} paths hit token limit"
            )
            skipped_prompts += 1
            continue

        # Get alternate paths (all paths that aren't the selected base, including token-limited ones for uncertainty calculation)
        alternate_paths = [p for p in paths if p["rollout_id"] != base_data["rollout_id"]]

        # Calculate metrics
        base_answer_rate = sum(
            [alt["clean_answer"] == base_data["clean_answer"] for alt in alternate_paths]
        )
        base_answer_cut_short = base_data["finish_reason"] == "length"
        num_random_cut_short = sum(
            [alt["finish_reason"] == "length" for alt in alternate_paths]
        )

        aggregated_result = {
            "base_answer_rate": base_answer_rate,
            "base_answer_cut_short": base_answer_cut_short,
            "num_random_cut_short": num_random_cut_short,
            "alternate_answers": [alt["clean_answer"] for alt in alternate_paths],
            "alternate_finish_reasons": [alt["finish_reason"] for alt in alternate_paths],
            "base_answer_type": base_answer_type,
            "all_sampled_answers": [
                p["clean_answer"] for p in paths
            ],  # keep track of all answers
            **base_data,
        }

        if return_alternate_texts:
            aggregated_result["alternate_texts"] = [
                alt["output_text"] for alt in alternate_paths
            ]

        aggregated_results.append(aggregated_result)

    # Sort by uncertainty (same logic as original)
    def uncertainty_score(datapoint):
        """Very ad hoc uncertainty/entropy score, where we want to ignore answers that went over the max token limit"""
        score = 0
        if datapoint["base_answer_cut_short"]:
            score += 1000
        score += 100 * datapoint["num_random_cut_short"]
        score += datapoint["base_answer_rate"]
        return score

    aggregated_results.sort(key=uncertainty_score)

    if skipped_prompts > 0:
        print(f"Skipped {skipped_prompts} prompts where all paths hit token limit")

    return aggregated_results


def main(
    model_name: str,
    dataset_names: str,
    # data selection parameters
    num_examples: int = 100,
    shuffle: bool = True,
    start_index: int = 0,
    math_min_level: int = 3,
    # generation parameters
    num_paths: int = 10,
    max_new_tokens: int = 10000,
    temperature: float = 0.6,
    batch_size: int = 32,
    return_logprobs: bool = True,
    enable_prefix_caching: bool = True,
    quantization: Optional[str] = None,
    tensor_parallel_size: int = 1,
    # base answer selection
    base_answer_type: Literal["correct", "incorrect", "mode"] = "mode",
    # output parameters
    return_alternate_texts: bool = True,
    output_suffix: str = "",
    no_append: bool = False,
    seed: int = 42,
    # HuggingFace upload
    hf_repo_id: Optional[str] = None,
):
    set_seed(seed)
    random.seed(seed)  # for base answer selection

    with open("config.json") as f:
        config = json.load(f)
        dataset_metadata_filename = config["save_locations"]["dataset_metadata_file"]
        data_dir = config["save_locations"]["collection_folder"]
        answer_model_name = config["experiment_parameters"]["answer_model"]

    with open(dataset_metadata_filename) as f:
        datasets_metadata = json.load(f)

    model_nickname = MODEL_METADATA[model_name]["nickname"]
    output_dir = f"{data_dir}/{model_nickname}"
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in dataset_names.split(","):
        print(f"Generating paths for {dataset_name} with temperature={temperature}...")

        # Load generation LLM
        gen_llm = LLM(
            model=model_name,
            dtype="auto",
            enable_prefix_caching=enable_prefix_caching,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
        )

        dataset = load_data(
            gen_llm.get_tokenizer(),
            datasets_metadata[dataset_name],
            n=num_examples,
            shuffle=shuffle,
            start_index=start_index,
            math_min_level=math_min_level,
        )

        # Generate all paths in parallel batches (no greedy, all temperature sampled)
        all_paths = generate_all_paths_batched(
            gen_llm,
            dataset,
            num_paths=num_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            return_logprobs=return_logprobs,
        )

        # Clear generation LLM
        del gen_llm
        clear_cuda()

        # Parse answers
        answer_llm = LLM(model=answer_model_name, dtype="bfloat16")
        parsed_results = parse_answer(answer_llm, all_paths)

        # Aggregate and select base answers
        aggregated_results = aggregate_results(
            parsed_results,
            num_paths=num_paths,
            base_answer_type=base_answer_type,
            return_alternate_texts=return_alternate_texts,
        )

        # Save results (append by default if file exists)
        output_filename = f"{output_dir}/{dataset_name.lower()}{output_suffix}.json"
        if os.path.exists(output_filename) and not no_append:
            with open(output_filename) as f:
                existing = json.load(f)
            existing_pids = {r["prompt_id"] for r in existing}
            new_records = [r for r in aggregated_results if r["prompt_id"] not in existing_pids]
            collisions = len(aggregated_results) - len(new_records)
            if collisions > 0:
                print(
                    f"Skipping {collisions} new record(s) whose prompt_id already exists "
                    f"in {output_filename} (use --no_append to overwrite instead)."
                )
            combined = existing + new_records  # no re-sort: keep existing positions stable
            with open(output_filename, "w") as f:
                json.dump(combined, f, indent=2)
            print(
                f"Appended {len(new_records)} new records to {output_filename} "
                f"(was {len(existing)}, now {len(combined)})."
            )
        else:
            with open(output_filename, "w") as f:
                json.dump(aggregated_results, f, indent=2)
            print(f"Saved {len(aggregated_results)} results to {output_filename}")

        # Optionally upload to HuggingFace Hub
        if hf_repo_id is not None:
            from datasets import Dataset as HFDataset
            hf_split_name = f"{model_nickname}/{dataset_name.lower()}"
            hf_upload = HFDataset.from_list(aggregated_results)
            hf_upload.push_to_hub(hf_repo_id, config_name=hf_split_name)
            print(f"Uploaded {len(aggregated_results)} results to {hf_repo_id} (config: {hf_split_name})")

        # Clear answer LLM
        del answer_llm
        clear_cuda()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data collection with temperature sampling and configurable base answer selection"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model to analyze")
    parser.add_argument(
        "--dataset_names",
        type=str,
        required=True,
        help="Comma-separated list of dataset names",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to sample from each dataset",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle the dataset before sampling"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help=(
            "Offset into the (shuffled) dataset before taking num_examples. "
            "Use to extend an earlier collection without re-doing the same examples. "
            "prompt_id starts at start_index."
        ),
    )
    parser.add_argument(
        "--math_min_level",
        type=int,
        default=3,
        help="Minimum hendrycks_math difficulty level (1-5) to keep, MATH_open only.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "Suffix appended to the output filename, before .json (e.g. '_v2'). "
            "By default empty, so the output filename is '<dataset>.json'."
        ),
    )
    parser.add_argument(
        "--no_append",
        action="store_true",
        help=(
            "If set, overwrite any existing output file rather than appending to it. "
            "By default we append: read the existing file, drop any new records whose "
            "prompt_id already appears, concat existing+new, and write back."
        ),
    )
    parser.add_argument(
        "--num_paths", type=int, default=10, help="Number of paths to generate per prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10000,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling (default: 0.6)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for parallel generation"
    )
    parser.add_argument(
        "--return_logprobs",
        action="store_true",
        help="Return log probabilities for paths",
    )
    parser.add_argument(
        "--return_alternate_texts",
        action="store_true",
        help="Return texts of alternate paths",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--enable_prefix_caching", action="store_true", help="Enable prefix caching"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size for the generation LLM (number of GPUs to shard across).",
    )
    parser.add_argument(
        "--base_answer_type",
        type=str,
        default="mode",
        choices=["correct", "incorrect", "mode"],
        help="How to select base answer: 'correct' (random correct), 'incorrect' (random incorrect), 'mode' (most common)",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="HuggingFace repo ID to upload results to (e.g. 'org/dataset-name'). Each dataset is uploaded as a separate config named '<model>/<dataset>'.",
    )
    args = parser.parse_args()

    main(**vars(args))
