import argparse
import json
import os
import random
from typing import List
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.utils import MODEL_METADATA, SENTENCE_DELIMITERS, clear_cuda, set_seed


def collect_stumps(
    llm : LLM,
    prompt_token_ids : List[int],
    output_token_ids : List[int]
):
    """
    Collect all starts of branches (i.e.m stumps) at which we're forking. Use existing generated base path.

    llm : vllm.LLM
        vLLM model for log probabilities.
    prompt_token_ids : list[int]
        Tokenized prompt (shared across all paths)
    output_token_ids : list[int]
        Tokenized base path after prompt (shared across all paths)
    max_new_tokens : int
        Maximum number of alternative tokens to consider at each token position.
    num_branches : float
        Minimum probability to create a branch at the altenative token.

    Returns
    list[dict]
        Collection of stumps (places in base path at which we want to branch off).
        stump = {token_ids: ..., t: ..., token_id : ..., token_prob: ...} 
    """
    delimiter_token_ids = llm.get_tokenizer().convert_tokens_to_ids(SENTENCE_DELIMITERS)

    stumps = []
    cutoff = 0
    for i in range(len(output_token_ids)):
        if output_token_ids[i] in delimiter_token_ids:
            stump_token_ids = output_token_ids[cutoff:i+1] # include punctuation at the end
            stumps.append({
                "stump_token_ids": stump_token_ids,
                "prompt_and_stump_token_ids": prompt_token_ids + stump_token_ids,
                "t": cutoff # where we started, not where we finished!
            })
        cutoff = i + 1

    return stumps

def generate_branches(
    llm : LLM,
    stumps : List,
    num_branches : int,
    max_new_tokens : int,
    temperature : float
):
    """
    Starting at each stump in our collection, sample continuations (i.e., branches).

    llm : vllm.LLM
        vLLM model for generation.
    stumps : list[dict]
        List of stumps, each of which is a list of token ids.
        stump = {token_ids: ..., t: ..., token_id : ..., token_prob: ...} 
    num_branches : int
        Number of new branches to generate for each stump.
    max_new_tokens : int
        Maximum length of each branch (not counting the stump).

    Returns
    list[dict]
        Records for each branch generated. Includes information about where we branched, the generated text,
        the stopping reason, and the probability that the branch was sampled.
    """
    # generate branches!
    sampling_params = SamplingParams(
        n=num_branches, # output num_branches paths for each stump
        temperature=temperature, # random sampling
        logprobs=0, # return logprobs for sampled branch
        max_tokens=max_new_tokens
    )
    # put in vLLM input format (generate from full path, so include prompt ids)
    llm_inputs = [{'prompt_token_ids': stump["prompt_and_stump_token_ids"]} for stump in stumps]
    branch_outputs = llm.generate(llm_inputs, sampling_params)

    # post-process into ans_df
    branch_results = []
    for i in range(len(branch_outputs)):
        stump = stumps[i]
        for branch_output in branch_outputs[i].outputs: # go through generated outputs
            output_token_ids = stump['stump_token_ids'] + branch_output.token_ids # put together stump + branch (everything after prompt)
            output_text = llm.get_tokenizer().decode(output_token_ids, skip_special_tokens=True)
            branch_results.append({
                # stump data
                't': stump['t'], # fork token index
                # branch data
                'output_text': output_text, # full text (after prompt)
                'post_stump_output_text': branch_output.text, # branched text (after stump)
                'finish_reason': branch_output.finish_reason,
                'output_length': len(branch_output.token_ids),
                'cumulative_logprob': branch_output.cumulative_logprob,
                'norm_cumulative_logprob': branch_output.cumulative_logprob * (1 / max(1, len(branch_output.token_ids))),
                # answer data added later!
            })

    return branch_results

def main(
    model_name : str = "gpt2",
    dataset_name : str = "AQuA",
    dataset_size : int = 10,
    # forking paths parameters
    num_branches : int = 30,
    max_new_tokens : int = 10000,
    temperature : float = 0.7,
    # control parameters
    seed : int = 42
):
    set_seed(seed)

    with open('config.json') as f:
        config = json.load(f)
        data_dir = config["save_locations"]["data_folder"] # input
        forking_paths_dir = config["save_locations"]["forking_paths_folder"] # output
        answer_model_name = config["experiment_parameters"]["answer_model"]

    # load input
    model_nickname = MODEL_METADATA[model_name]['nickname']
    with open(f'{data_dir}/{model_nickname}/{dataset_name.lower()}') as f:
        dataset = json.load(f)

    # create output dir
    output_dir = f'{forking_paths_dir}/{model_nickname}/{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)

    # load LLM
    base_llm = LLM(model=model_name, dtype="bfloat16")

    # process dataset!
    num_examples = len(dataset) if dataset_size is None else min(dataset_size, len(dataset))
    for prompt_index in range(num_examples):
        # skip if already processed
        result_path = os.path.join(output_dir, f'{prompt_index:02d}.json')
        if os.path.exists(result_path):
            print(f"Results for prompt #{prompt_index} already exist, skipping")
            continue

        print("Question:")
        print(dataset[prompt_index]["question"])
        print("Base path:")
        print(dataset[prompt_index]["output_text"])

        # collect stumps for prompt
        stumps = collect_stumps(
            base_llm,
            dataset[prompt_index]['prompt_token_ids'],
            dataset[prompt_index]['output_token_ids'],
        )

        print("Done collecting stumps.")
        print(f"Number of stumps: {len(stumps)}")
        print("-" * 30)
        
        random_stump = random.choice(stumps)
        print("-" * 30)
        print(f"Random stump (t = {random_stump['t']}):")
        print(base_llm.get_tokenizer().decode(random_stump["stump_token_ids"], skip_special_tokens=True))

        # generate branches for prompt
        branches = generate_branches(
            base_llm,
            stumps,
            num_branches=num_branches,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # save results
        with open(result_path, "w+") as f:
            json.dump(branches, f, indent=2)
        print(f"Saving results to {result_path}")

    # switch to answer model, and extract answers here
    if answer_model_name is None:
        return # done here!

    print("Parsing final answers")
    # clear cache
    del base_llm
    clear_cuda()

    # load answer LLM
    answer_llm = LLM(model=answer_model_name, dtype="bfloat16")

    # parse results
    for prompt_index in range(num_examples):
        # not the prettiest, but re-read the results we just generated
        print(f"Parsing answers for prompt #{prompt_index}")
        result_path = os.path.join(output_dir, f'{prompt_index:02d}.json')
        with open(result_path) as f:
            branches = json.load(f)
        
        # copy relevant information from datapoint
        datapoint = dataset[prompt_index]
        branch_dataset = [{
            "dataset_type": datapoint["dataset_type"],
            "question": datapoint["question"],
            "all_answers": datapoint["all_answers"],
            "all_letters": datapoint["all_letters"],
            **branch # output_text comes from branch!
        } for branch in branches]

        # feed generated answers (ext_full) into answer extraction prompt template
        parse_results = parse_answer(
            answer_llm,
            branch_dataset
        )

        # save results to same path (overwrite)
        # NOTE: we're copying info from base answer; this is redundant! can remove info if taking up too much space
        with open(result_path, "w+") as f:
            json.dump(parse_results, f, indent=2)
        print(f"Saving parsed results to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forking paths with vLLM")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="AQuA", help="Dataset name")
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of examples to process (leave as None to process entire dataset)")
    parser.add_argument("--num_branches", type=int, default=30, help="Number of branches to generate")
    parser.add_argument("--max_new_tokens", type=int, default=3000, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(**vars(args))
            