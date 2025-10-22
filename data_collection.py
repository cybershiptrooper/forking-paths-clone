import os
import json
import yaml
from typing import List
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.utils import MODEL_METADATA, clear_cuda, set_seed
from utils.prompt_utils import get_cot_prompt, get_alignment_prompt
from utils.data_utils import DATASET_TO_FORMAT


def load_data(
    tokenizer : PreTrainedTokenizer, 
    dataset : dict, 
    n : int = 100,
    shuffle : bool = True
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

    Returns 
    list[dict]
        loaded dataset with question, correct_letter, correct_answer, all_letters, all_answers, type, and prompt
    """
    # load dataset
    print(f"Loading {dataset['name']} dataset...")
    if dataset['hf']:
        hf_dataset = load_dataset(dataset['source'], dataset['hf_name'], split=dataset['hf_split'])
    else:
        with open(f"{dataset['source']}/test.jsonl") as f:
            json_dataset = [json.loads(line) for line in f]
        # ignore weird Task_id in PythonIO task
        json_dataset = [{k: v for k, v in d.items() if k != 'Task_id'} for d in json_dataset]
        hf_dataset = Dataset.from_list(json_dataset)

    # optionally shuffle dataset
    if shuffle:
        hf_dataset = hf_dataset.shuffle(seed=42)

    # sample `n` prompts from the dataset
    # do stratified sampling for alignment datasets
    if dataset['name'] == 'WildJailBreak':
        dataset_refuse = hf_dataset.filter(lambda x: x['data_type'] == 'adversarial_harmful').select(range(n//2))
        dataset_comply = hf_dataset.filter(lambda x: x['data_type'] != 'adversarial_harmful').select(range(n//2))
        hf_dataset = concatenate_datasets([dataset_refuse, dataset_comply])
    elif dataset['name'] == 'Just-Eval':
        dataset_refuse = hf_dataset.filter(lambda x: x['category'] == 'safety').select(range(n//2))
        dataset_comply = hf_dataset.filter(lambda x: x['category'] != 'safety').select(range(n//2))
        hf_dataset = concatenate_datasets([dataset_refuse, dataset_comply])
    elif len(hf_dataset) > n:
        hf_dataset = hf_dataset.select(range(n))

    data = []
    for example in hf_dataset:
        datapoint = DATASET_TO_FORMAT[dataset['name']](example)
        if dataset['type'] == "alignment":
            prompt_str = get_alignment_prompt(tokenizer, datapoint['question_with_choices'], alignment_type=None)
        else:
            prompt_str = get_cot_prompt(tokenizer, datapoint['question_with_choices'])

        data.append({
            **datapoint, # pass down info about datapoint
            'dataset_name': dataset['name'],
            'dataset_type': dataset['type'], 
            'prompt': prompt_str
        })

    return data

def generate_base_paths(
    llm : LLM,
    dataset : List[dict],
    max_new_tokens : int = 10000,
    return_logprobs : bool = True
):
    """
    Generate base path with greedy decoding.

    llm : vllm.LLM
        vLLM model for generation.
    dataset : list[dict]
        entries from load_data function
    max_new_tokens : int
        maximum length of base path
    return_logprobs : bool
        whether to return the log probabilities of the base path (used in streamlit visualizations)

    Returns 
    list[dict]
        updated dataset with base text, tokenized prompt, tokenized base path, and (optionally) log probs
    """
    prompts = [d['prompt'] for d in dataset]
    # return logprobs for base path (used for steering analysis)
    sampling_params = SamplingParams(
        temperature=0., # greedy sampling
        max_tokens=max_new_tokens,
        logprobs=0 if return_logprobs else None,
    )
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output, datapoint in zip(outputs, dataset):
        result = {
            'output_text': output.outputs[0].text,
            'finish_reason' : output.outputs[0].finish_reason,
            'prompt_token_ids': output.prompt_token_ids,
            'output_token_ids': output.outputs[0].token_ids,
            "base": True,
            **datapoint
        }

        # get log probability for each token
        if return_logprobs:
            result['output_logprobs'] = [
                output.outputs[0].logprobs[t][output.outputs[0].token_ids[t]].logprob
                for t in range(len(output.outputs[0].token_ids))
            ]

        results.append(result)

    return results

def generate_alternate_paths(
    llm : LLM,
    dataset : List[dict],
    max_new_tokens : int = 10000,
    num_paths : int = 10,
):
    """
    Generate a few random paths to quickly estimate uncertainty.

    llm : vllm.LLM
        vLLM model for generation.
    dataset : list[dict]
        entries from load_data function
    max_new_tokens : int
        maximum length of random paths
    """
    prompts = [d['prompt'] for d in dataset]
    sampling_params = SamplingParams(
        n=num_paths,
        temperature=1.0, # truly random sampling??
        max_tokens=max_new_tokens
    )
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output, datapoint in zip(outputs, dataset):
        for i in range(num_paths):
            results.append({
                'output_text': output.outputs[i].text,
                'finish_reason' : output.outputs[i].finish_reason,
                "base": False,
                **datapoint
            })

    return results

def sort_by_uncertainty(
    base_results : List[dict],
    alternates_results : List[dict],
    num_paths : int,
    return_alternate_texts : bool = True
):
    """
    Aggregate base paths & alternate paths (n x base) to estimate uncertainty for each datapoint,
    and sort by uncertainty (least certain first) for forking paths analysis.

    base_results : list[dict]
        list of base paths + extracted answers
    alternates_results : list[dict]
        list of n randomly sampled paths per base path + extracted answers
    num_paths : int
        number of randomly sampled paths per base path
    return_alternate_texts : bool
        whether to return texts of random paths (not used during analysis, so can drop to save space)

    Returns
    list[dict]
        aggregated results with base path data + info about random paths, sorted by uncertainty
    """
    aggregated_results = []
    for i in range(len(base_results)):
        base_data = base_results[i]
        alternate_paths = [alternates_results[i * num_paths + n] for n in range(num_paths)]
        assert all([alt['prompt'] == base_data['prompt'] for alt in alternate_paths]) # should all have the same prompt
        # how often do the sampled answers match the base answer?
        base_answer_rate = sum([alt['clean_answer'] == base_data['clean_answer'] for alt in alternate_paths])
        # how often did the model blab for longer than intended? (discard these)
        base_answer_cut_short = (base_data['finish_reason'] == 'length')
        num_random_cut_short = sum([alt['finish_reason'] == 'length' for alt in alternate_paths])

        aggregated_result = {
            "base_answer_rate": base_answer_rate,
            "base_answer_cut_short": base_answer_cut_short,
            "num_random_cut_short": num_random_cut_short,
            "alternate_answers": [alt['clean_answer'] for alt in alternate_paths],
            'alternate_finish_reasons': [alt['finish_reason'] for alt in alternate_paths],
            **base_data
        }
        if return_alternate_texts:
            aggregated_result['alternate_texts'] = [alt['output_text'] for alt in alternate_paths]

        aggregated_results.append(aggregated_result)
    
    def uncertainty_score(datapoint):
        """Very ad hoc uncertainty/entropy score, where we want to ignore answers that went over the max token limit"""
        score = 0
        # filter out long answers
        if datapoint['base_answer_cut_short']:
            score += 1000 # immediately discard base answers that are too long
        score += 100 * datapoint['num_random_cut_short'] # likewise, don't want random answers cut short
        # score by # of times alternate answers == base path answer (lower is better)
        score += datapoint['base_answer_rate']
        return score

    aggregated_results.sort(key=uncertainty_score)

    return aggregated_results
    
def main(
    model_name : str,
    dataset_names : str,
    # data selection parameters
    num_examples : int = 100, 
    shuffle : bool = True,
    # generation parameters
    num_paths : int = 10,
    max_new_tokens : int = 10000,
    return_logprobs : bool = True,
    # output parameters
    return_alternate_texts : bool = True,
    seed : int = 42
):
    set_seed(seed) # just in case

    with open('config.json') as f:
        config = json.load(f)
        dataset_metadata_filename = config["save_locations"]["dataset_metadata_file"]
        data_dir = config["save_locations"]["data_folder"]
        answer_model_name = config["experiment_parameters"]["answer_model"]

    with open(dataset_metadata_filename) as f:
        datasets_metadata = json.load(f)

    model_nickname = MODEL_METADATA[model_name]['nickname']
    output_dir = f'{data_dir}/{model_nickname}'
    os.makedirs(output_dir, exist_ok=True)

    # generate base paths for each dataset
    for dataset_name in dataset_names.split(','):
        print(f"Generating base paths for {dataset_name}...")
        # not great, but re-load base / answer LLM for each dataset separately
        base_llm = LLM(model=model_name, dtype="bfloat16")

        dataset = load_data(base_llm.get_tokenizer(), datasets_metadata[dataset_name], n=num_examples, shuffle=shuffle)
        
        base_paths = generate_base_paths(base_llm, dataset, max_new_tokens=max_new_tokens, return_logprobs=return_logprobs)
        alternate_paths = generate_alternate_paths(base_llm, dataset, max_new_tokens=max_new_tokens, num_paths=num_paths)

        # clear cache
        del base_llm
        clear_cuda()

        answer_llm = LLM(model=answer_model_name, dtype="bfloat16")
        
        parse_results = parse_answer(
            answer_llm,
            base_paths + alternate_paths # preserve metadata
        )
        base_results = parse_results[:len(base_paths)]
        alternates_results = parse_results[len(base_paths):]

        data_selection_results = sort_by_uncertainty(
            base_results,
            alternates_results,
            num_paths=num_paths,
            return_alternate_texts=return_alternate_texts
        )

        # save results
        with open(f"{output_dir}/{dataset_name.lower()}.json", 'w') as f:
            json.dump(data_selection_results, f, indent=2)

        # clear cache ahead of next dataset (not the cleanest, but oh well)
        del answer_llm
        clear_cuda()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model to analyze")
    parser.add_argument("--dataset_names", type=str, required=True, help="Comma-separated list of dataset names")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to sample from each dataset")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the dataset before sampling")
    parser.add_argument("--num_paths", type=int, default=10, help="Number of alternate paths to generate")
    parser.add_argument("--max_new_tokens", type=int, default=10000, help="Maximum number of new tokens to generate")
    parser.add_argument("--return_logprobs", action='store_true', help="Return log probabilities for base paths")
    parser.add_argument("--return_alternate_texts", action='store_true', help="Return texts of randomly sampled alternative paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(**vars(args))