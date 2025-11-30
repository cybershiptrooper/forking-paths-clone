
from typing import Counter, List
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.probe_utils import get_activations
from utils.utils import clear_cuda, set_seed

def steer(
    
):
    # 1. break up by stumps
    # - list of prompt_token_ids of length T
    pass

    # 2. generate completion for n ~= 10 samples with steering
    pass

    # 3. return results
    pass


def generate_rollouts(
    llm : LLM,
    prompt_token_ids : List[int],
    # rollout parameters
    num_paths : int = 500,
    temperature : float = 0.7,
    max_new_tokens : int = 10000,
    # data parameters
    **datapoint_kwargs
):
    """
    Generate many random paths to collect steering vectors per outcome.

    llm : vllm.LLM
        vLLM model for generation.
    prompt_token_ids : List[int]
        input ids for the prompt
    """
    sampling_params = SamplingParams(
        n=num_paths,
        temperature=temperature, # truly random sampling??
        max_tokens=max_new_tokens
    )
    output = llm.generate([{"prompt_token_ids": prompt_token_ids}], sampling_params)[0]
    results = []
    for i in range(num_paths):
        results.append({
            'output_text': output.outputs[i].text,
            'finish_reason' : output.outputs[i].finish_reason,
            'prompt_token_ids': output.prompt_token_ids,
            'output_token_ids': output.outputs[i].token_ids,
            **datapoint_kwargs
        })

    return results


def run_steering_experiment(
    model_name : str,
    answer_model_name : str,
    base_data : dict,
    # rollout parameters
    num_paths : int = 500,
    temperature : float = 0.7,
    max_new_tokens : int = 10000,
    # steering parameters
    layer : int = 0,
    token_index : int = -1,
    num_outcomes_to_steer : int = 3
):
    # 1. roll out N times & parse outcomes
    base_llm = LLM(model=model_name, dtype="bfloat16")
    rollouts = generate_rollouts(
        base_llm, 
        base_data["prompt_token_ids"],
        num_paths=num_paths,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        # propagate base data params
        dataset_type=base_data["dataset_type"],
        question=base_data["question"],
        all_letters=base_data["all_letters"],
        all_answers=base_data["all_answers"],
    )

    del base_llm
    clear_cuda()
    
    answer_llm = LLM(model=answer_model_name, dtype="bfloat16")
    rollouts_with_outcomes = parse_answer(
        answer_llm,
        rollouts
    )

    del answer_llm
    clear_cuda()

    # 2. collect activations from rollouts
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
    
    inputs = tokenizer.pad(
        {"input_ids": [r["prompt_token_ids"] + r["output_token_ids"] for r in rollouts_with_outcomes]},
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    activations = get_activations(model, inputs, layer)[:, token_index, :] # (N rollouts, hidden dim)

    # 3. create steering vector per outcome
    steering_vectors = []
    outcome_counts = Counter([r["clean_answer"] for r in rollouts_with_outcomes])
    for outcome, _ in outcome_counts.most_common(num_outcomes_to_steer):
        outcome_indices = [
            i for i in range(len(rollouts_with_outcomes)) 
            if rollouts_with_outcomes[i]["clean_answer"] == outcome
        ]
        other_outcome_indices = [
            i for i in range(len(rollouts_with_outcomes)) 
            if rollouts_with_outcomes[i]["clean_answer"] != outcome
        ]

        # do we want to subsample?? (not sure if it'll really do much)

        steering_vector = torch.mean(activations[outcome_indices], dim=0) - torch.mean(activations[other_outcome_indices], dim=0)
        steering_vectors.append({
            "outcome": outcome,
            "steering_vector": steering_vector
        })

    del model, tokenizer
    clear_cuda()

    # 4. steer towards each outcome
    # - easy steer looks annoying as heck...
    pass

    # 5. parse outcomes
    pass


def main(
    model_name : str = "gpt2",
    dataset_name : str = "AQuA",  
    seed : int = 42
):
    set_seed(seed)
    # 1. load all forking paths
    pass

    # 2. run steering
    pass

    # 3. save results
    pass