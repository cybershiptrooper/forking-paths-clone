import argparse
import datetime
import json
import os
from typing import List, Optional
from collections import Counter
import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils.answer_utils import parse_answer
from utils.probe_utils import get_activations
from utils.utils import MODEL_METADATA, clear_cuda, set_seed

def steer(
    tokenizer,
    model,
    prompt_token_ids : List[int],
    output_token_ids : List[int],
    ts : List[int],
    layer : int,
    steering_vector : torch.Tensor,
    temperature : float = 0.7,
    max_new_tokens : int = 10000,
    num_samples : int = 10,
    batch_size : int = 8,
    # data parameters (passed down after steering result)
    **datapoint_kwargs
):
    # 1. break up by stumps
    # - list of prompt_token_ids of length T
    input_ids = [
        prompt_token_ids + output_token_ids[:t]
        for t in ts
    ]
    inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 2. generate completion for n ~= 10 samples with steering
    def steer_hook(module, input, output):
        # output = (batch size * num samples, token length or 1, hidden dim)
        output[:, -1, :] = output[:, -1, :].clone() + steering_vector
        return output
    
    steer_hook_handle = model.model.layers[layer].register_forward_hook(steer_hook)

    outputs = [] # (Ts * num samples, output_length)
    print(f"Steering: {inputs['input_ids'].shape}")
    for b in trange(0, len(inputs["input_ids"]), batch_size, desc="Steering..."):
        batch_inputs = {
            "input_ids": inputs["input_ids"][b:b + batch_size],
            "attention_mask": inputs["attention_mask"][b:b + batch_size]
        }
        batch_outputs = model.generate(
            **batch_inputs, 
            do_sample=True, 
            num_return_sequences=num_samples,
            temperature=temperature, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )[:, batch_inputs["input_ids"].shape[1]:] # (batch size * num samples, output length)
        outputs += tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

    steer_hook_handle.remove()

    results = []
    for t_index, t in enumerate(ts):
        for sample_index in range(num_samples):
            results.append({
                "t": int(t),
                "output_text": outputs[t_index * num_samples + sample_index], # regroup outputs
                **datapoint_kwargs
            })

    # 3. return results
    return results


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
    print("Generated rollouts:", len(output.outputs))
    print(output.outputs[0].text)
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
    ts : List[int],
    # rollout parameters
    num_paths : int = 500,
    temperature : float = 0.7,
    max_new_tokens : int = 10000,
    # steering parameters
    layer : int = 0,
    token_index : int = -1,
    num_outcomes_to_steer : int = 3,
    num_steer_samples : int = 10,
    batch_size : int = 8
):
    # 1. roll out N times & parse outcomes
    base_llm = LLM(model=model_name, dtype="bfloat16")
    tokenizer = base_llm.get_tokenizer()
    print("Prompt:", base_data["question"])
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": base_data["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    base_data["prompt_token_ids"] = prompt_token_ids

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
    outcome_counts = Counter([r["clean_answer"] for r in rollouts_with_outcomes])
    print("Outcome counts:")
    print(outcome_counts)

    del answer_llm
    clear_cuda()

    # 2. collect activations from rollouts
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id # set padding token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)

    inputs = tokenizer.pad(
        {"input_ids": [r["prompt_token_ids"] + r["output_token_ids"] for r in rollouts_with_outcomes]},
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print("Collection activations:", inputs['input_ids'].shape)
    activations = get_activations(model, inputs, layer, batch_size=4)[:, token_index, :] # (N rollouts, hidden dim)

    # 3. create steering vector per outcome
    steering_vectors = []
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
        if len(outcome_indices) == 0:
            print("Skipping - should contain at least one example of the outcome")
            continue
        if len(other_outcome_indices) == 0:
            print("Skipping - should contain at least one example alternative")
            continue

        steering_vector = torch.mean(activations[outcome_indices], dim=0) - torch.mean(activations[other_outcome_indices], dim=0)
        steering_vectors.append({
            "outcome": outcome,
            "steering_vector": steering_vector
        })

    # 4. steer towards each outcome
    # - easy steer looks annoying as heck...
    steering_results = []
    for steer_data in steering_vectors:
        steering_results += steer(
            tokenizer,
            model,
            base_data["prompt_token_ids"],
            base_data["output_token_ids"],
            ts,
            layer,
            steer_data["steering_vector"].to(model.device).to(model.dtype),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_samples=num_steer_samples,
            batch_size=batch_size,
            # info to pass down
            steer_outcome=steer_data["outcome"],
            # propagate base data for answer parsing
            dataset_type=base_data["dataset_type"],
            question=base_data["question"],
            all_letters=base_data["all_letters"],
            all_answers=base_data["all_answers"],
        )
        clear_cuda()

    del model, tokenizer
    clear_cuda()

    # 5. parse outcomes
    answer_llm = LLM(model=answer_model_name, dtype="bfloat16")
    steering_results_with_outcomes = parse_answer(
        answer_llm,
        steering_results
    )
    del answer_llm
    clear_cuda()

    # might want to remove output text!
    return steering_results_with_outcomes # len = Ts * num steer; keys = [t, steer_outcome, clean_answer, output_text, ... (some base data stuff)]


def main(
    model_name : str = "gpt2",
    dataset_name : str = "AQuA",
    # rollout parameters
    num_paths : int = 500,
    # generation parameters
    temperature : float = 0.7,
    max_new_tokens : int = 10000,
    # steering parameters
    layer : int = 0,
    token_index : int = -1,
    num_outcomes_to_steer : int = 3,
    num_steer_samples : int = 10,
    batch_size : int = 8,
    # experiment parameters
    start_index : Optional[int] = None,
    end_index : Optional[int] = None,  
    seed : int = 42
):
    set_seed(seed)

    with open('config.json') as f:
        config = json.load(f)
    answer_model_name = config['experiment_parameters']['answer_model']
    streamlit_folder = config['save_locations']['streamlit_folder']
    steer_folder = config['save_locations']['steer_folder']
    model_nickname = MODEL_METADATA[model_name]['nickname']
    example_ids = sorted([
        filename.split('.')[0] for filename in os.listdir(f'{streamlit_folder}/{model_nickname}/{dataset_name.lower()}')
        if filename != "base_data.json" # ignore base data
    ])

    for i, example_index in enumerate(example_ids):
        # optionally specify range of examples
        if start_index is not None and i < start_index:
            continue
        if end_index is not None and i >= end_index:
            continue

        # 1. load base data & timestamps
        with open(f'{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json') as f:
            base_data = json.load(f)[int(example_index)]
        # load ts
        outcome_df = pd.read_csv(f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{example_index}.csv")
        timestamps = sorted(outcome_df.t.unique()) # (T,)

        print(f"Steering example #{example_index}")
        print(base_data["question"])

        # 2. run steering
        steering_results = run_steering_experiment(
            model_name,
            answer_model_name,
            base_data,
            timestamps,
            # rollout parameters
            num_paths=num_paths,
            # generation parameters
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # steering parameters
            layer=layer,
            token_index=token_index,
            num_outcomes_to_steer=num_outcomes_to_steer,
            num_steer_samples=num_steer_samples,
            batch_size=batch_size
        )

        results = {
            "hyperparameters": {
                "layer": layer,
                "token_index": token_index
            },
            "results": steering_results
        }

        # 3. save results
        output_dir = f"{steer_folder}/{model_nickname}/{dataset_name.lower()}/{example_index}" 
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        with open(f"{output_dir}/results-{now}.json", "w+") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset_name', type=str, default='AQuA')
    parser.add_argument('--num_paths', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=10000)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--token_index', type=int, default=-1)
    parser.add_argument('--num_outcomes_to_steer', type=int, default=3)
    parser.add_argument('--num_steer_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))
