import argparse
import json
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.cot_analysis import split_tokens_into_sentences
from utils.circuit_discovery.factory import get_circuit_discovery_algorithm
from utils.objectives import kl_logits_objective


DEFAULT_PROMPT = (
    "The capital of France is Paris. Answer in 100 words or less, "
    "what are the most popular things in the city to do?"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunk_sentences(sentences, sentence_chunk: int) -> List[dict]:
    if sentence_chunk <= 1:
        return [{"start": s.start, "end": s.end} for s in sentences]

    chunks = []
    for i in range(0, len(sentences), sentence_chunk):
        start = sentences[i].start
        end = sentences[min(i + sentence_chunk - 1, len(sentences) - 1)].end
        chunks.append({"start": start, "end": end})
    return chunks


def parse_layers(layers_str: str) -> List[int]:
    return [int(x.strip()) for x in layers_str.split(",") if x.strip()]


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    prompt: str = DEFAULT_PROMPT,
    num_new_branches: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    masking_algorithm: str = "EAP",
    objective: str = "kl_logits",
    analysis_timestep: int = None,
    layers_to_analyse: str = "8,12,16,20,24",
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    seed: int = 42,
    num_steps: int = 50,
    lr: float = 1e-1,
    output_dir: str = "results/mask_learning",
):
    set_seed(seed)
    layers = parse_layers(layers_to_analyse)

    if masking_algorithm.lower() != "eap":
        raise ValueError("Only EAP masking is implemented.")
    if objective != "kl_logits":
        raise ValueError("Only kl_logits objective is implemented.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    if analysis_timestep is None:
        analysis_timestep = prompt_len
    analysis_timestep = min(analysis_timestep, prompt_len)

    prefix_ids = torch.tensor(prompt_ids[:analysis_timestep])
    sentences = split_tokens_into_sentences(prefix_ids, tokenizer, min_sentence_length=1)
    sentence_chunks = chunk_sentences(sentences, sentence_chunk)

    # vLLM sampling
    sampling_params = SamplingParams(
        n=num_new_branches,
        temperature=temperature,
        logprobs=0,
        max_tokens=max_new_tokens,
    )
    llm = LLM(model=model_name, dtype="auto")
    outputs = llm.generate([formatted_text], sampling_params)
    branch_token_ids = [out.token_ids for out in outputs[0].outputs]
    branch_texts = [out.text for out in outputs[0].outputs]
    del llm

    circuit = get_circuit_discovery_algorithm(
        "eap",
        model_name=model_name,
        layers=layers,
        analysis_timestep=analysis_timestep,
        sentence_chunks=sentence_chunks,
        sentence_gap=sentence_gap,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    edge_mask, metrics = circuit.learn_mask(
        prompt=formatted_text,
        prompt_token_ids=prompt_ids,
        branch_token_ids=branch_token_ids,
        objective_fn=kl_logits_objective,
        num_steps=num_steps,
        lr=lr,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    edge_mask.to_json(os.path.join(run_dir, "mask.json"))
    metrics_payload = {
        "config": {
            "model_name": model_name,
            "prompt": prompt,
            "analysis_timestep": analysis_timestep,
            "layers_to_analyse": layers,
            "sentence_gap": sentence_gap,
            "sentence_chunk": sentence_chunk,
            "num_new_branches": num_new_branches,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "masking_algorithm": masking_algorithm,
            "objective": objective,
            "num_steps": num_steps,
            "lr": lr,
            "seed": seed,
        },
        "losses": metrics["losses"],
        "branches": branch_texts,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved mask to {os.path.join(run_dir, 'mask.json')}")
    print(f"Saved metrics to {os.path.join(run_dir, 'metrics.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn EAP mask over attention patterns")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--masking_algorithm", type=str, default="EAP")
    parser.add_argument("--objective", type=str, default="kl_logits")
    parser.add_argument("--analysis_timestep", type=int, default=None)
    parser.add_argument("--layers_to_analyse", type=str, default="8,12,16,20,24")
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--output_dir", type=str, default="results/mask_learning")
    args = parser.parse_args()
    main(**vars(args))
