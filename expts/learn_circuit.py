"""Circuit discovery experiment using nodewise attribution patching.

Uses a simple illustrating example (from controlled_ablations_v2.py) with a
prompt about Paris, generates branches via vLLM, then runs nodewise attribution
with an HF model loaded with eager attention.

Usage:
    uv run python expts/learn_circuit.py
    uv run python expts/learn_circuit.py --num_new_branches 4 --layers_to_analyse 8,12,16
"""

import argparse
import gc
import os
from typing import List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.activation_patching_effecient import load_custom_model_eager
from utils.circuit_discovery import get_algorithm
from utils.cot_analysis import split_tokens_into_sentences
from utils.utils import clear_cuda, set_seed


def generate_branches_vllm(
    model_name: str,
    prefix_token_ids: List[int],
    num_branches: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[List[int]]:
    """Generate branch continuations using vLLM.

    Args:
        model_name: HuggingFace model name.
        prefix_token_ids: Shared prefix to continue from.
        num_branches: Number of branches to generate.
        max_new_tokens: Max tokens per branch.
        temperature: Sampling temperature.
        seed: Random seed.

    Returns:
        List of token id lists (continuation only, no prefix).
    """
    llm = LLM(model=model_name, dtype="bfloat16", seed=seed)
    sampling_params = SamplingParams(
        n=num_branches,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(
        [{"prompt_token_ids": prefix_token_ids}],
        sampling_params,
    )

    branches = []
    for completion in outputs[0].outputs:
        branches.append(list(completion.token_ids))

    # Free vLLM
    del llm
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clear_cuda()

    return branches


DEFAULT_PROMPT = (
    "The capital of France is Paris. "
    "Answer in 100 words or less, what are the most popular things "
    "in the city to do?"
)


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    num_new_branches: int = 8,
    masking_algorithm: str = "nodewise_attribution",
    analysis_timestep: int = -1,
    objective: str = "kl_divergence",
    layers_to_analyse: str = "8,12,16,20,24",
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    seed: int = 42,
    temp: float = 0.6,
    max_new_tokens: int = 150,
    output_dir: str = "results/circuit_discovery",
    min_sentence_length: int = 10,
    batch_size: int = 4,
    prompt: str = "",
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x.strip()) for x in layers_to_analyse.split(",")]

    # --- 1. Tokenize prompt ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not prompt:
        prompt = DEFAULT_PROMPT
    chat = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    prefix_token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
    prefix_len = len(prefix_token_ids)

    if analysis_timestep == -1:
        analysis_timestep = prefix_len

    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Prefix length: {prefix_len} tokens")
    print(f"Analysis timestep: {analysis_timestep}")

    # --- 2. Split prefix into sentences ---
    prefix_tensor = torch.tensor(prefix_token_ids)
    sentences = split_tokens_into_sentences(
        prefix_tensor, tokenizer, min_sentence_length=min_sentence_length
    )
    # Clip sentences to analysis timestep
    sentences = [s for s in sentences if s.start < analysis_timestep]
    if sentences and sentences[-1].end >= analysis_timestep:
        from utils.utils import Sentence
        last = sentences[-1]
        sentences[-1] = Sentence(start=last.start, end=analysis_timestep - 1)

    print(f"Found {len(sentences)} sentences in prefix")
    for i, s in enumerate(sentences):
        text = tokenizer.decode(prefix_token_ids[s.start : s.end + 1])
        print(f"  S{i}: tokens {s.start}-{s.end}: {repr(text[:80])}")

    # --- 3. Generate branches via vLLM ---
    print(f"\nGenerating {num_new_branches} branches via vLLM (temp={temp})...")
    branch_token_ids = generate_branches_vllm(
        model_name=model_name,
        prefix_token_ids=prefix_token_ids,
        num_branches=num_new_branches,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        seed=seed,
    )
    print(f"Generated {len(branch_token_ids)} branches")
    for i, branch in enumerate(branch_token_ids[:3]):
        text = tokenizer.decode(branch[:50])
        print(f"  Branch {i}: {repr(text[:100])}...")

    # --- 4. Load HF model with eager attention ---
    print(f"\nLoading HF model for attribution: {model_name}")
    model, tokenizer = load_custom_model_eager(model_name, device=device)

    # --- 5. Run circuit discovery ---
    print(f"\nRunning {masking_algorithm} (layers={layers}, gap={sentence_gap}, chunk={sentence_chunk})...")
    algorithm = get_algorithm(masking_algorithm)

    node_mask = algorithm.discover(
        model=model,
        tokenizer=tokenizer,
        prefix_token_ids=prefix_token_ids,
        branch_token_ids=branch_token_ids,
        sentences=sentences,
        layers=layers,
        analysis_timestep=analysis_timestep,
        sentence_gap=sentence_gap,
        sentence_chunk=sentence_chunk,
        objective_name=objective,
        batch_size=batch_size,
    )

    # Add extra metadata
    node_mask.metadata.update({
        "model_name": model_name,
        "prompt": prompt,
        "prefix_length": prefix_len,
        "temperature": temp,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "min_sentence_length": min_sentence_length,
    })

    # --- 6. Save ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"node_mask_{masking_algorithm}_gap{sentence_gap}_chunk{sentence_chunk}.json",
    )
    node_mask.to_json(output_path)
    print(f"\nNodeMask saved to: {output_path}")

    # Print summary
    for layer_idx, scores in sorted(node_mask.scores.items()):
        print(
            f"  Layer {layer_idx}: "
            f"max={scores.max():.6f}, mean={scores.mean():.6f}, "
            f"shape={list(scores.shape)}"
        )

    # Cleanup
    del model
    del tokenizer
    clear_cuda()

    return node_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run circuit discovery (nodewise attribution patching)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument(
        "--masking_algorithm",
        type=str,
        default="nodewise_attribution",
        choices=["nodewise_attribution"],
    )
    parser.add_argument(
        "--analysis_timestep",
        type=int,
        default=-1,
        help="Token position for analysis (-1 = prompt length)",
    )
    parser.add_argument("--objective", type=str, default="kl_divergence")
    parser.add_argument("--layers_to_analyse", type=str, default="8,12,16,20,24")
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument(
        "--output_dir", type=str, default="results/circuit_discovery"
    )
    parser.add_argument("--min_sentence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Custom prompt (default: Paris example)",
    )
    args = parser.parse_args()
    main(**vars(args))
