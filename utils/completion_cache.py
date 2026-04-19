"""Cache for vLLM completions (base completion + branches).

Given the same (model_name, formatted_text, analysis_timestep, temperature,
seed, max_sampling_tokens, num_branches), vLLM produces identical outputs.
This module caches those outputs to avoid redundant generation.
"""

import hashlib
import json
import os


DEFAULT_CACHE_DIR = "cache/completions"


def compute_cache_key(
    model_name: str,
    formatted_text: str,
    analysis_timestep: int,
    temperature: float,
    seed: int,
    max_sampling_tokens: int,
    num_branches: int,
) -> str:
    """Compute a deterministic 16-char hex hash from generation parameters."""
    key_data = json.dumps(
        {
            "model_name": model_name,
            "formatted_text": formatted_text,
            "analysis_timestep": analysis_timestep,
            "temperature": temperature,
            "seed": seed,
            "max_sampling_tokens": max_sampling_tokens,
            "num_branches": num_branches,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def load_from_cache(cache_key: str, cache_dir: str = DEFAULT_CACHE_DIR):
    """Load cached completions by key. Returns None on miss."""
    path = os.path.join(cache_dir, f"{cache_key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_to_cache(cache_key: str, data: dict, cache_dir: str):
    """Save completions to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved to cache: {path}")


def get_or_generate(
    model_name: str,
    formatted_text: str,
    prompt_len: int,
    analysis_timestep: int,
    num_branches: int,
    temperature: float,
    max_sampling_tokens: int,
    seed: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """Get cached completions or generate with vLLM.

    Returns dict with keys:
        input_ids: list[int] — full prefix token IDs (prompt + base completion)
        branches: list[dict] — each with 'text' and 'token_ids'
        cache_key: str — 16-char hex hash
    """
    cache_key = compute_cache_key(
        model_name, formatted_text, analysis_timestep,
        temperature, seed, max_sampling_tokens, num_branches,
    )

    cached = load_from_cache(cache_key, cache_dir)
    if cached is not None:
        print(f"  Cache hit ({cache_key}): loaded {len(cached['branches'])} branches")
        cached["cache_key"] = cache_key
        return cached

    print(f"  Cache miss ({cache_key}): generating with vLLM...")

    # Import heavy dependencies only on cache miss
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from utils.utils import clear_cuda

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(formatted_text, return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, prompt_len]

    llm = LLM(model=model_name, dtype="auto")

    # Generate base completion if analysis_timestep extends beyond prompt
    if analysis_timestep > prompt_len:
        needed = analysis_timestep - prompt_len
        print(f"  Generating base completion ({needed} tokens needed)...")
        base_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_sampling_tokens,
            seed=seed,
        )
        base_outputs = llm.generate([formatted_text], base_params)
        base_output = base_outputs[0].outputs[0]
        base_token_ids = list(base_output.token_ids)[:needed]
        base_ids_tensor = torch.tensor([base_token_ids], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, base_ids_tensor], dim=-1)
        print(
            f"  Extended input_ids to {input_ids.shape[-1]} tokens "
            f"(prompt={prompt_len} + base_completion={len(base_token_ids)})."
        )
        if input_ids.shape[-1] < analysis_timestep:
            print(
                f"  Warning: base completion shorter than expected "
                f"({input_ids.shape[-1]} < {analysis_timestep})."
            )

    # Generate branches from prefix up to analysis_timestep
    effective_timestep = min(analysis_timestep, input_ids.shape[-1])
    prefix_text = tokenizer.decode(input_ids[0, :effective_timestep])
    print(f"  Generating {num_branches} branches...")
    branch_params = SamplingParams(
        n=num_branches,
        temperature=temperature,
        max_tokens=max_sampling_tokens,
        seed=seed,
    )
    branch_outputs = llm.generate([prefix_text], branch_params)

    branches = []
    for output in branch_outputs[0].outputs:
        branches.append(
            {
                "text": output.text,
                "token_ids": list(output.token_ids),
            }
        )

    # Cleanup vLLM
    del llm
    clear_cuda()
    print(f"  Generated {len(branches)} branches, vLLM cleaned up.")

    # Save to cache
    data = {
        "model_name": model_name,
        "seed": seed,
        "temperature": temperature,
        "max_sampling_tokens": max_sampling_tokens,
        "num_branches": num_branches,
        "analysis_timestep": analysis_timestep,
        "prompt_len": prompt_len,
        "input_ids": input_ids[0].tolist(),
        "branches": branches,
    }
    _save_to_cache(cache_key, data, cache_dir)

    data["cache_key"] = cache_key
    return data
