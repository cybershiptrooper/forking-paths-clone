"""Learn a circuit mask over sentence-to-sentence attention patterns.

Pipeline:
1. Tokenize input, split into sentences (clipped at analysis timestep)
2. Generate new branches from the analysis timestep using vLLM
3. Run circuit discovery (integrated gradients) to learn per-head masks
4. Evaluate sparsity-vs-KL at multiple thresholds
5. Save the learned NodeMask to JSON
"""

import os
import argparse
import json

import torch
from utils.cot_analysis import chunk_sentences
from utils.cot_analysis import remove_bos_from_sentences
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda
from utils.cot_analysis import split_tokens_into_sentences
from utils.objectives import get_objective
from utils.masks import NodeMask
from utils.circuit_discovery.factory import create_circuit_discovery
from utils.circuit_eval import evaluate_at_thresholds


def load_model_eager(model_name: str, device: str = "cuda"):
    """Load model with eager attention for circuit discovery (needs attention weights)."""
    print(f"Loading {model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_branches(
    model_name: str,
    prefix_text: str,
    num_branches: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    """Generate multiple continuations from prefix using vLLM.

    Returns list of dicts with 'text' and 'token_ids' for each branch.
    """
    print(f"Loading vLLM model for generation ({num_branches} branches)...")
    llm = LLM(model=model_name, dtype="auto")
    sampling_params = SamplingParams(
        n=num_branches,
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate([prefix_text], sampling_params)

    branches = []
    for output in outputs[0].outputs:
        branches.append(
            {
                "text": output.text,
                "token_ids": list(output.token_ids),
            }
        )

    # Cleanup vLLM
    del llm
    clear_cuda()
    print(f"Generated {len(branches)} branches, vLLM cleaned up.")
    return branches


def _parse_layers_arg(value: str):
    lowered = value.lower()
    if lowered == "all":
        return "all"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid layer value: {value!r}. Use integers or 'all'."
        ) from exc


def _resolve_layers_to_analyse(layers_to_analyse, model):
    if layers_to_analyse is None:
        return [8, 12, 16, 20, 24]
    if isinstance(layers_to_analyse, str):
        if layers_to_analyse.lower() == "all":
            return list(range(model.config.num_hidden_layers))
        return [int(layers_to_analyse)]
    if len(layers_to_analyse) == 1 and isinstance(layers_to_analyse[0], str):
        if layers_to_analyse[0].lower() == "all":
            return list(range(model.config.num_hidden_layers))
    if any(isinstance(l, str) and l.lower() == "all" for l in layers_to_analyse):
        raise ValueError("Use 'all' by itself (no other layer indices).")
    return [int(l) for l in layers_to_analyse]


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    num_new_branches: int = 8,
    masking_algorithm: str = "nodewise_attribution",
    analysis_timestep: int = None,
    objective: str = "kl_divergence",
    layers_to_analyse: list[int] = None,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    ablate_non_target_layers: bool = False,
    renormalize_masked_attention: bool = True,
    num_ig_steps: int = 10,
    no_negate_scores: bool = False,
    max_new_tokens: int = 150,
    min_sentence_length: int = 10,
    temperature: float = 0.6,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/circuit_discovery",
    thresholds: list[float] = None,
):
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # Step 1: Prepare input (example from controlled_ablations_v2.py)
    # =====================================================================
    print("=" * 80)
    print("Step 1: Preparing input...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prompt = (
    #     "The capital of France is Paris. Answer in 100 words or less, "
    #     "what are the most popular things in the city to do?"
    # )
    prompt = "A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?"
    chat = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    if analysis_timestep is None:
        analysis_timestep = prompt_len + 200

    print(f"Prompt length: {prompt_len} tokens")
    print(f"Analysis timestep: {analysis_timestep}")
    print(f"Formatted text:\n{formatted_text}\n")

    # =====================================================================
    # Step 2: Generate base completion (if needed) and branches with vLLM
    # =====================================================================
    print("=" * 80)
    print("Step 2: Generating with vLLM...")
    print("=" * 80)

    llm = LLM(model=model_name, dtype="auto")

    # Generate base completion if analysis_timestep extends beyond prompt
    if analysis_timestep > prompt_len:
        needed = analysis_timestep - prompt_len
        print(f"Generating base completion ({needed} tokens needed)...")
        base_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        base_outputs = llm.generate([formatted_text], base_params)
        base_output = base_outputs[0].outputs[0]
        base_token_ids = list(base_output.token_ids)[:needed]
        base_ids_tensor = torch.tensor([base_token_ids], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, base_ids_tensor], dim=-1)
        print(
            f"Extended input_ids to {input_ids.shape[-1]} tokens "
            f"(prompt={prompt_len} + base_completion={len(base_token_ids)})."
        )
        if input_ids.shape[-1] < analysis_timestep:
            print(
                f"Warning: base completion shorter than expected "
                f"({input_ids.shape[-1]} < {analysis_timestep}). "
                f"Adjusting analysis_timestep."
            )
            analysis_timestep = input_ids.shape[-1]

    # Generate branches from prefix up to analysis_timestep
    prefix_text = tokenizer.decode(input_ids[0, :analysis_timestep])
    print(f"Generating {num_new_branches} branches...")
    branch_params = SamplingParams(
        n=num_new_branches,
        temperature=temperature,
        max_tokens=max_new_tokens,
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
    print(f"Generated {len(branches)} branches, vLLM cleaned up.")

    for i, b in enumerate(branches):
        print(f"  Branch {i}: {len(b['token_ids'])} tokens — {repr(b['text'][:80])}...")

    # =====================================================================
    # Step 3: Split into sentences
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 3: Splitting into sentences...")
    print("=" * 80)

    token_ids_for_splitting = input_ids[0, :analysis_timestep]
    sentences = split_tokens_into_sentences(
        token_ids_for_splitting, tokenizer, min_sentence_length=min_sentence_length
    )
    sentences = remove_bos_from_sentences(sentences)
    sentences = chunk_sentences(sentences, sentence_chunk)

    print(f"Found {len(sentences)} sentence chunks:")
    for i, s in enumerate(sentences):
        text = tokenizer.decode(input_ids[0, s.start : s.end + 1])
        print(f"  S{i}: [{s.start}:{s.end}] = {repr(text)}")

    # =====================================================================
    # Step 4: Load HuggingFace model (eager attention)
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 4: Loading model with eager attention...")
    print("=" * 80)

    model, tokenizer = load_model_eager(model_name, device=device)
    input_ids = input_ids.to(device)
    layers_to_analyse_is_all = (
        isinstance(layers_to_analyse, list)
        and len(layers_to_analyse) == 1
        and isinstance(layers_to_analyse[0], str)
        and layers_to_analyse[0].lower() == "all"
    )
    layers_to_analyse = _resolve_layers_to_analyse(layers_to_analyse, model)

    # Convert branches to tensors
    continuations = []
    for b in branches:
        cont_ids = torch.tensor([b["token_ids"]], device=device)
        continuations.append(cont_ids)

    # =====================================================================
    # Step 5: Circuit discovery
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Step 5: Running {masking_algorithm}...")
    print("=" * 80)

    objective_fn = get_objective(objective)
    discoverer = create_circuit_discovery(
        masking_algorithm,
        model=model,
        tokenizer=tokenizer,
        layers=layers_to_analyse,
        objective_fn=objective_fn,
        sentence_gap=sentence_gap,
        ablate_non_target_layers=ablate_non_target_layers,
        renormalize_masked_attention=renormalize_masked_attention,
        num_ig_steps=num_ig_steps,
        negate_scores=not no_negate_scores,
    )

    node_mask = discoverer.discover(
        input_ids=input_ids,
        sentences=sentences,
        continuations=continuations,
    )

    # Add sentence text to metadata
    for i, s in enumerate(node_mask.sentences):
        s["text"] = tokenizer.decode(input_ids[0, s["start"] : s["end"] + 1])

    # =====================================================================
    # Step 6: Evaluate at thresholds
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating sparsity vs KL at thresholds...")
    print("=" * 80)

    threshold_results = evaluate_at_thresholds(
        model=model,
        node_mask=node_mask,
        input_ids=input_ids,
        sentences=sentences,
        continuations=continuations,
        objective_fn=objective_fn,
        thresholds=thresholds,
        layers=layers_to_analyse,
        ablate_non_target_layers=ablate_non_target_layers,
        renormalize_masked_attention=renormalize_masked_attention,
        tokenizer=tokenizer,
    )

    node_mask.metadata["threshold_evaluation"] = threshold_results
    node_mask.metadata["seed"] = seed
    node_mask.metadata["temperature"] = temperature
    node_mask.metadata["max_new_tokens"] = max_new_tokens
    node_mask.metadata["num_branches"] = num_new_branches

    # =====================================================================
    # Step 7: Save results
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 7: Saving results...")
    print("=" * 80)
    layers_str = (
        "_all"
        if layers_to_analyse_is_all
        else "_".join(str(l) for l in layers_to_analyse)
    )
    output_file = os.path.join(
        output_dir,
        f"circuit_{masking_algorithm}_layers{layers_str}"
        f"_branches{num_new_branches}_ig{num_ig_steps}.json",
    )
    node_mask.to_json(output_file)
    print(f"Saved NodeMask to {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Layers: {layers_to_analyse}")
    print(f"  Heads per layer: {node_mask.metadata.get('num_heads', '?')}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Algorithm: {masking_algorithm}")
    print(f"  IG steps: {num_ig_steps}")
    print(f"  Branches: {num_new_branches}")
    print("\nThreshold evaluation:")
    for r in threshold_results:
        print(
            f"  t={r['threshold']:.1e} → sparsity={r['sparsity']:.2%}, KL={r['kl_divergence']:.2e}"
        )

    # Cleanup
    del model
    clear_cuda()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn a circuit mask over sentence-to-sentence attention patterns"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument(
        "--masking_algorithm",
        choices=["nodewise_attribution", "EAP", "subnetwork_probing"],
        default="nodewise_attribution",
    )
    parser.add_argument(
        "--analysis_timestep",
        type=int,
        default=None,
        help="Token index for analysis (default: prompt length)",
    )
    parser.add_argument("--objective", default="kl_divergence")
    parser.add_argument(
        "--layers_to_analyse",
        type=_parse_layers_arg,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Layer indices to analyze, or 'all' for every layer.",
    )
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument(
        "--ablate_non_target_layers",
        action="store_true",
        help="Ablate all attention heads in layers outside --layers_to_analyse",
    )
    parser.add_argument(
        "--no_renormalize_masked_attention",
        dest="renormalize_masked_attention",
        action="store_false",
        help="Do not renormalize post-softmax attention after applying the mask.",
    )
    parser.add_argument("--num_ig_steps", type=int, default=10)
    parser.add_argument(
        "--no_negate_scores",
        action="store_true",
        help="Store raw IG scores (positive = increases KL). "
        "Default negates so positive = helps retention.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/circuit_discovery")
    parser.add_argument(
        "--min_sentence_length",
        type=int,
        default=10,
        help="Minimum number of tokens for a sentence (after splitting).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            -5e-3,
            -1e-3,
            -5e-4,
            -1e-4,
            -5e-5,
            -1e-5,
            -5e-6,
            -1e-6,
            -5e-7,
            -5e-8,
            -1e-8,
            -1e-9,
            -1e-10,
            -1e-11,
            -1e-12,
            0.0,
            1e-12,
            1e-11,
            1e-10,
            1e-9,
            1e-8,
            5e-8,
            1e-7,
            5e-7,
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
        ],
        help="Thresholds for sparsity-vs-KL evaluation",
    )
    args = parser.parse_args()
    main(**vars(args))
