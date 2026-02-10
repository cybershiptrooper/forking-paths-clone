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
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import Sentence, set_seed, clear_cuda
from utils.cot_analysis import split_tokens_into_sentences
from utils.objectives import get_objective
from utils.masks import NodeMask
from utils.circuit_discovery.factory import create_circuit_discovery


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


def chunk_sentences(sentences: list[Sentence], chunk_size: int) -> list[Sentence]:
    """Merge consecutive sentences into chunks of chunk_size.

    Args:
        sentences: List of Sentence(start, end) namedtuples
        chunk_size: Number of sentences per chunk

    Returns:
        List of Sentence chunks with merged token ranges
    """
    if chunk_size <= 1:
        return sentences
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        chunks.append(Sentence(start=group[0].start, end=group[-1].end))
    return chunks


def remove_bos_from_sentences(sentences: list[Sentence]) -> list[Sentence]:
    """If any sentence starts at index 0, clamp to index 1 to skip BOS token."""
    result = []
    for s in sentences:
        if s.start == 0:
            result.append(Sentence(start=2, end=s.end))
        else:
            result.append(s)
    return result


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


def evaluate_at_thresholds(
    model,
    node_mask: NodeMask,
    input_ids: torch.Tensor,
    sentences: list[Sentence],
    continuations: list[torch.Tensor],
    objective_fn,
    thresholds: list[float],
    layers: list[int],
    ablate_non_target_layers: bool = False,
    tokenizer=None,
) -> list[dict]:
    """Evaluate KL divergence and sparsity at different mask thresholds.

    For each threshold:
    - Compute sparsity from the mask
    - Re-run model with thresholded binary mask
    - Compute KL divergence with clean output
    """
    from utils.circuit_discovery.nodewise_attribution import (
        llama_attention_forward_with_differentiable_mask,
        expand_sentence_mask_to_tokens,
    )
    from utils.circuit_discovery.base import CircuitDiscovery, AblationHandle
    from utils.utils import get_attention_module

    device = next(model.parameters()).device
    num_heads = model.config.num_attention_heads
    prefix_len = input_ids.shape[-1]
    num_sents = len(sentences)
    max_cont_len = max(c.shape[-1] for c in continuations)
    total_seq_len = prefix_len + max_cont_len

    # Build token_to_sent
    token_to_sent = torch.full((total_seq_len,), -1, dtype=torch.long)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx
    token_to_sent = token_to_sent.to(device)

    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)  # no gap for eval

    # Optionally ablate non-target layers
    non_target_handles = []
    if ablate_non_target_layers:
        import types

        num_total_layers = model.config.num_hidden_layers
        target_set = set(layers)
        non_target = [l for l in range(num_total_layers) if l not in target_set]
        print(f"Ablating {len(non_target)} non-target layers for evaluation...")
        for layer_idx in non_target:
            attn_module = get_attention_module(model, layer_idx)
            original_forward = attn_module.forward
            attn_module._circuit_mask = torch.zeros(
                num_heads, num_sents, num_sents, device=device
            )
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module.forward = types.MethodType(
                llama_attention_forward_with_differentiable_mask, attn_module
            )
            non_target_handles.append(AblationHandle(attn_module, original_forward))

    # Compute clean logits
    print("Computing clean logits for threshold evaluation...")
    clean_logits_list = []
    model.eval()
    with torch.no_grad():
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            clean_logits_list.append(model(full_input).logits.cpu())

    results = []
    for threshold in thresholds:
        sparsity = node_mask.sparsity(threshold)

        # Build binary masks: 1 if |score| >= threshold, 0 otherwise
        binary_masks = {}
        for l in layers:
            m = torch.ones(num_heads, num_sents, num_sents, device=device)
            for h in range(num_heads):
                scores = node_mask.scores[l][h]
                for i in range(num_sents):
                    for j in range(num_sents):
                        if abs(scores[i][j]) < threshold:
                            m[h, i, j] = 0.0
            binary_masks[l] = m

        # Patch model
        import types

        handles = []
        for layer_idx in layers:
            attn_module = get_attention_module(model, layer_idx)
            original_forward = attn_module.forward
            attn_module._circuit_mask = binary_masks[layer_idx]
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module.forward = types.MethodType(
                llama_attention_forward_with_differentiable_mask, attn_module
            )
            handles.append(AblationHandle(attn_module, original_forward))

        # Compute masked logits
        total_kl = 0.0
        total_tokens = 0
        per_token_kl_branches = []
        per_sent_kl_branches = []
        with torch.no_grad():
            for cont_idx, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                pos_mask = torch.zeros(1, full_len, device=device)
                pos_mask[0, prefix_len - 1 : full_len - 1] = 1.0

                logits = model(full_input).logits
                clean = clean_logits_list[cont_idx][:, :full_len].to(device)
                kl = objective_fn(clean, logits, pos_mask)
                total_kl += kl.item() * pos_mask.sum().item()
                total_tokens += pos_mask.sum().item()

                # Per-token KL for this branch (continuation tokens only)
                log_clean = torch.nn.functional.log_softmax(clean.detach(), dim=-1)
                log_masked = torch.nn.functional.log_softmax(logits, dim=-1)
                kl_tokens = torch.nn.functional.kl_div(
                    log_masked, log_clean, log_target=True, reduction="none"
                ).sum(dim=-1)  # (1, seq_len)
                branch_kl = kl_tokens[0, prefix_len - 1 : full_len - 1].cpu().tolist()
                per_token_kl_branches.append(branch_kl)

                # Per-sentence KL for this branch
                if tokenizer is not None:
                    cont_token_ids = cont[0]  # (num_cont_tokens,)
                    cont_sents = split_tokens_into_sentences(
                        cont_token_ids, tokenizer, min_sentence_length=5
                    )
                    sent_kl_list = []
                    for s in cont_sents:
                        # s.start/end are relative to cont; branch_kl is indexed the same way
                        s_kl = branch_kl[s.start : s.end + 1]
                        avg = sum(s_kl) / max(len(s_kl), 1)
                        text = tokenizer.decode(cont_token_ids[s.start : s.end + 1].tolist())
                        sent_kl_list.append({"text": text, "mean_kl": avg})
                    per_sent_kl_branches.append(sent_kl_list)

        # Cleanup
        for h in handles:
            h.remove()

        avg_kl = total_kl / max(total_tokens, 1)
        entry = {
            "threshold": threshold,
            "sparsity": sparsity,
            "kl_divergence": avg_kl,
            "per_token_kl": per_token_kl_branches,
        }
        if per_sent_kl_branches:
            entry["per_sentence_kl"] = per_sent_kl_branches
        results.append(entry)
        print(
            f"  threshold={threshold:.3f} | sparsity={sparsity:.2%} | KL={avg_kl:.6f}"
        )

    # Cleanup non-target layer ablation
    for h in non_target_handles:
        h.remove()

    return results


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
    num_ig_steps: int = 10,
    no_negate_scores: bool = False,
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/circuit_discovery",
    thresholds: list[float] = None,
):
    if layers_to_analyse is None:
        layers_to_analyse = [8, 12, 16, 20, 24]
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
            max_tokens=10000,
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
        token_ids_for_splitting, tokenizer, min_sentence_length=10
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

    output_file = os.path.join(
        output_dir,
        f"circuit_{masking_algorithm}_layers{'_'.join(map(str, layers_to_analyse))}"
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
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24],
    )
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument(
        "--ablate_non_target_layers",
        action="store_true",
        help="Ablate all attention heads in layers outside --layers_to_analyse",
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
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="Thresholds for sparsity-vs-KL evaluation",
    )
    args = parser.parse_args()
    main(**vars(args))
