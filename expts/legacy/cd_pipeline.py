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

from utils.utils import set_seed, clear_cuda, Sentence
from utils.cot_analysis import split_tokens_into_sentences
from utils.objectives import get_objective
from utils.importance_sampling import extract_answer_ids, reward_based_answer_ids
from utils.masks import NodeMask
from utils.circuit_discovery.factory import create_circuit_discovery, get_available_algorithms
from utils.circuit_eval import evaluate_at_thresholds
from utils.rewards import (
    extract_boxed,
    compute_correctness_rewards,
    compute_cot_length_rewards,
    find_answer_token_positions,
)


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
    max_sampling_tokens: int,
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
        max_tokens=max_sampling_tokens,
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
    model_to_analyse: str = None,
    prompt: str = None,
    num_new_branches: int = 8,
    masking_algorithm: str = "nodewise_attribution",
    pair_aggregation: str = "mean",
    mask_granularity: str = "head",
    analysis_timestep: int = None,
    objective: str = "kl_divergence",
    layers_to_analyse: list[int] = None,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    mask_mode: str = "prefix",
    ablate_non_target_layers: bool = False,
    renormalize_masked_attention: bool = True,
    num_ig_steps: int = 10,
    no_negate_scores: bool = False,
    num_random_samples: int = 5,
    max_sampling_tokens: int = 150,
    num_tokens_to_analyse: int = None,
    min_sentence_length: int = 10,
    temperature: float = 0.6,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/circuit_discovery",
    sparsities: list[float] = None,
    reward_type: str = "none",
    correct_answer: str = None,
    data_path: str = None,
    prompt_index: int = None,
    dataset_type: str = "open ended",
    answer_only: bool = False,
    judge_model: str = "meta-llama/llama-3.2-3b-instruct",
    judge_answers: bool = False,
    file_name: str = None,
):
    # Default model_to_analyse to model_name
    if model_to_analyse is None:
        model_to_analyse = model_name
    if sparsities is None:
        sparsities = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    # Default num_tokens_to_analyse to max_sampling_tokens
    if num_tokens_to_analyse is None:
        num_tokens_to_analyse = max_sampling_tokens

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # Step 0: Load data from collection JSON if provided
    # =====================================================================
    if data_path is not None and prompt_index is not None:
        print(f"Loading record {prompt_index} from {data_path}...")
        with open(data_path) as f:
            records = json.load(f)
        record = records[prompt_index]
        prompt = record["question"]
        if correct_answer is None and "correct_answer" in record:
            correct_answer = extract_boxed(record["correct_answer"]) or record["correct_answer"]
        if "dataset_type" in record:
            dataset_type = record["dataset_type"]
        print(f"  Question: {prompt[:120]}...")
        print(f"  Correct answer: {correct_answer}")

    branch_rewards = None
    position_mask_overrides = None

    # =====================================================================
    # Step 1: Prepare input (example from controlled_ablations_v2.py)
    # =====================================================================
    print("=" * 80)
    print("Step 1: Preparing input...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if prompt is None:
        prompt = (
            "A rectangular band formation is a formation with $m$ band members "
            "in each of $r$ rows, where $m$ and $r$ are integers. A particular "
            "band has less than 100 band members. The director arranges them in "
            "a rectangular formation and finds that he has two members left over. "
            "If he increases the number of members in each row by 1 and reduces "
            "the number of rows by 2, there are exactly enough places in the new "
            "formation for each band member. What is the largest number of "
            "members the band could have?"
        )
    chat = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    if analysis_timestep is None:
        analysis_timestep = prompt_len + 200
    else:
        analysis_timestep = prompt_len + analysis_timestep

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
            max_tokens=max_sampling_tokens,
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

    # Compute correctness rewards while vLLM is still available (uses OpenRouter)
    if reward_type == "correctness":
        if correct_answer is None:
            raise ValueError(
                "--reward_type correctness requires --correct_answer or "
                "--data_path + --prompt_index with a correct_answer field."
            )
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
        branch_rewards = compute_correctness_rewards(
            branches, correct_answer, prompt, prefix_text, client, judge_model
        )
        del client
        print(f"Correctness rewards: {branch_rewards}")

    # Cleanup vLLM
    del llm
    clear_cuda()
    print(f"Generated {len(branches)} branches, vLLM cleaned up.")

    for i, b in enumerate(branches):
        reward_str = f" reward={branch_rewards[i]:+.1f}" if branch_rewards is not None else ""
        print(f"  Branch {i}: {len(b['token_ids'])} tokens{reward_str} — {repr(b['text'][:80])}...")

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
    num_prefix_sentences = len(sentences)

    # Optionally add generation sentences (for "generation" / "both" modes)
    gen_sentences_raw = []
    first_branch_tokens = None
    if mask_mode in ("generation", "both"):
        first_branch_tokens = torch.tensor(branches[0]["token_ids"])
        gen_sentences_raw = split_tokens_into_sentences(
            first_branch_tokens, tokenizer, min_sentence_length=min_sentence_length
        )
        gen_sentences_raw = chunk_sentences(gen_sentences_raw, sentence_chunk)
        # Offset to absolute positions in the full sequence
        gen_sentences = [
            Sentence(start=s.start + analysis_timestep, end=s.end + analysis_timestep)
            for s in gen_sentences_raw
        ]
        sentences = list(sentences) + gen_sentences
        print(f"Mask mode '{mask_mode}': added {len(gen_sentences)} generation sentences")

    print(f"Found {len(sentences)} sentence chunks ({num_prefix_sentences} prefix):")
    for i, s in enumerate(sentences):
        label = "P" if i < num_prefix_sentences else "G"
        if i < num_prefix_sentences:
            text = tokenizer.decode(input_ids[0, s.start : s.end + 1])
        else:
            # Decode from first branch tokens using the raw (pre-offset) boundaries
            raw = gen_sentences_raw[i - num_prefix_sentences]
            text = tokenizer.decode(first_branch_tokens[raw.start : raw.end + 1].tolist())
        print(f"  {label}{i}: [{s.start}:{s.end}] = {repr(text)}")

    # Compute CoT length rewards (uses sentence splitting on branches)
    if reward_type == "cot_length":
        branch_rewards = compute_cot_length_rewards(
            branches, tokenizer, min_sentence_length=min_sentence_length
        )
        print(f"CoT length rewards: {branch_rewards}")

    # Extract answer IDs (always, for all metrics including IS and contrastive)
    answer_ids_tensor = None
    num_answers = None
    answer_labels = None
    if branch_rewards is not None:
        # Option C: reward-based bucketing
        # correctness (+1/-1) → binary; cot_length (continuous) → per-value
        use_binary = reward_type == "correctness"
        answer_ids_list, answer_labels = reward_based_answer_ids(
            branch_rewards, binary=use_binary,
        )
        grouping_method = "reward_binary" if use_binary else "reward_unique"
    elif judge_answers:
        # Option B: LLM judge clustering (+ Option A normalization)
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError(
                "--judge_answers requires OPENROUTER_API_KEY in environment."
            )
        judge_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=openrouter_key,
        )
        answer_ids_list, answer_labels = extract_answer_ids(
            branches, prefix_text,
            judge_client=judge_client, judge_model=judge_model, question=prompt,
        )
        del judge_client
        grouping_method = "judge"
    else:
        # Option A: normalized \\boxed{} extraction (default)
        answer_ids_list, answer_labels = extract_answer_ids(branches, prefix_text)
        grouping_method = "boxed_normalized"

        # Auto-fallback to judge if >50% of branches lack a boxed answer
        no_answer_count = sum(
            1 for label in answer_labels if label.startswith("__no_answer_")
        )
        no_answer_branches = sum(
            1 for a in answer_ids_list
            if answer_labels[a].startswith("__no_answer_")
        )
        if no_answer_branches > len(branches) * 0.5:
            print(
                f"  {no_answer_branches}/{len(branches)} branches lack "
                f"\\boxed{{}} answers. Falling back to judge model..."
            )
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError(
                    "More than 50% of branches lack \\boxed{} answers. "
                    "Judge-based answer extraction requires OPENROUTER_API_KEY "
                    "in the environment."
                )
            judge_client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=openrouter_key,
            )
            answer_ids_list, answer_labels = extract_answer_ids(
                branches, prefix_text,
                judge_client=judge_client, judge_model=judge_model,
                question=prompt,
            )
            del judge_client
            grouping_method = "judge_fallback"

    answer_ids_tensor = torch.tensor(answer_ids_list, dtype=torch.long)
    num_answers = len(answer_labels)
    print(f"\nAnswer groups ({num_answers}, method={grouping_method}):")
    for aid, label in enumerate(answer_labels):
        count = sum(1 for a in answer_ids_list if a == aid)
        print(f"  [{aid}] {label!r}: {count} branches")

    # =====================================================================
    # Step 4: Load HuggingFace model (eager attention)
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Step 4: Loading model with eager attention ({model_to_analyse})...")
    print("=" * 80)

    model, tokenizer = load_model_eager(model_to_analyse, device=device)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
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
        cont_ids = torch.tensor([b["token_ids"]], device=target_device)
        continuations.append(cont_ids)

    # Truncate continuations for discovery: optimize mask over only the first
    # num_tokens_to_analyse tokens, but keep full continuations for eval and
    # answer extraction (which need the complete branch to find \boxed{}).
    if num_tokens_to_analyse < max_sampling_tokens:
        discovery_continuations = [c[:, :num_tokens_to_analyse] for c in continuations]
        print(f"  Truncating continuations for discovery: {max_sampling_tokens} -> {num_tokens_to_analyse} tokens")
    else:
        discovery_continuations = continuations

    # Build answer-only position masks if requested
    if answer_only:
        print("Building answer-only position masks...")
        prefix_len = input_ids.shape[-1]
        position_mask_overrides = []
        for b in branches:
            pm = find_answer_token_positions(
                b["text"], b["token_ids"], tokenizer, prefix_len
            )
            if pm is not None:
                pm = pm.to(target_device)
            position_mask_overrides.append(pm)
        n_found = sum(1 for pm in position_mask_overrides if pm is not None)
        print(f"  Found answer tokens in {n_found}/{len(branches)} branches")
        if n_found == 0:
            print("  WARNING: No answer tokens found in any branch. Falling back to full mask.")
            position_mask_overrides = None

    # =====================================================================
    # Step 5: Circuit discovery
    # =====================================================================
    # Cost warning for per-head activation patching
    if "nodewise_activation_patching" in masking_algorithm and mask_granularity == "head":
        from utils.masks import (
            build_gap_filter as _bgf,
            build_mode_filter as _bmf,
            build_causal_filter as _bcf,
            build_combined_filter as _bcombf,
        )
        _num_heads = model.config.num_attention_heads
        _ns = len(sentences)
        _cf = _bcombf(
            _bgf(_ns, sentence_gap),
            _bmf(num_prefix_sentences, _ns, mask_mode),
            _bcf(_ns),
        )
        _num_active = int((~_cf).sum().item())
        _total = len(layers_to_analyse) * _num_heads * _num_active * len(continuations)
        print(
            f"\n*** WARNING: Per-head activation patching will require "
            f"{_total:,} forward passes. ***"
        )
        print(
            f"    {len(layers_to_analyse)} layers x {_num_heads} heads "
            f"x {_num_active} active pairs x {len(continuations)} branches"
        )
        _confirm = input("Continue? Type 'yes' to proceed: ").strip().lower()
        if _confirm != "yes":
            print("Aborted by user.")
            del model
            clear_cuda()
            return

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
        pair_aggregation=pair_aggregation,
        mask_granularity=mask_granularity,
    )

    node_mask = discoverer.discover(
        input_ids=input_ids,
        sentences=sentences,
        continuations=discovery_continuations,
        mask_mode=mask_mode,
        num_prefix_sentences=num_prefix_sentences,
        branch_rewards=branch_rewards,
        position_mask_overrides=position_mask_overrides,
        answer_ids=answer_ids_tensor,
        num_answers=num_answers,
    )

    # Add sentence text to metadata
    for i, s in enumerate(node_mask.sentences):
        if i < num_prefix_sentences:
            s["text"] = tokenizer.decode(input_ids[0, s["start"] : s["end"] + 1])
        else:
            raw = gen_sentences_raw[i - num_prefix_sentences]
            s["text"] = tokenizer.decode(first_branch_tokens[raw.start : raw.end + 1].tolist())

    # =====================================================================
    # Step 6: Evaluate at thresholds
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating sparsity vs KL at thresholds...")
    print("=" * 80)

    # Compute thresholds dynamically from target sparsities
    thresholds = node_mask.thresholds_for_sparsities(sparsities)
    print(f"  Target sparsities: {sparsities}")
    print(f"  Derived {len(thresholds)} thresholds from mask scores")

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
        num_random_samples=num_random_samples,
        branch_rewards=branch_rewards,
        position_mask_overrides=position_mask_overrides,
        answer_ids_fine=answer_ids_tensor,
        num_answers_fine=num_answers,
    )

    node_mask.metadata["threshold_evaluation"] = threshold_results
    node_mask.metadata["objective"] = objective
    node_mask.metadata["seed"] = seed
    node_mask.metadata["temperature"] = temperature
    node_mask.metadata["max_sampling_tokens"] = max_sampling_tokens
    node_mask.metadata["num_tokens_to_analyse"] = num_tokens_to_analyse
    node_mask.metadata["num_branches"] = num_new_branches
    node_mask.metadata["num_random_samples"] = num_random_samples
    node_mask.metadata["reward_type"] = reward_type
    if branch_rewards is not None:
        node_mask.metadata["branch_rewards"] = branch_rewards
    node_mask.metadata["answer_only"] = answer_only
    if correct_answer is not None:
        node_mask.metadata["correct_answer"] = correct_answer
    if answer_labels is not None:
        node_mask.metadata["answer_labels"] = answer_labels
        node_mask.metadata["answer_ids"] = answer_ids_tensor.tolist()
        node_mask.metadata["num_answers"] = num_answers

    # =====================================================================
    # Step 7: Save results
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 7: Saving results...")
    print("=" * 80)
    if file_name is not None:
        if not file_name.endswith(".json"):
            file_name += ".json"
        output_file = os.path.join(output_dir, file_name)
    else:
        layers_str = (
            "_all"
            if layers_to_analyse_is_all
            else "_".join(str(l) for l in layers_to_analyse)
        )
        if "activation_patching" in masking_algorithm:
            output_file = os.path.join(
                output_dir,
                f"circuit_{masking_algorithm}_layers{layers_str}"
                f"_branches{num_new_branches}.json",
            )
        else:
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
    print(f"  Mask granularity: {mask_granularity}")
    print(f"  Pair aggregation: {pair_aggregation}")
    print(f"  IG steps: {num_ig_steps}")
    print(f"  Branches: {num_new_branches}")
    print(f"  Objective: {objective}")
    print(f"  Reward type: {reward_type}")
    if branch_rewards is not None:
        print(f"  Branch rewards: {branch_rewards}")
    print(f"  Answer only: {answer_only}")
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
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model for vLLM generation (branches).",
    )
    parser.add_argument(
        "--model_to_analyse",
        type=str,
        default=None,
        help="Model loaded with eager attention for circuit discovery. "
        "Defaults to --model_name if not specified.",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument(
        "--masking_algorithm",
        choices=get_available_algorithms(),
        default="nodewise_attribution",
    )
    parser.add_argument(
        "--pair_aggregation",
        choices=["sum", "mean", "median", "max"],
        default="mean",
        help="How to aggregate token-pair AP+IG scores into sentence-pair mask scores.",
    )
    parser.add_argument(
        "--mask_granularity",
        choices=["head", "layer", "pair"],
        default="head",
        help="Score granularity: 'head' (per-head), 'layer' (shared across heads), "
        "'pair' (shared across layers and heads).",
    )
    parser.add_argument(
        "--analysis_timestep",
        type=int,
        default=None,
        help="Token index for analysis (default: prompt length)",
    )
    parser.add_argument(
        "--objective",
        choices=["kl_divergence", "log_prob", "answer_kl", "reward_gap"],
        default="kl_divergence",
        help="Local: kl_divergence, log_prob (per-token). "
        "Global: answer_kl (Obj 1, faithfulness), reward_gap (Obj 2, reward).",
    )
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
        "--mask_mode",
        choices=["prefix", "generation", "both"],
        default="prefix",
        help="Which query-key region to learn: prefix (MASK 1), "
        "generation (MASK 2), or both.",
    )
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
        "--num_random_samples",
        type=int,
        default=5,
        help="Number of random score masks (K) to sample for baseline comparison.",
    )
    parser.add_argument(
        "--no_negate_scores",
        action="store_true",
        help="Store raw IG scores (positive = increases KL). "
        "Default negates so positive = helps retention.",
    )
    parser.add_argument("--max_sampling_tokens", type=int, default=150,
        help="Max tokens for vLLM generation (base completion and branches).")
    parser.add_argument("--num_tokens_to_analyse", type=int, default=None,
        help="Number of continuation tokens to use for local KL objectives. "
        "Defaults to max_sampling_tokens. Set lower to analyse only the "
        "beginning of each branch while still sampling to completion.")
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
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        help="Target sparsity levels (0-1) for evaluation. Thresholds are "
        "computed dynamically from the learned mask scores.",
    )
    parser.add_argument(
        "--reward_type",
        choices=["none", "correctness", "cot_length"],
        default="none",
        help="Reward type for reward-weighted circuit discovery.",
    )
    parser.add_argument(
        "--correct_answer",
        type=str,
        default=None,
        help="Ground truth answer string (for correctness reward).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to collection JSON with question/answer records.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Index into collection JSON to load question + correct_answer.",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["open ended", "multiple choice", "alignment"],
        default="open ended",
        help="Dataset type for answer parsing.",
    )
    parser.add_argument(
        "--answer_only",
        action="store_true",
        help="Restrict position mask to answer tokens only (\\boxed{...}).",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/llama-3.2-3b-instruct",
        help="Model for LLM-based answer judging (OpenRouter).",
    )
    parser.add_argument(
        "--judge_answers",
        action="store_true",
        help="Use LLM judge to cluster branch answers by mathematical "
        "equivalence (for global objectives). Requires OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Custom output file name (saved under --output_dir). "
        "Auto-appends .json if missing. Overrides the default naming convention.",
    )
    # First parse to check for --config
    args, _ = parser.parse_known_args()
    if args.config:
        from utils.expt_config import load_config

        config = load_config(args.config)
        # Apply config values as new argparse defaults; CLI args will override
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
    # Re-parse with config-informed defaults
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("config", None)
    main(**kwargs)
