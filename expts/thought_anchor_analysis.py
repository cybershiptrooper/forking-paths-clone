"""Attention suppression circuit discovery (Thought Anchors, Bogdan et al. 2506.19143).

For each sentence i in the prefix, suppresses all attention to i across all
layers and heads, then measures the KL divergence at each subsequent sentence j.
The resulting (S, S) suppression scores are packaged as a pair-granularity
NodeMask that can be evaluated by evaluate_mask.py.

Pipeline:
1. Tokenize input, split into sentences (identical to learn_circuit.py)
2. Get or generate branches via completion cache
3. Extract answer IDs for evaluation metrics
4. Load model with eager attention
5. Compute suppression scores (S+1 forward passes on the prefix)
6. Save as NodeMask with cache_key for evaluation

Usage:
    uv run python -m expts.thought_anchor_analysis \\
        --config expts/configs/tests/answer_kl_patching_test.yaml

    uv run python -m expts.circuit_discovery.evaluate_mask \\
        --mask_path results/circuit_discovery/answer_kl_patching_test_suppression.json
"""

import os
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda, Sentence, get_attention_module
from utils.cot_analysis import (
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)
from utils.objectives import get_objective
from utils.importance_sampling import extract_answer_ids, reward_based_answer_ids
from utils.masks import NodeMask
from utils.circuit_eval import (
    build_token_to_sent_map,
    install_mask_hooks,
    remove_handles,
)
from utils.completion_cache import get_or_generate, DEFAULT_CACHE_DIR
from utils.rewards import (
    extract_boxed,
    compute_correctness_rewards,
    compute_cot_length_rewards,
    find_answer_token_positions,
)


def load_model_eager(model_name: str, device: str = "cuda"):
    """Load model with eager attention for suppression analysis."""
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


def compute_suppression_scores(
    model,
    input_ids,
    sentences,
    token_to_sent,
    sentence_gap,
):
    """Suppress attention to each sentence and measure KL at subsequent sentences.

    For each sentence i, zeros all attention to i across all layers/heads,
    runs a forward pass on the prefix, and measures per-sentence KL.

    Returns:
        scores: list[list[float]] of shape (S, S) where scores[src][tgt] =
            KL at sentence src when attention to tgt is suppressed.
            Higher = tgt is more important for src.
    """
    num_sents = len(sentences)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    all_layers = list(range(num_layers))
    device = next(model.parameters()).device
    seq_len = input_ids.shape[-1]

    # No gap filter for suppression — we want to freely zero any column
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool, device=device)

    # Clean forward pass
    model.eval()
    with torch.no_grad():
        clean_logits = model(input_ids).logits
    log_clean = F.log_softmax(clean_logits.float().cpu(), dim=-1)
    del clean_logits
    torch.cuda.empty_cache()

    # Install hooks once with all-ones mask, then swap per iteration
    ones_mask = torch.ones(num_heads, num_sents, num_sents, device=device)
    binary_masks = {layer: ones_mask for layer in all_layers}
    handles = install_mask_hooks(
        model, all_layers, binary_masks, token_to_sent, gap_filter, renormalize=True,
    )

    # suppression_kl[i][j] = KL at sentence j when attention to i is suppressed
    suppression_kl = [[0.0] * num_sents for _ in range(num_sents)]

    for s_suppress in range(num_sents):
        # Zero out column s_suppress (all attention TO this sentence)
        mask = torch.ones(num_heads, num_sents, num_sents, device=device)
        mask[:, :, s_suppress] = 0.0

        for layer_idx in all_layers:
            attn_module = get_attention_module(model, layer_idx)
            attn_module._circuit_mask = mask

        with torch.no_grad():
            logits = model(input_ids).logits

        log_masked = F.log_softmax(logits.float().cpu(), dim=-1)
        kl_tokens = F.kl_div(
            log_masked, log_clean, log_target=True, reduction="none",
        ).sum(dim=-1)[0]  # (seq_len,)

        for s_affected in range(num_sents):
            if s_affected < s_suppress + sentence_gap:
                continue
            start = sentences[s_affected].start
            end = min(sentences[s_affected].end, seq_len - 1)
            if start >= seq_len:
                continue
            suppression_kl[s_suppress][s_affected] = (
                kl_tokens[start : end + 1].mean().item()
            )

        max_kl = max(suppression_kl[s_suppress])
        print(f"  [{s_suppress}/{num_sents - 1}] max KL = {max_kl:.4f}")

        del logits, log_masked, kl_tokens
        torch.cuda.empty_cache()

    remove_handles(handles)

    # Transpose: scores[src][tgt] = importance of tgt for src
    scores = [[0.0] * num_sents for _ in range(num_sents)]
    for i in range(num_sents):
        for j in range(num_sents):
            scores[j][i] = suppression_kl[i][j]

    return scores


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    prompt: str = None,
    num_new_branches: int = 8,
    analysis_timestep: int = None,
    objective: str = "kl_divergence",
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    mask_mode: str = "prefix",
    renormalize_masked_attention: bool = True,
    max_sampling_tokens: int = 150,
    num_tokens_to_analyse: int = None,
    min_sentence_length: int = 10,
    temperature: float = 0.6,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/circuit_discovery",
    reward_type: str = "none",
    correct_answer: str = None,
    data_path: str = None,
    prompt_index: int = None,
    dataset_type: str = "open ended",
    answer_only: bool = False,
    judge_model: str = "meta-llama/llama-3.2-3b-instruct",
    judge_answers: bool = False,
    file_name: str = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    # Accepted but ignored (for config compatibility with learn_circuit.py)
    model_to_analyse: str = None,
    masking_algorithm: str = None,
    pair_aggregation: str = None,
    mask_granularity: str = None,
    layers_to_analyse=None,
    ablate_non_target_layers: bool = False,
    num_ig_steps: int = None,
    no_negate_scores: bool = False,
    num_random_samples: int = 5,
    sparsities: list[float] = None,
    importance_sampling_method: str = "snis",
    importance_sampling_temperature: float = None,
    **kwargs,
):
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

    # =====================================================================
    # Step 1: Prepare input
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

    # =====================================================================
    # Step 2: Get or generate branches (cached)
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 2: Getting branches (cached)...")
    print("=" * 80)

    cached = get_or_generate(
        model_name=model_name,
        formatted_text=formatted_text,
        prompt_len=prompt_len,
        analysis_timestep=analysis_timestep,
        num_branches=num_new_branches,
        temperature=temperature,
        max_sampling_tokens=max_sampling_tokens,
        seed=seed,
        cache_dir=cache_dir,
    )
    input_ids = torch.tensor([cached["input_ids"]])
    branches = cached["branches"]
    cache_key = cached["cache_key"]

    if input_ids.shape[-1] < analysis_timestep:
        print(
            f"Warning: base completion shorter than expected "
            f"({input_ids.shape[-1]} < {analysis_timestep}). "
            f"Adjusting analysis_timestep."
        )
        analysis_timestep = input_ids.shape[-1]

    prefix_text = tokenizer.decode(input_ids[0, :analysis_timestep])

    # Compute correctness rewards (uses OpenRouter, not vLLM)
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
            raw = gen_sentences_raw[i - num_prefix_sentences]
            text = tokenizer.decode(first_branch_tokens[raw.start : raw.end + 1].tolist())
        print(f"  {label}{i}: [{s.start}:{s.end}] = {repr(text)}")

    # Compute CoT length rewards
    if reward_type == "cot_length":
        branch_rewards = compute_cot_length_rewards(
            branches, tokenizer, min_sentence_length=min_sentence_length
        )
        print(f"CoT length rewards: {branch_rewards}")

    # =====================================================================
    # Step 4: Extract answer IDs
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 4: Extracting answer IDs...")
    print("=" * 80)

    answer_ids_tensor = None
    num_answers = None
    answer_labels = None
    if branch_rewards is not None:
        use_binary = reward_type == "correctness"
        answer_ids_list, answer_labels = reward_based_answer_ids(
            branch_rewards, binary=use_binary,
        )
        grouping_method = "reward_binary" if use_binary else "reward_unique"
    elif judge_answers:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("--judge_answers requires OPENROUTER_API_KEY in environment.")
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
        answer_ids_list, answer_labels = extract_answer_ids(branches, prefix_text)
        grouping_method = "boxed_normalized"

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
    # Step 5: Load model with eager attention
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Step 5: Loading model with eager attention ({model_name})...")
    print("=" * 80)

    model, tokenizer = load_model_eager(model_name, device=device)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)

    # =====================================================================
    # Step 6: Compute suppression scores
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Step 6: Computing attention suppression scores ({len(sentences)} sentences)...")
    print("=" * 80)

    token_to_sent = build_token_to_sent_map(
        sentences, input_ids.shape[-1], target_device,
    )

    scores = compute_suppression_scores(
        model=model,
        input_ids=input_ids,
        sentences=sentences,
        token_to_sent=token_to_sent,
        sentence_gap=sentence_gap,
    )

    # =====================================================================
    # Step 7: Package as NodeMask and save
    # =====================================================================
    print("\n" + "=" * 80)
    print("Step 7: Saving NodeMask...")
    print("=" * 80)

    # Build sentence dicts with text
    sentence_dicts = []
    for i, s in enumerate(sentences):
        if i < num_prefix_sentences:
            text = tokenizer.decode(input_ids[0, s.start : s.end + 1])
        else:
            raw = gen_sentences_raw[i - num_prefix_sentences]
            text = tokenizer.decode(first_branch_tokens[raw.start : raw.end + 1].tolist())
        sentence_dicts.append({"start": s.start, "end": s.end, "text": text})

    node_mask = NodeMask(
        model_name=model_name,
        algorithm="attention_suppression",
        layers=list(range(model.config.num_hidden_layers)),
        sentences=sentence_dicts,
        objective_name=objective,
        metadata={
            "mask_granularity": "pair",
            "sentence_gap": sentence_gap,
            "num_heads": model.config.num_attention_heads,
            "mask_mode": mask_mode,
            "num_prefix_sentences": num_prefix_sentences,
            "num_continuations": len(branches),
            "negate_scores": False,
            "cache_key": cache_key,
            "renormalize_masked_attention": renormalize_masked_attention,
            "objective": objective,
            "seed": seed,
            "temperature": temperature,
            "max_sampling_tokens": max_sampling_tokens,
            "num_tokens_to_analyse": num_tokens_to_analyse,
            "num_branches": num_new_branches,
            "reward_type": reward_type,
            "answer_only": answer_only,
            "importance_sampling_method": importance_sampling_method,
            "importance_sampling_temperature": importance_sampling_temperature,
        },
        scores=scores,
    )

    if branch_rewards is not None:
        node_mask.metadata["branch_rewards"] = branch_rewards
    if correct_answer is not None:
        node_mask.metadata["correct_answer"] = correct_answer
    if answer_labels is not None:
        node_mask.metadata["answer_labels"] = answer_labels
        node_mask.metadata["answer_ids"] = answer_ids_tensor.tolist()
        node_mask.metadata["num_answers"] = num_answers

    # Determine output path
    if file_name is not None:
        base = file_name.removesuffix(".json")
        output_file = os.path.join(output_dir, f"{base}_suppression.json")
    else:
        output_file = os.path.join(
            output_dir, f"attention_suppression_branches{num_new_branches}.json"
        )
    node_mask.to_json(output_file)
    print(f"Saved NodeMask to {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Algorithm: attention_suppression")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Layers: all ({model.config.num_hidden_layers})")
    print(f"  Mask granularity: pair")
    print(f"  Sentence gap: {sentence_gap}")
    print(f"  Branches: {num_new_branches}")
    print(f"  Objective: {objective}")
    print(f"  Cache key: {cache_key}")
    if branch_rewards is not None:
        print(f"  Branch rewards: {branch_rewards}")

    # Cleanup
    del model
    clear_cuda()
    print("\nDone!")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention suppression circuit discovery (Thought Anchors)"
    )
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML/JSON config file. CLI args override config values.")
    parser.add_argument("--model_name", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument("--analysis_timestep", type=int, default=None)
    parser.add_argument("--objective",
        choices=["kl_divergence", "log_prob", "answer_kl", "reward_gap"],
        default="kl_divergence")
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument("--mask_mode", choices=["prefix", "generation", "both"],
        default="prefix")
    parser.add_argument("--no_renormalize_masked_attention",
        dest="renormalize_masked_attention", action="store_false")
    parser.add_argument("--max_sampling_tokens", type=int, default=150)
    parser.add_argument("--num_tokens_to_analyse", type=int, default=None)
    parser.add_argument("--min_sentence_length", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/circuit_discovery")
    parser.add_argument("--reward_type",
        choices=["none", "correctness", "cot_length"], default="none")
    parser.add_argument("--correct_answer", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--dataset_type",
        choices=["open ended", "multiple choice", "alignment"],
        default="open ended")
    parser.add_argument("--answer_only", action="store_true")
    parser.add_argument("--judge_model", type=str,
        default="meta-llama/llama-3.2-3b-instruct")
    parser.add_argument("--judge_answers", action="store_true")
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    # Accepted for config compatibility with learn_circuit.py (ignored)
    parser.add_argument("--model_to_analyse", type=str, default=None)
    parser.add_argument("--masking_algorithm", type=str, default=None)
    parser.add_argument("--pair_aggregation", type=str, default=None)
    parser.add_argument("--mask_granularity", type=str, default=None)
    parser.add_argument("--layers_to_analyse", nargs="+", default=None)
    parser.add_argument("--ablate_non_target_layers", action="store_true")
    parser.add_argument("--num_ig_steps", type=int, default=None)
    parser.add_argument("--no_negate_scores", action="store_true")
    parser.add_argument("--num_random_samples", type=int, default=5)
    parser.add_argument("--sparsities", type=float, nargs="+", default=None)
    parser.add_argument("--importance_sampling_method",
        choices=["snis", "geometric_mean", "tempered_snis"], default="snis",
        help="IS method saved into mask metadata so evaluate_mask.py uses it "
        "when computing IS-based metrics (does not affect suppression scores "
        "themselves, which are computed by raw KL).")
    parser.add_argument("--importance_sampling_temperature",
        type=float, default=None,
        help="Scalar temperature T for --importance_sampling_method tempered_snis. "
        "Saved into mask metadata and used at evaluation time.")

    args, _ = parser.parse_known_args()
    if args.config:
        from utils.expt_config import load_config
        config = load_config(args.config)
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("config", None)
    main(**kwargs)
