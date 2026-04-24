"""Evaluate a learned circuit mask at multiple sparsity thresholds.

Loads a NodeMask JSON (produced by learn_circuit.py), reconstructs the
tensors from the completion cache, runs evaluate_at_thresholds, and writes
the results back into the same JSON file.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import Sentence, clear_cuda
from utils.objectives import get_objective
from utils.masks import NodeMask
from utils.circuit_eval import evaluate_at_thresholds
from utils.completion_cache import load_from_cache, DEFAULT_CACHE_DIR
from utils.rewards import find_answer_token_positions
from utils.importance_sampling import (
    merge_no_answer_variants,
    build_binary_answer_ids,
    reward_based_answer_ids,
)


DEFAULT_SPARSITIES = [
    0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
]


def _load_model_eager(model_name: str, device: str = "cuda"):
    """Load model with eager attention for evaluation (needs attention weights)."""
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


def evaluate(
    mask_path: str,
    sparsities: list[float] = None,
    num_random_samples: int = 5,
    device: str = "cuda",
    cache_dir: str = DEFAULT_CACHE_DIR,
):
    """Evaluate a saved NodeMask at multiple sparsity thresholds.

    Loads branches and input_ids from the completion cache (keyed by
    cache_key in mask metadata), runs evaluation, and writes results back.
    """
    if sparsities is None:
        sparsities = list(DEFAULT_SPARSITIES)

    # ``device="auto"`` means the HF model will be sharded via device_map;
    # staging tensors still need a concrete device. Use cuda:0 for staging —
    # they are moved to the model's device after load.
    tensor_staging_device = "cuda:0" if device == "auto" else device

    # =====================================================================
    # Load mask and reconstruct tensors from cache
    # =====================================================================
    print("=" * 80)
    print(f"Loading mask from {mask_path}...")
    print("=" * 80)

    node_mask = NodeMask.from_json(mask_path)
    meta = node_mask.metadata

    cache_key = meta.get("cache_key")
    if cache_key is None:
        raise ValueError(
            f"Mask at {mask_path} is missing 'cache_key' in metadata. "
            f"Was it produced by the new learn_circuit.py?"
        )

    cached = load_from_cache(cache_key, cache_dir)
    if cached is None:
        raise ValueError(
            f"Cache file for key '{cache_key}' not found in {cache_dir}. "
            f"Re-run learn_circuit.py to regenerate the cache."
        )

    print(f"  Loaded {len(cached['branches'])} branches from cache ({cache_key})")

    input_ids = torch.tensor([cached["input_ids"]], device=tensor_staging_device)
    sentences = [
        Sentence(start=s["start"], end=s["end"]) for s in node_mask.sentences
    ]
    continuations = [
        torch.tensor([b["token_ids"]], device=tensor_staging_device)
        for b in cached["branches"]
    ]
    branch_rewards = meta.get("branch_rewards")

    # Build fine-grained answer IDs (for answer_kl) from metadata,
    # merging __no_answer_* variants into a single __no_answer bucket.
    answer_ids_fine = None
    num_answers_fine = None
    if "answer_ids" in meta and "answer_labels" in meta:
        ids_list = meta["answer_ids"]
        labels = meta["answer_labels"]
        ids_list, labels, num_answers_fine = merge_no_answer_variants(ids_list, labels)
        answer_ids_fine = torch.tensor(ids_list, dtype=torch.long)
        print(f"  Fine-grained answer groups ({num_answers_fine}): {labels}")

    # Build binary answer IDs (for reward_gap) — correct vs incorrect.
    # Prefer reward-sign bucketing when branch_rewards is available (matches
    # the learn-phase bucketing exactly). Fall back to string-match against
    # correct_answer when there are no rewards.
    answer_ids_binary = None
    num_answers_binary = None
    correct_answer = meta.get("correct_answer")
    binary_source = None
    if branch_rewards is not None:
        bin_ids, bin_labels = reward_based_answer_ids(branch_rewards, binary=True)
        # reward_based_answer_ids collapses to one label if all rewards share
        # sign — in that case binary bucketing is meaningless. Skip it.
        if len(bin_labels) == 2:
            answer_ids_binary = torch.tensor(bin_ids, dtype=torch.long)
            num_answers_binary = 2
            binary_source = "reward_sign"
    if answer_ids_binary is None and answer_ids_fine is not None and correct_answer is not None:
        bin_ids, bin_labels, num_answers_binary = build_binary_answer_ids(
            ids_list, labels, correct_answer,
        )
        answer_ids_binary = torch.tensor(bin_ids, dtype=torch.long)
        binary_source = "correct_answer_match"
    if answer_ids_binary is not None:
        bin_ids_list = answer_ids_binary.tolist()
        n_correct = sum(1 for b in bin_ids_list if b == 0)
        n_incorrect = len(bin_ids_list) - n_correct
        print(f"  Binary answer groups (source={binary_source}): "
              f"{n_correct} correct, {n_incorrect} incorrect")
        if n_correct == 0 or n_incorrect == 0:
            print(
                f"  WARNING: binary bucketing collapsed — one side empty. "
                f"reward_gap / p_target / contrastive metrics will be degenerate."
            )
            if binary_source == "correct_answer_match":
                print(
                    f"    correct_answer={correct_answer!r} did not match any "
                    f"label in {labels!r}. Consider updating normalize_answer."
                )

    # Recompute position_mask_overrides if answer_only
    position_mask_overrides = None
    if meta.get("answer_only"):
        print("Rebuilding answer-only position masks...")
        tokenizer = AutoTokenizer.from_pretrained(node_mask.model_name)
        prefix_len = len(cached["input_ids"])
        position_mask_overrides = []
        for b in cached["branches"]:
            pm = find_answer_token_positions(
                b["text"], b["token_ids"], tokenizer, prefix_len
            )
            if pm is not None:
                pm = pm.to(tensor_staging_device)
            position_mask_overrides.append(pm)
        n_found = sum(1 for pm in position_mask_overrides if pm is not None)
        print(f"  Found answer tokens in {n_found}/{len(cached['branches'])} branches")
        if n_found == 0:
            print("  WARNING: No answer tokens found. Falling back to full mask.")
            position_mask_overrides = None

    # =====================================================================
    # Load model
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Loading model with eager attention ({node_mask.model_name})...")
    print("=" * 80)

    model, tokenizer = _load_model_eager(node_mask.model_name, device=device)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
    continuations = [c.to(target_device) for c in continuations]
    if position_mask_overrides is not None:
        position_mask_overrides = [
            pm.to(target_device) if pm is not None else None
            for pm in position_mask_overrides
        ]

    # =====================================================================
    # Evaluate at thresholds
    # =====================================================================
    print("\n" + "=" * 80)
    print("Evaluating sparsity vs KL at thresholds...")
    print("=" * 80)

    objective_fn = get_objective(meta["objective"])
    thresholds = node_mask.thresholds_for_sparsities(sparsities)
    print(f"  Target sparsities: {sparsities}")
    print(f"  Derived {len(thresholds)} thresholds from mask scores")

    is_method = meta.get("importance_sampling_method")
    if is_method is None:
        print(
            "  Warning: mask has no 'importance_sampling_method' in metadata "
            "(produced before the Part 1 refactor). Defaulting to 'snis'."
        )
        is_method = "snis"
    is_temperature = meta.get("importance_sampling_temperature")
    if is_method == "tempered_snis" and is_temperature is None:
        raise ValueError(
            "mask metadata has importance_sampling_method='tempered_snis' "
            "but no 'importance_sampling_temperature'."
        )

    threshold_results = evaluate_at_thresholds(
        model=model,
        node_mask=node_mask,
        input_ids=input_ids,
        sentences=sentences,
        continuations=continuations,
        objective_fn=objective_fn,
        thresholds=thresholds,
        layers=node_mask.layers,
        ablate_non_target_layers=meta.get("ablate_non_target_layers", False),
        renormalize_masked_attention=meta.get("renormalize_masked_attention", True),
        tokenizer=tokenizer,
        num_random_samples=num_random_samples,
        branch_rewards=branch_rewards,
        position_mask_overrides=position_mask_overrides,
        answer_ids_fine=answer_ids_fine,
        num_answers_fine=num_answers_fine,
        answer_ids_binary=answer_ids_binary,
        num_answers_binary=num_answers_binary,
        temperature=meta.get("temperature", 1.0),
        importance_sampling_method=is_method,
        importance_sampling_temperature=is_temperature,
    )

    # =====================================================================
    # Write results back
    # =====================================================================
    node_mask.metadata["threshold_evaluation"] = threshold_results
    node_mask.metadata["num_random_samples"] = num_random_samples

    node_mask.to_json(mask_path)
    print(f"\nSaved evaluation results to {mask_path}")

    # Print summary
    print("\nThreshold evaluation:")
    for r in threshold_results:
        print(
            f"  t={r['threshold']:.1e} → sparsity={r['sparsity']:.2%}, "
            f"KL={r['kl_divergence']:.2e}"
        )

    # Cleanup
    del model
    clear_cuda()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a learned circuit mask at multiple sparsity thresholds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to a NodeMask JSON file produced by learn_circuit.py.",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=DEFAULT_SPARSITIES,
        help="Target sparsity levels (0-1) for evaluation. Thresholds are "
        "computed dynamically from the learned mask scores.",
    )
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=3,
        help="Number of random score masks (K) to sample for baseline comparison.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory for completion cache (must match learn_circuit.py).",
    )

    # First parse to check for --config
    args, _ = parser.parse_known_args()
    if args.config:
        from utils.expt_config import load_config

        config = load_config(args.config)
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
    # Re-parse with config-informed defaults
    args = parser.parse_args()

    evaluate(
        mask_path=args.mask_path,
        sparsities=args.sparsities,
        num_random_samples=args.num_random_samples,
        device=args.device,
        cache_dir=args.cache_dir,
    )
