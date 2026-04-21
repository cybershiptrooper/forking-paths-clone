"""Evaluate a learned circuit mask by resampling branches through the masked model.

For each sparsity threshold, installs the mask as attention hooks, then
generates new branches via HuggingFace ``model.generate()``.  Saves a
sidecar JSON alongside the mask containing:

- Original branch completions and extracted answers
- Resampled branch completions and extracted answers per threshold
- Answer frequency distributions for unmasked / masked-IS / masked-resample
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import Sentence, clear_cuda
from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
)
from utils.circuit_eval import (
    build_binary_masks,
    build_token_to_sent_map,
    install_mask_hooks,
    install_non_target_ablation,
    remove_handles,
)
from utils.circuit_discovery.base import AblationHandle
from utils.completion_cache import load_from_cache, DEFAULT_CACHE_DIR
from utils.importance_sampling import extract_answer_ids, normalize_answer
from utils.rewards import extract_boxed


DEFAULT_SPARSITIES = [
    0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0,
]


def _load_model_eager(model_name: str, device: str = "cuda"):
    """Load model with eager attention (needed for attention hooks)."""
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


def _extract_answers(texts: list[str], prefix_text: str) -> tuple[list[str], list[int], list[str]]:
    """Extract boxed answers from branch texts, return (raw_answers, answer_ids, labels)."""
    raw = []
    no_ans = 0
    for t in texts:
        full = prefix_text + t
        boxed = extract_boxed(full)
        if boxed is not None:
            raw.append(normalize_answer(boxed))
        else:
            raw.append(f"__no_answer_{no_ans}")
            no_ans += 1

    unique: list[str] = []
    seen: dict[str, int] = {}
    for a in raw:
        if a not in seen:
            seen[a] = len(unique)
            unique.append(a)
    ids = [seen[a] for a in raw]
    return raw, ids, unique


def _answer_distribution(answer_ids: list[int], labels: list[str]) -> dict[str, float]:
    """Frequency-based answer distribution from answer IDs."""
    n = len(answer_ids)
    if n == 0:
        return {}
    counts: dict[int, int] = {}
    for aid in answer_ids:
        counts[aid] = counts.get(aid, 0) + 1
    return {labels[k]: counts[k] / n for k in sorted(counts)}


def _generate_branches(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_branches: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    batch_size: int = 1,
) -> list[dict]:
    """Generate *num_branches* from *input_ids* with HF generate, sampling.

    Sequences are drawn in chunks of *batch_size* via ``num_return_sequences``.
    Larger batches are much faster on long generations but use more KV-cache
    memory; tune down if you hit OOM.
    """
    prefix_len = input_ids.shape[-1]
    branches: list[dict] = []
    produced = 0
    chunk_idx = 0
    while produced < num_branches:
        chunk = min(batch_size, num_branches - produced)
        chunk_seed = seed + chunk_idx
        torch.manual_seed(chunk_seed)
        torch.cuda.manual_seed_all(chunk_seed)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=chunk,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        # outputs: (chunk, prefix_len + gen_len)
        for row in outputs:
            gen_ids = row[prefix_len:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            branches.append({"text": gen_text, "token_ids": gen_ids})
        produced += chunk
        chunk_idx += 1

    return branches


def evaluate_resample(
    mask_path: str,
    sparsities: list[float] | None = None,
    num_resample_branches: int | None = None,
    max_new_tokens: int | None = None,
    device: str = "cuda",
    cache_dir: str = DEFAULT_CACHE_DIR,
    seed: int = 42,
    batch_size: int = 1,
):
    """Resample branches through the masked model at each threshold.

    Saves a sidecar JSON at ``<mask_path_stem>_resample.json``.
    """
    if sparsities is None:
        sparsities = list(DEFAULT_SPARSITIES)

    # ==================================================================
    # Load mask and reconstruct from cache
    # ==================================================================
    print("=" * 80)
    print(f"Loading mask from {mask_path}...")
    print("=" * 80)

    node_mask = NodeMask.from_json(mask_path)
    meta = node_mask.metadata

    cache_key = meta.get("cache_key")
    if cache_key is None:
        raise ValueError(
            f"Mask at {mask_path} is missing 'cache_key' in metadata."
        )

    cached = load_from_cache(cache_key, cache_dir)
    if cached is None:
        raise ValueError(
            f"Cache file for key '{cache_key}' not found in {cache_dir}."
        )

    print(f"  Loaded {len(cached['branches'])} branches from cache ({cache_key})")

    input_ids = torch.tensor([cached["input_ids"]], device=device)
    sentences = [
        Sentence(start=s["start"], end=s["end"]) for s in node_mask.sentences
    ]

    temperature = meta.get("temperature", 0.6)
    original_num_branches = len(cached["branches"])
    if num_resample_branches is None:
        num_resample_branches = original_num_branches
    if max_new_tokens is None:
        # Match original branch length
        max_branch_len = max(len(b["token_ids"]) for b in cached["branches"])
        max_new_tokens = max_branch_len

    # ==================================================================
    # Load model
    # ==================================================================
    print("\n" + "=" * 80)
    print(f"Loading model with eager attention ({node_mask.model_name})...")
    print("=" * 80)

    model, tokenizer = _load_model_eager(node_mask.model_name, device=device)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)

    # ==================================================================
    # Prepare mask infrastructure
    # ==================================================================
    prefix_len = input_ids.shape[-1]
    num_heads = model.config.num_attention_heads
    num_sents = len(sentences)
    layers = node_mask.layers

    # Need enough token_to_sent space for generation
    max_total_len = prefix_len + max_new_tokens
    token_to_sent = build_token_to_sent_map(sentences, max_total_len, target_device)

    sentence_gap = meta.get("sentence_gap", 0)
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=target_device)

    mask_mode = meta.get("mask_mode", "prefix")
    num_prefix_sents = meta.get("num_prefix_sentences", num_sents)
    mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode, device=target_device)
    causal_filter = build_causal_filter(num_sents, device=target_device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

    ablate_non_target = meta.get("ablate_non_target_layers", False)
    renormalize = meta.get("renormalize_masked_attention", True)

    # ==================================================================
    # Original branches — extract answers
    # ==================================================================
    prefix_text = tokenizer.decode(cached["input_ids"], skip_special_tokens=True)
    orig_texts = [b["text"] for b in cached["branches"]]
    orig_answers, orig_ids, orig_labels = _extract_answers(orig_texts, prefix_text)
    orig_dist = _answer_distribution(orig_ids, orig_labels)

    correct_answer = meta.get("correct_answer")
    target_norm = normalize_answer(correct_answer) if correct_answer is not None else None
    def _fraction_correct(answers: list[str]) -> float | None:
        if target_norm is None or not answers:
            return None
        return sum(1 for a in answers if a == target_norm) / len(answers)

    orig_frac_correct = _fraction_correct(orig_answers)

    print(f"\nOriginal answer distribution ({len(cached['branches'])} branches):")
    for ans, prob in sorted(orig_dist.items(), key=lambda x: -x[1]):
        print(f"  {ans}: {prob:.1%}")
    if orig_frac_correct is not None:
        print(f"  fraction_correct (vs {correct_answer!r}): {orig_frac_correct:.1%}")

    # IS-based answer probs from existing evaluation (if available)
    existing_eval = meta.get("threshold_evaluation", [])

    # ==================================================================
    # Install non-target layer ablation (stays for all thresholds)
    # ==================================================================
    non_target_handles: list[AblationHandle] = []
    if ablate_non_target:
        non_target_handles = install_non_target_ablation(
            model, layers, num_heads, num_sents,
            token_to_sent, combined_filter, renormalize, target_device,
        )

    # ==================================================================
    # Resample at each threshold
    # ==================================================================
    thresholds = node_mask.thresholds_for_sparsities(sparsities)
    print(f"\nResampling at {len(thresholds)} thresholds...")
    print(f"  Branches per threshold: {num_resample_branches}")
    print(f"  Max new tokens: {max_new_tokens}")

    threshold_results = []

    for threshold in thresholds:
        sparsity = node_mask.sparsity(threshold, gap_filter=combined_filter.cpu())

        print(f"\n  threshold={threshold:.1e} | sparsity={sparsity:.2%}")

        # Build and install mask hooks
        binary_masks = build_binary_masks(
            node_mask, threshold, layers, num_heads, num_sents,
            combined_filter, target_device,
        )
        handles = install_mask_hooks(
            model, layers, binary_masks, token_to_sent,
            combined_filter, renormalize,
        )

        # Generate new branches
        model.eval()
        new_branches = _generate_branches(
            model, tokenizer, input_ids,
            num_resample_branches, max_new_tokens,
            temperature, seed,
            batch_size=batch_size,
        )

        remove_handles(handles)
        del binary_masks
        torch.cuda.empty_cache()

        # Extract answers from resampled branches
        new_texts = [b["text"] for b in new_branches]
        new_answers, new_ids, new_labels = _extract_answers(new_texts, prefix_text)
        new_dist = _answer_distribution(new_ids, new_labels)

        # Find matching IS-based fine-grained answer probs from existing eval.
        # Note: `answer_probs_masked` is BINARY (correct/incorrect) — we want
        # the fine-grained per-answer vector here, which is saved under
        # `answer_probs_masked_fine` and aligns with meta["answer_labels"].
        is_probs_fine = None
        answer_labels_is = meta.get("answer_labels")
        for ev in existing_eval:
            if abs(ev.get("threshold", float("inf")) - threshold) < 1e-12:
                is_probs_fine = ev.get("answer_probs_masked_fine")
                break

        is_dist_fine = None
        if is_probs_fine is not None and answer_labels_is is not None:
            is_dist_fine = {
                answer_labels_is[i]: is_probs_fine[i]
                for i in range(min(len(is_probs_fine), len(answer_labels_is)))
            }

        frac_correct = _fraction_correct(new_answers)
        entry = {
            "threshold": threshold,
            "sparsity": sparsity,
            "resample_branches": [
                {"text": b["text"], "answer": new_answers[i]}
                for i, b in enumerate(new_branches)
            ],
            "resample_answer_distribution": new_dist,
            "resample_fraction_correct": frac_correct,
            "masked_is_answer_distribution_fine": is_dist_fine,
        }
        threshold_results.append(entry)

        print(f"    Resample distribution: {new_dist}")
        if frac_correct is not None:
            print(f"    Resample fraction_correct: {frac_correct:.1%}")
        if is_dist_fine:
            print(f"    IS (fine) distribution:    {is_dist_fine}")

    # ==================================================================
    # Cleanup non-target ablation
    # ==================================================================
    remove_handles(non_target_handles)

    # ==================================================================
    # Build and save sidecar
    # ==================================================================
    # Unify all answer labels across original + all thresholds
    all_labels = set(orig_labels)
    for tr in threshold_results:
        all_labels.update(tr["resample_answer_distribution"].keys())
        if tr["masked_is_answer_distribution_fine"]:
            all_labels.update(tr["masked_is_answer_distribution_fine"].keys())
    all_labels = sorted(all_labels)

    sidecar = {
        "mask_path": mask_path,
        "model_name": node_mask.model_name,
        "num_resample_branches": num_resample_branches,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "batch_size": batch_size,
        "correct_answer": correct_answer,
        "correct_answer_normalized": target_norm,
        "answer_labels": all_labels,
        "original": {
            "branches": [
                {"text": orig_texts[i], "answer": orig_answers[i]}
                for i in range(len(orig_texts))
            ],
            "answer_distribution": orig_dist,
            "fraction_correct": orig_frac_correct,
        },
        "thresholds": threshold_results,
    }

    # Save sidecar alongside mask
    stem, ext = os.path.splitext(mask_path)
    sidecar_path = f"{stem}_resample{ext}"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"\nSaved resample sidecar to {sidecar_path}")

    # Cleanup
    del model
    clear_cuda()
    print("Done!")

    return sidecar_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample branches through a masked model at multiple thresholds"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--mask_path", type=str, required=True,
        help="Path to a NodeMask JSON file produced by learn_circuit.py.",
    )
    parser.add_argument(
        "--sparsities", type=float, nargs="+", default=DEFAULT_SPARSITIES,
        help="Target sparsity levels (0-1) for resampling.",
    )
    parser.add_argument(
        "--num_resample_branches", type=int, default=None,
        help="Number of branches to generate per threshold (default: same as original).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Max tokens to generate per branch (default: match original branch length).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
        help="Directory for completion cache.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of sequences generated per model.generate() call (via "
        "num_return_sequences). Larger is faster but uses more KV-cache "
        "memory; tune down on OOM.",
    )

    args, _ = parser.parse_known_args()
    if args.config:
        from utils.expt_config import load_config
        config = load_config(args.config)
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
    args = parser.parse_args()

    evaluate_resample(
        mask_path=args.mask_path,
        sparsities=args.sparsities,
        num_resample_branches=args.num_resample_branches,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        cache_dir=args.cache_dir,
        seed=args.seed,
        batch_size=args.batch_size,
    )
