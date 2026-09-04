"""Evaluate per-sentence masks via attention masking and prefix compression.

Accepts masks from column-SNP, suppress, or thought anchors (pair scores
aggregated to per-sentence).  Three evaluation modes:

1. **Attention masking** — zero all attention to dropped sentences (full
   token sequence, positions unchanged).
2. **Prefix compression** — physically remove dropped sentences from the
   token sequence and run the model on the shorter input.
3. **Token KL** — force the first sentence after the context region and
   measure per-token KL between clean and compressed logits.

Supports sweeping over multiple num_keep values and probe suffixes in a
single model load.

Usage::

    # Single num_keep
    uv run python -m expts.direct_answer_circuit_discovery.eval_column \
        --mask_path results/.../mask.json \
        --num_keep 10 \
        --output results/.../mask.eval_column.json

    # Sweep num_keep values
    uv run python -m expts.direct_answer_circuit_discovery.eval_column \
        --mask_path results/.../mask.json \
        --num_keep_list 3,5,10,15,20,25,30,35 \
        --output results/.../mask.eval_column.json

    # With alternate suffix
    uv run python -m expts.direct_answer_circuit_discovery.eval_column \
        --mask_path results/.../mask.json \
        --num_keep 10 \
        --probe_suffix_list "default,so_answer" \
        --output results/.../mask.eval_column.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_prompt_filter,
    build_combined_filter,
)
from utils.circuit_eval import build_token_to_sent_map, Sentence
from utils.utils import set_seed, clear_cuda, get_attention_module
from utils.base_path_selection import select_base_from_record

from expts.direct_answer_circuit_discovery.probe import (
    build_answer_probe,
    answer_probs_from_logits,
)
from expts.direct_answer_circuit_discovery.learn import _build_prefix

from utils.cot_analysis import (
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)


DEFAULT_SUFFIX = " </think> I think the answer is"
DEFAULT_ANSWER_LETTERS = [" A", " B", " C", " D"]

SUFFIX_ALIASES = {
    "default": " </think> I think the answer is",
    "so_answer": " </think> So, the answer is",
}


def _scores_to_per_sentence(nm: NodeMask) -> List[float]:
    """Extract a per-sentence importance vector from any mask granularity."""
    g = nm.granularity
    if g == "column":
        return list(nm.scores)
    if g == "pair":
        S = len(nm.scores)
        per_sent = []
        for j in range(S):
            col_vals = [nm.scores[i][j] for i in range(S) if nm.scores[i][j] != 0]
            per_sent.append(sum(col_vals) / len(col_vals) if col_vals else 0.0)
        return per_sent
    # Suppress: scores is a 2D list where all rows are identical (broadcast)
    if isinstance(nm.scores, list) and nm.scores and isinstance(nm.scores[0], list):
        return list(nm.scores[0])
    # Already 1D
    return list(nm.scores)


def _select_kept_sentences(
    scores: List[float],
    num_frozen_prompt: int = 0,
    sparsity: Optional[float] = None,
    num_keep: Optional[int] = None,
) -> List[int]:
    """Return indices of sentences to KEEP.

    If *num_frozen_prompt* > 0, the first that many sentences (prompt
    sentences) are always kept; sparsity/num_keep applies only to the
    remaining reasoning sentences.
    """
    S = len(scores)
    frozen = set(range(num_frozen_prompt))
    rankable = [j for j in range(S) if j not in frozen]
    if num_keep is not None:
        n_keep_reasoning = max(1, min(num_keep, len(rankable)))
    elif sparsity is not None:
        n_keep_reasoning = max(1, int(round((1.0 - sparsity) * len(rankable))))
    else:
        raise ValueError("Either sparsity or num_keep must be provided")
    ranked = sorted(rankable, key=lambda j: scores[j], reverse=True)
    kept = frozen | set(ranked[:n_keep_reasoning])
    return sorted(kept)


def _build_column_mask(
    num_sents: int,
    kept: List[int],
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (num_heads, S, S) binary mask that zeros columns of dropped sentences."""
    mask_2d = torch.zeros(num_sents, num_sents, device=device)
    for j in kept:
        mask_2d[:, j] = 1.0
    return mask_2d.unsqueeze(0).expand(num_heads, -1, -1).contiguous()


def _kl(clean_p: torch.Tensor, masked_p: torch.Tensor) -> float:
    return F.kl_div(
        masked_p.log().unsqueeze(0),
        clean_p.log().unsqueeze(0),
        log_target=True,
        reduction="batchmean",
    ).item()


def _token_kl(clean_logits: torch.Tensor, masked_logits: torch.Tensor) -> float:
    """Mean per-token KL(clean || masked) over all positions."""
    clean_lp = F.log_softmax(clean_logits, dim=-1)
    masked_lp = F.log_softmax(masked_logits, dim=-1)
    # KL(clean || masked) = sum_v clean(v) * (log clean(v) - log masked(v))
    kl_per_token = (clean_lp.exp() * (clean_lp - masked_lp)).sum(dim=-1)
    return kl_per_token.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-sentence masks")
    parser.add_argument("--mask_path", required=True)
    # Sparsity specification: one of these three
    parser.add_argument("--sparsity", type=float, default=None,
                        help="Fractional sparsity (legacy). Use --num_keep or --num_keep_list instead.")
    parser.add_argument("--num_keep", type=int, default=None,
                        help="Absolute number of reasoning sentences to keep.")
    parser.add_argument("--num_keep_list", type=str, default=None,
                        help="Comma-separated list of num_keep values to sweep (e.g. 3,5,10,15,20,25,30,35).")
    # Model/data overrides
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--analysis_sentence_step", type=int, default=None)
    parser.add_argument("--sentences_after_prefix", type=int, default=None)
    parser.add_argument("--sentence_gap", type=int, default=None)
    # Probe suffix
    parser.add_argument("--probe_suffix", type=str, default=DEFAULT_SUFFIX)
    parser.add_argument("--probe_suffix_list", type=str, default=None,
                        help="Comma-separated suffix aliases to sweep (e.g. 'default,so_answer').")
    parser.add_argument("--answer_letters", type=str, nargs="+", default=None)
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default="eager",
                        choices=["eager", "sdpa"])
    parser.add_argument("--token_kl", action="store_true", default=False,
                        help="Compute token-level KL on the next sentence after context.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- Resolve num_keep list ----
    if args.num_keep_list is not None:
        num_keep_values = [int(x.strip()) for x in args.num_keep_list.split(",")]
    elif args.num_keep is not None:
        num_keep_values = [args.num_keep]
    elif args.sparsity is not None:
        num_keep_values = None  # will use sparsity path
    else:
        raise ValueError("One of --sparsity, --num_keep, or --num_keep_list is required.")

    # ---- Resolve suffix list ----
    if args.probe_suffix_list is not None:
        suffix_keys = [s.strip() for s in args.probe_suffix_list.split(",")]
        suffixes = {k: SUFFIX_ALIASES.get(k, k) for k in suffix_keys}
    else:
        suffixes = {"default": args.probe_suffix}

    # ---- Load mask and extract per-sentence scores ----
    nm = NodeMask.from_json(args.mask_path)
    md = nm.metadata

    model_name = args.model_name or nm.model_name
    data_path = args.data_path or md.get("data_path")
    prompt_index = args.prompt_index if args.prompt_index is not None else md.get("prompt_index")
    ass = args.analysis_sentence_step or md.get("analysis_sentence_step")
    k = args.sentences_after_prefix if args.sentences_after_prefix is not None else md.get("sentences_after_prefix", 0)
    sentence_gap = args.sentence_gap if args.sentence_gap is not None else md.get("sentence_gap", 0)
    answer_letters = args.answer_letters or list(DEFAULT_ANSWER_LETTERS)

    per_sent_scores_raw = _scores_to_per_sentence(nm)

    print(f"Mask: {args.mask_path}")
    print(f"  granularity={nm.granularity}, algorithm={nm.algorithm}")
    print(f"  mask has {len(per_sent_scores_raw)} sentence scores")

    # ---- Build prefix ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix_ids, sentences, prompt_text, correct_answer, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer,
        prompt=None,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=ass,
        sentences_after_prefix=k,
        min_sentence_length=10,
        sentence_chunk=1,
    )
    prefix_len = prefix_ids.shape[-1]

    # Truncate scores to match the masked sentence count
    num_sents = len(sentences)
    per_sent_scores = per_sent_scores_raw[:num_sents]
    if len(per_sent_scores_raw) > num_sents:
        print(f"  (truncated {len(per_sent_scores_raw)} scores to {num_sents} masked sentences)")

    num_frozen = md.get("num_frozen_prompt_sentences", 0)
    n_reasoning = num_sents - num_frozen
    if num_frozen > 0:
        print(f"  frozen prompt sentences: {num_frozen} (always kept)")
    print(f"  {num_sents} sentences ({num_frozen} prompt + {n_reasoning} reasoning)")

    # ---- Build continuation sentence for token KL ----
    continuation_ids = None
    if args.token_kl:
        with open(data_path) as f:
            records = json.load(f)
        record = records[prompt_index]
        base_token_ids = select_base_from_record(record, "stored", tokenizer)
        full_token_ids = torch.tensor(
            record["prompt_token_ids"] + list(base_token_ids)
        )
        all_sents = split_tokens_into_sentences(
            full_token_ids, tokenizer, min_sentence_length=10
        )
        all_sents = remove_bos_from_sentences(all_sents)
        all_sents = chunk_sentences(all_sents, 1)
        cont_sent_idx = ass + k
        if cont_sent_idx < len(all_sents):
            s = all_sents[cont_sent_idx]
            continuation_ids = full_token_ids[s.start : s.end + 1].unsqueeze(0)
            cont_text = tokenizer.decode(continuation_ids[0])
            print(f"  token KL continuation sentence [{cont_sent_idx}]: {cont_text[:80]}...")
        else:
            print(f"  WARNING: no continuation sentence at index {cont_sent_idx} (only {len(all_sents)} sentences). Skipping token KL.")
            continuation_ids = None

    # ---- If using sparsity instead of num_keep, convert ----
    if num_keep_values is None:
        nk = max(1, int(round((1.0 - args.sparsity) * n_reasoning)))
        num_keep_values = [nk]

    print(f"  num_keep sweep: {num_keep_values}")
    print(f"  suffix sweep: {list(suffixes.keys())}")

    # ---- Load model ----
    print(f"Loading {model_name} (eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    all_layers = list(range(num_layers))

    # ---- Build filters ----
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix", device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    prompt_filter_t = build_prompt_filter(num_frozen, num_sents, device=device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter, prompt_filter_t)

    from utils.circuit_eval import install_mask_hooks, remove_handles

    # ---- Per-suffix evaluation ----
    all_suffix_results = {}
    for suffix_key, suffix_text in suffixes.items():
        print(f"\n{'='*60}")
        print(f"Suffix: {suffix_key!r} = {suffix_text!r}")
        print(f"{'='*60}")

        probe = build_answer_probe(tokenizer, suffix_text, answer_letters)
        suffix_ids = probe.suffix_ids
        full_input = torch.cat([prefix_ids, suffix_ids.unsqueeze(0)], dim=-1).to(device)
        token_to_sent = build_token_to_sent_map(sentences, full_input.shape[-1], device)

        # ---- Clean reference ----
        print("  Computing clean reference...")
        with torch.no_grad():
            clean_logits = model(full_input).logits
        clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
        del clean_logits
        clear_cuda()

        # ---- KL_max ----
        frozen_kept = list(range(num_frozen)) if num_frozen > 0 else []
        zero_mask = _build_column_mask(num_sents, frozen_kept, num_heads, device)
        zero_masks = {l: zero_mask for l in all_layers}
        handles = install_mask_hooks(
            model, all_layers, zero_masks, token_to_sent, combined_filter,
            renormalize=True,
        )
        with torch.no_grad():
            logits = model(full_input).logits
        p_zero = answer_probs_from_logits(logits, probe, prefix_len).cpu()
        remove_handles(handles)
        del logits
        clear_cuda()
        kl_max = _kl(clean_p, p_zero)
        print(f"  KL_max = {kl_max:.6f}")

        # ---- Clean token KL reference (on uncompressed prefix + continuation) ----
        clean_cont_logits = None
        if continuation_ids is not None:
            clean_cont_input = torch.cat([prefix_ids, continuation_ids], dim=-1).to(device)
            with torch.no_grad():
                clean_cont_logits = model(clean_cont_input).logits
            # Keep only the logits at continuation token positions
            cont_start = prefix_len
            cont_end = cont_start + continuation_ids.shape[-1]
            clean_cont_logits = clean_cont_logits[0, cont_start - 1 : cont_end - 1, :].cpu()
            del clean_cont_input
            clear_cuda()

        # ---- Sweep num_keep ----
        sweep_rows = []
        for nk in num_keep_values:
            kept = _select_kept_sentences(per_sent_scores, num_frozen, num_keep=nk)
            dropped = [j for j in range(num_sents) if j not in set(kept)]
            n_kept_reasoning = len([j for j in kept if j >= num_frozen])
            actual_sparsity = len(dropped) / n_reasoning if n_reasoning > 0 else 0.0

            print(f"\n  --- num_keep={nk} (actual kept reasoning: {n_kept_reasoning}) ---")

            row = {
                "num_keep": nk,
                "num_kept_reasoning": n_kept_reasoning,
                "sparsity": actual_sparsity,
                "kept_sentences": kept,
            }

            # Eval 1: attention masking
            attn_mask = _build_column_mask(num_sents, kept, num_heads, device)
            attn_masks = {l: attn_mask for l in all_layers}
            handles = install_mask_hooks(
                model, all_layers, attn_masks, token_to_sent, combined_filter,
                renormalize=True,
            )
            with torch.no_grad():
                logits = model(full_input).logits
            p_attn = answer_probs_from_logits(logits, probe, prefix_len).cpu()
            remove_handles(handles)
            del logits
            clear_cuda()

            kl_attn = _kl(clean_p, p_attn)
            kl_norm_attn = kl_attn / kl_max if kl_max > 0 else float("nan")
            row["attn_kl"] = kl_attn
            row["attn_kl_norm"] = kl_norm_attn
            row["attn_answer_probs"] = p_attn.tolist()
            print(f"    attn KL={kl_attn:.6f}, KL_norm={kl_norm_attn:.4f}")

            # Eval 2: prefix compression
            kept_token_ids = []
            for j in kept:
                s = sentences[j]
                kept_token_ids.append(prefix_ids[0, s.start : s.end + 1])
            if len(sentences) > 0:
                last_masked_end = sentences[-1].end
                if last_masked_end + 1 < prefix_len:
                    kept_token_ids.append(prefix_ids[0, last_masked_end + 1 : prefix_len])
            if kept_token_ids:
                compressed_prefix = torch.cat(kept_token_ids).unsqueeze(0)
            else:
                compressed_prefix = prefix_ids[:, :1]

            compressed_input = torch.cat([compressed_prefix, suffix_ids.unsqueeze(0)], dim=-1).to(device)
            compressed_prefix_len = compressed_prefix.shape[-1]

            with torch.no_grad():
                logits = model(compressed_input).logits
            p_comp = answer_probs_from_logits(logits, probe, compressed_prefix_len).cpu()
            del logits
            clear_cuda()

            kl_comp = _kl(clean_p, p_comp)
            kl_norm_comp = kl_comp / kl_max if kl_max > 0 else float("nan")
            row["comp_kl"] = kl_comp
            row["comp_kl_norm"] = kl_norm_comp
            row["comp_answer_probs"] = p_comp.tolist()
            row["compressed_prefix_len"] = compressed_prefix_len
            row["original_prefix_len"] = prefix_len
            print(f"    comp KL={kl_comp:.6f}, KL_norm={kl_norm_comp:.4f}")

            # Eval 3: token KL (compression + force next sentence)
            if continuation_ids is not None and clean_cont_logits is not None:
                comp_cont_input = torch.cat(
                    [compressed_prefix, continuation_ids], dim=-1
                ).to(device)
                with torch.no_grad():
                    comp_cont_logits_full = model(comp_cont_input).logits
                cont_start_comp = compressed_prefix_len
                cont_end_comp = cont_start_comp + continuation_ids.shape[-1]
                comp_cont_logits = comp_cont_logits_full[
                    0, cont_start_comp - 1 : cont_end_comp - 1, :
                ].cpu()
                del comp_cont_logits_full, comp_cont_input
                clear_cuda()

                tkl = _token_kl(clean_cont_logits, comp_cont_logits)
                row["token_kl"] = tkl
                print(f"    token KL={tkl:.6f}")

            sweep_rows.append(row)

        all_suffix_results[suffix_key] = {
            "suffix_text": suffix_text,
            "kl_max": kl_max,
            "rows": sweep_rows,
        }

    # ---- Save ----
    result = {
        "mask_path": args.mask_path,
        "model_name": model_name,
        "prompt_index": prompt_index,
        "algorithm": nm.algorithm,
        "granularity": nm.granularity,
        "num_sentences": num_sents,
        "num_frozen_prompt_sentences": num_frozen,
        "num_reasoning_sentences": n_reasoning,
        "sentence_gap": sentence_gap,
        "analysis_sentence_step": ass,
        "sentences_after_prefix": k,
        "suffixes": all_suffix_results,
    }

    # Backward-compat fields for single-suffix, single-num_keep usage
    if len(suffixes) == 1 and len(num_keep_values) == 1:
        suf = next(iter(all_suffix_results.values()))
        r = suf["rows"][0]
        result["target_sparsity"] = r["sparsity"]
        result["actual_sparsity"] = r["sparsity"]
        result["num_kept"] = r["num_kept_reasoning"] + num_frozen
        result["kl_max"] = suf["kl_max"]
        result["rows"] = [
            {
                "row": "kl_max",
                "mode": "all_zero",
                "sparsity": 1.0,
                "kl": suf["kl_max"],
            },
            {
                "row": "eval_attention",
                "mode": "column_mask",
                "target_sparsity": r["sparsity"],
                "sparsity": r["sparsity"],
                "kl": r["attn_kl"],
                "kl_normalized": r["attn_kl_norm"],
                "answer_probs": r.get("attn_answer_probs"),
                "kept_sentences": r["kept_sentences"],
            },
            {
                "row": "eval_compression",
                "mode": "prefix_compression",
                "target_sparsity": r["sparsity"],
                "sparsity": r["sparsity"],
                "kl": r["comp_kl"],
                "kl_normalized": r["comp_kl_norm"],
                "answer_probs": r.get("comp_answer_probs"),
                "kept_sentences": r["kept_sentences"],
                "compressed_prefix_len": r.get("compressed_prefix_len"),
                "original_prefix_len": r.get("original_prefix_len"),
            },
        ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
