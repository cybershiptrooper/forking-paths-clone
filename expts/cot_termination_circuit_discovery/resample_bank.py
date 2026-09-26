"""Regenerate a termination bank's training candidates under the current mask.

Used by train_with_periodic_bank_resampling.py between training segments.
The candidate bank all termination losses re-score is normally sampled once
from the clean (unmasked) model and stays fixed for the whole run, so it
becomes off-policy as the mask changes the model's behaviour.  This script
refreshes it: it loads the Hard-Concrete gate state from a training
checkpoint, keeps the top (1 - target_sparsity) fraction of sentence-pair
gates (the same matched-target readout used at evaluation), installs that
binary mask, samples fresh continuations from the analysis point, and
writes a new bank JSON.

Conventions (all deliberate, documented in the report):
- new candidates are LABELED under the clean model (mask removed for the
  probe), matching how the original bank's labels were produced;
- the held-out candidate half is copied unchanged from the original bank,
  so held-out diagnostics stay comparable across refreshes;
- for boundary-hazard objectives, per-boundary metadata is rebuilt for the
  new candidates but the wrap-up event token set is FROZEN from the
  original boundary data (it is a property of the prompt and model, not of
  the candidates).

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.resample_bank \
        --original_bank results/cot_termination/banks/aqua_p080_s57.json \
        --checkpoint results/snp_sweep/bank_resampling/ckpt/run_x.pt \
        --target_sparsity 0.1 \
        --seed_offset 1 \
        --output_bank results/snp_sweep/bank_resampling/banks/run_x_seg1.json \
        [--original_boundary_data ... --output_boundary_data ...]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from utils.utils import set_seed, clear_cuda
from utils.masks import (
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_prompt_filter,
    build_combined_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix,
    load_model_eager,
)
from expts.cot_termination_circuit_discovery.eval_utils import (
    build_binary_mask,
)
from expts.cot_termination_circuit_discovery.eval_termination_rollouts import (
    _install,
    _clear,
    _probe_label,
    PROBE_SUFFIX_TEXT,
    THINK_END_ID,
)
from expts.cot_termination_circuit_discovery.build_termination_bank import (
    to_candidate,
    TARGET_CLUSTER,
)
from expts.cot_termination_circuit_discovery.build_boundary_data import (
    _paragraph_boundaries,
    _clean_hazards,
    _probe_at_boundary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_bank", required=True,
                    help="The clean-model bank this run started from "
                    "(source of metadata and the held-out half).")
    ap.add_argument("--checkpoint", required=True,
                    help="Trainer checkpoint holding the current log_alpha.")
    ap.add_argument("--target_sparsity", type=float, required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_candidates", type=int, default=None,
                    help="Defaults to the original bank's training-half size.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sentence_gap", type=int, default=0)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="Added to the seed so each refresh draws fresh "
                    "samples (use the segment index).")
    ap.add_argument("--output_bank", required=True)
    ap.add_argument("--original_boundary_data", default=None)
    ap.add_argument("--output_boundary_data", default=None)
    ap.add_argument("--fallback_bank", default=None,
                    help="Bank used by the previous training segment. If "
                    "the fresh candidates are outcome-uniform (no cluster-0 "
                    "candidate, or nothing outside cluster 0 — either way "
                    "some losses would have no training signal), this bank "
                    "is copied to --output_bank instead.")
    ap.add_argument("--fallback_boundary_data", default=None)
    args = ap.parse_args()
    if bool(args.original_boundary_data) != bool(args.output_boundary_data):
        raise ValueError("Pass both --original_boundary_data and "
                         "--output_boundary_data, or neither.")
    set_seed(args.seed + 1000 * args.seed_offset)

    with open(args.original_bank) as f:
        bank = json.load(f)
    data_path = bank["data_path"]
    prompt_index = int(bank["prompt_index"])
    step = int(bank["analysis_sentence_step"])
    horizon = int(bank["horizon"])
    trace_answer = (bank["trace_answer"] or "").strip()
    all_letters = bank.get("all_letters") or ["A", "B", "C", "D"]
    n_candidates = args.n_candidates or len(bank["candidates"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]

    prefix_ids, sentences, _, _, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=int(bank.get("sentences_after_prefix", 0)),
        min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    state = torch.load(args.checkpoint, map_location="cpu",
                       weights_only=False)
    la = state["log_alpha"]
    la_key = "_tensor" if "_tensor" in la else "pair"
    log_alpha = la[la_key].float()
    while log_alpha.dim() > 2 and log_alpha.shape[0] == 1:
        log_alpha = log_alpha.squeeze(0)
    if log_alpha.shape != (num_sents, num_sents):
        raise ValueError(
            f"checkpoint log_alpha shape {tuple(log_alpha.shape)} != "
            f"({num_sents}, {num_sents})"
        )
    print(f"checkpoint step {state['step']}, log_alpha "
          f"[{log_alpha.min():.2f}, {log_alpha.max():.2f}]")

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode,
                                    device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    prompt_filter = build_prompt_filter(num_prompt_sentences, num_sents,
                                        device=device)
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter)
    valid_filter = ~combined_filter.bool()
    binary = build_binary_mask(log_alpha.to(device), "top_k",
                               args.target_sparsity, valid_filter)

    token_to_sent = torch.full((prefix_len + horizon + 8,), -1,
                               dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        token_to_sent[s.start:s.end + 1] = idx

    # ---- generate under the current mask ----
    _install(model, layers, binary, num_heads, token_to_sent, combined_filter)
    gen_ids_all = []
    while len(gen_ids_all) < n_candidates:
        b = min(args.batch_size, n_candidates - len(gen_ids_all))
        with torch.no_grad():
            out = model.generate(
                input_ids=prefix_ids.expand(b, -1),
                max_new_tokens=horizon,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        for row in out:
            gen = row[prefix_len:]
            # trim trailing padding after </think> is irrelevant; keep as-is
            gen_ids_all.append(gen.tolist())
        del out
        clear_cuda()
    _clear(model, layers)

    # ---- label under the CLEAN model (mask removed) ----
    samples = []
    for ids in gen_ids_all:
        think_pos = ids.index(THINK_END_ID) if THINK_END_ID in ids else None
        s = {
            "token_ids": ids,
            "n_tokens": len(ids),
            "think_pos": think_pos,
            "terminated": think_pos is not None,
            "probe_label": None,
            "probe_letter_probs": None,
            "regex_answer": None,
        }
        if think_pos is not None:
            label, probs = _probe_label(
                model, tokenizer, prefix_ids,
                torch.tensor(ids), think_pos, suffix_ids, letter_ids, device,
            )
            s["probe_label"] = label
            s["probe_letter_probs"] = probs
        if s["terminated"] and s["probe_label"] == trace_answer:
            s["cluster_id"] = 0
        elif s["terminated"]:
            s["cluster_id"] = 1
        else:
            s["cluster_id"] = 2
        samples.append(s)
        clear_cuda()
    counts = [sum(s["cluster_id"] == c for s in samples) for c in range(3)]
    print(f"resampled {len(samples)} candidates, clusters {counts}")

    n0 = counts[TARGET_CLUSTER]
    if n0 == 0 or n0 == len(samples):
        # Outcome-uniform refresh: no cluster-0 candidate (nothing to
        # reward) or everything in cluster 0 (pairwise losses have no
        # pairs).  Keep the previous segment's bank instead of writing a
        # degenerate one; the refresh interval effectively becomes
        # adaptive.  This happens legitimately when the mask already
        # terminates (or never terminates) essentially every sample.
        fb = args.fallback_bank or args.original_bank
        print(f"outcome-uniform refresh (clusters {counts}); keeping the "
              f"previous bank {fb}")
        with open(fb) as f:
            prev = json.load(f)
        prev["resample_kept_previous_bank"] = True
        prev["resample_uniform_cluster_counts"] = counts
        prev["resampled_from_checkpoint"] = args.checkpoint
        prev["resampled_at_step"] = int(state["step"])
        os.makedirs(os.path.dirname(args.output_bank), exist_ok=True)
        with open(args.output_bank, "w") as f:
            json.dump(prev, f)
        print(f"wrote {args.output_bank} (previous bank, carried forward)")
        if args.original_boundary_data:
            fbd = args.fallback_boundary_data or args.original_boundary_data
            with open(fbd) as f:
                prev_bd = json.load(f)
            prev_bd["bank_path"] = args.output_bank
            prev_bd["resample_kept_previous_bank"] = True
            os.makedirs(os.path.dirname(args.output_boundary_data),
                        exist_ok=True)
            with open(args.output_boundary_data, "w") as f:
                json.dump(prev_bd, f)
            print(f"wrote {args.output_boundary_data} "
                  f"(previous boundary data, carried forward)")
        remove_handles(handles)
        return

    new_bank = dict(bank)
    new_bank["candidates"] = [to_candidate(s) for s in samples]
    # held-out half copied unchanged from the original bank
    new_bank["heldout_candidates"] = bank["heldout_candidates"]
    new_bank["resampled_from_checkpoint"] = args.checkpoint
    new_bank["resampled_at_step"] = int(state["step"])
    new_bank["resample_seed_offset"] = args.seed_offset
    new_bank["resample_cluster_counts"] = counts
    os.makedirs(os.path.dirname(args.output_bank), exist_ok=True)
    with open(args.output_bank, "w") as f:
        json.dump(new_bank, f)
    print(f"wrote {args.output_bank}")

    # ---- boundary metadata for the new candidates (hazard objectives) ----
    if args.original_boundary_data:
        with open(args.original_boundary_data) as f:
            bd = json.load(f)
        event_token_ids = bd["event_token_ids"]  # FROZEN from the original
        out_candidates = []
        for flat_idx, s in enumerate(samples):
            ids = s["token_ids"]
            terminated = s["terminated"]
            content = ids[: s["think_pos"]] if terminated else list(ids)
            boundaries = _paragraph_boundaries(content, tokenizer)
            clean_log_h = _clean_hazards(
                model, prefix_ids, content, boundaries, event_token_ids,
                device,
            )
            eligible, probe_labels, probe_p_trace = [], [], []
            for j in boundaries:
                label, probs = _probe_at_boundary(
                    model, prefix_ids, content, j, suffix_ids, letter_ids,
                    device,
                )
                eligible.append(label == trace_answer)
                probe_labels.append(label)
                probe_p_trace.append(probs.get(trace_answer))
            gaps = [
                float(boundaries[b + 1] - boundaries[b])
                for b in range(len(boundaries) - 1)
            ] + ([float(max(1, horizon - boundaries[-1]))]
                 if boundaries else [])
            out_candidates.append({
                "flat_index": flat_idx,
                "cluster_id": s["cluster_id"],
                "terminated": terminated,
                "n_content_tokens": len(content),
                "boundaries": boundaries,
                "eligible": eligible,
                "probe_labels": probe_labels,
                "probe_p_trace": probe_p_trace,
                "clean_log_h": clean_log_h,
                "gaps": gaps,
            })
            clear_cuda()
        new_bd = dict(bd)
        new_bd["bank_path"] = args.output_bank
        new_bd["candidates"] = out_candidates
        new_bd["resampled_from_checkpoint"] = args.checkpoint
        os.makedirs(os.path.dirname(args.output_boundary_data), exist_ok=True)
        with open(args.output_boundary_data, "w") as f:
            json.dump(new_bd, f)
        n_any = sum(1 for c in out_candidates if any(c["eligible"]))
        print(f"wrote {args.output_boundary_data} "
              f"({n_any}/{len(out_candidates)} candidates with >=1 eligible "
              f"boundary)")
    remove_handles(handles)


if __name__ == "__main__":
    main()
