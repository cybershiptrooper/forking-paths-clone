"""Scan (prompt, analysis step) pairs for bimodal termination behaviour.

For every record in the given collection files and every analysis-step
fraction, samples N continuations of at most --horizon tokens from the
reasoning prefix with vLLM (temperature 1.0 — the candidate-bank
convention: SNIS treats candidates as samples from the untempered
model), then labels each continuation:

- terminated: the dedicated ``</think>`` token appears in the sample;
- probe label: for terminated samples, the answer letter read from the
  forced suffix ``" I think the answer is"`` placed immediately after
  the sample's own ``</think>`` token (single extra forward via vLLM
  prompt + 1-token logprobs).  This avoids depending on the sample's
  post-``</think>`` prose, which the horizon usually truncates;
- cluster 0 = terminated AND probe label == the trace's own final
  answer (record ``clean_answer``); cluster 1 = terminated with a
  different or undecided label; cluster 2 = not terminated.

The target event is therefore "terminate early with the conclusion this
trace eventually reached" — deliberately NOT gated on the gold answer,
so traces committed to a wrong answer still have a non-empty target
cluster (agreement with gold is recorded separately for analysis).

Usage (one GPU):
    uv run python -m expts.cot_termination_circuit_discovery.scan_termination \
        --data_paths data/collection/qwen3_8b/aqua_filtered.json \
                     data/collection/qwen3_8b/gpqa_filtered.json \
                     data/collection/qwen3_8b/aqua.json \
                     data/collection/qwen3_8b/gpqa.json \
        --output_dir results/cot_termination/scan_v2
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from utils.cot_analysis import (
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)
from expts.cot_termination_circuit_discovery.grading import extract_letter

MIN_SENTENCE_LENGTH = 10
THINK_END_ID = 151668  # dedicated "</think>" token in the Qwen3 vocab
PROBE_SUFFIX_TEXT = " I think the answer is"


def build_pairs(records, data_path, tokenizer, fractions, max_prefix_tokens,
                min_reasoning_sentences=20):
    pairs = []
    for pi, rec in enumerate(records):
        trace_answer = (rec.get("clean_answer") or "").strip()
        all_letters = rec.get("all_letters") or ["A", "B", "C", "D"]
        if len(trace_answer) != 1 or trace_answer not in all_letters:
            continue
        full = torch.tensor(
            list(rec["prompt_token_ids"]) + list(rec["output_token_ids"])
        )
        prompt_len = len(rec["prompt_token_ids"])
        sents = split_tokens_into_sentences(
            full, tokenizer, min_sentence_length=MIN_SENTENCE_LENGTH
        )
        sents = remove_bos_from_sentences(sents)
        sents = chunk_sentences(sents, 1)
        n_sent = len(sents)
        n_prompt = sum(1 for s in sents if s.start < prompt_len)
        n_reason = n_sent - n_prompt
        if n_reason < min_reasoning_sentences:
            continue
        seen_steps = set()
        for frac in fractions:
            step = n_prompt + int(round(frac * n_reason))
            step = max(n_prompt + 10, min(step, n_sent - 2))
            if step in seen_steps:
                continue
            seen_steps.add(step)
            cut = sents[step - 1].end + 1
            if cut > max_prefix_tokens:
                continue
            pairs.append({
                "data_path": data_path,
                "prompt_index": pi,
                "analysis_sentence_step": step,
                "frac": frac,
                "prefix_len": int(cut),
                "remaining_tokens": int(full.shape[-1] - cut),
                "n_sentences": n_sent,
                "n_prompt_sentences": n_prompt,
                "prefix_ids": full[:cut].tolist(),
                "trace_answer": trace_answer,
                "correct_letter": (rec.get("correct_letter") or "").strip(),
                "all_letters": all_letters,
            })
    return pairs


def letter_token_ids(tokenizer, letters):
    out = {}
    for L in letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            out[L] = ids[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_paths", nargs="+", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_prefix_tokens", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)

    all_pairs = {}
    for dp in args.data_paths:
        with open(dp) as f:
            records = json.load(f)
        pairs = build_pairs(records, dp, tokenizer, args.fractions,
                            args.max_prefix_tokens)
        all_pairs[dp] = pairs
        print(f"{dp}: {len(records)} records -> {len(pairs)} valid pairs")

    llm = LLM(model=args.model_name, gpu_memory_utilization=0.9,
              max_model_len=args.max_prefix_tokens + args.horizon + 64,
              seed=args.seed)
    sp = SamplingParams(n=args.n_samples, max_tokens=args.horizon,
                        temperature=args.temperature, top_p=1.0,
                        seed=args.seed)
    sp_probe = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    for dp, pairs in all_pairs.items():
        if not pairs:
            continue
        prompts = [{"prompt_token_ids": p["prefix_ids"]} for p in pairs]
        outs = llm.generate(prompts, sp)

        # ---- probe labeling of terminated samples ----
        probe_prompts, probe_meta = [], []
        per_pair_samples = []
        for pair, out in zip(pairs, outs):
            samples = []
            for comp in out.outputs:
                ids = list(comp.token_ids)
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
                    text_after = tokenizer.decode(ids[think_pos + 1:])
                    s["regex_answer"] = extract_letter(
                        text_after, "".join(pair["all_letters"]))
                    probe_prompts.append({"prompt_token_ids":
                        pair["prefix_ids"] + ids[:think_pos + 1] + suffix_ids})
                    probe_meta.append((len(per_pair_samples), len(samples)))
                samples.append(s)
            per_pair_samples.append(samples)

        if probe_prompts:
            probe_outs = llm.generate(probe_prompts, sp_probe)
            for (pair_i, samp_i), pout in zip(probe_meta, probe_outs):
                pair = pairs[pair_i]
                lids = letter_token_ids(tokenizer, pair["all_letters"])
                lp_dict = pout.outputs[0].logprobs[0]
                raw = {L: lp_dict[tid].logprob
                       for L, tid in lids.items() if tid in lp_dict}
                s = per_pair_samples[pair_i][samp_i]
                if raw:
                    mx = max(raw.values())
                    exp = {L: math.exp(v - mx) for L, v in raw.items()}
                    z = sum(exp.values())
                    s["probe_letter_probs"] = {L: v / z for L, v in exp.items()}
                    s["probe_label"] = max(raw, key=raw.get)

        results = []
        for pair, samples in zip(pairs, per_pair_samples):
            for s in samples:
                if s["terminated"] and s["probe_label"] == pair["trace_answer"]:
                    s["cluster_id"] = 0
                elif s["terminated"]:
                    s["cluster_id"] = 1
                else:
                    s["cluster_id"] = 2
                s["matches_gold"] = (
                    s["probe_label"] == pair["correct_letter"]
                    if s["terminated"] else None
                )
            n_term = sum(s["terminated"] for s in samples)
            n_target = sum(s["cluster_id"] == 0 for s in samples)
            rec = {k: v for k, v in pair.items() if k != "prefix_ids"}
            rec.update({
                "n_samples": len(samples),
                "n_terminated": n_term,
                "n_target_cluster": n_target,
                "f_terminated": n_term / len(samples),
                "samples": samples,
            })
            results.append(rec)
            print(f"  p{pair['prompt_index']:03d} step={pair['analysis_sentence_step']} "
                  f"remain={pair['remaining_tokens']} f_term={rec['f_terminated']:.2f} "
                  f"target={n_target}")
        stem = os.path.splitext(os.path.basename(dp))[0]
        out_path = os.path.join(args.output_dir, f"scan_{stem}.json")
        with open(out_path, "w") as f:
            json.dump({"args": vars(args), "data_path": dp,
                       "results": results}, f)
        print(f"Saved {out_path} ({len(results)} pairs)")


if __name__ == "__main__":
    main()
