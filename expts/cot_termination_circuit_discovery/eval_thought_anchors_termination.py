"""Evaluate a Thought Anchors ranking on the termination task, in one process.

Thought Anchors scores each (later sentence, earlier sentence) pair by
leave-one-out attention suppression: zero all attention to sentence j,
forward the prefix once, record the KL at each downstream sentence
(``expts.thought_anchor_analysis.compute_suppression_scores``).  Here that
score matrix is thresholded at the requested target sparsities (keeping
the HIGHEST-scoring pairs, i.e. ablating the pairs Thought Anchors deems
least important — the baseline convention used in every prior comparison),
the resulting binary mask is installed, and fresh rollouts are generated
and graded exactly as in eval_termination_rollouts.py.

No mask file is written: the score matrix is thresholded in-process and
embedded in the eval JSON for reproducibility.

Usage (one GPU):
    uv run python -m expts.cot_termination_circuit_discovery.eval_thought_anchors_termination \
        --bank_path results/cot_termination/early_2200/banks/gpqa_p012_s34.json \
        --target_sparsities 0.1 \
        --horizon 4096 --n_rollouts 16 \
        --output_dir results/snp_sweep/early_2200/eval
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from utils.utils import set_seed
from utils.masks import (
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_prompt_filter,
    build_combined_filter,
)
from expts.thought_anchor_analysis import compute_suppression_scores
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
    _generate_and_grade,
    _summarize,
    PROBE_SUFFIX_TEXT,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--target_sparsities", nargs="+", type=float,
                    default=[0.1])
    ap.add_argument("--n_rollouts", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sentence_gap", type=int, default=0)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(args.bank_path) as f:
        bank = json.load(f)
    data_path = bank["data_path"]
    prompt_index = int(bank["prompt_index"])
    step = int(bank["analysis_sentence_step"])
    with open(data_path) as f:
        record = json.load(f)[prompt_index]
    trace_answer = (record.get("clean_answer") or "").strip()
    gold_letter = (record.get("correct_letter") or "").strip()
    all_letters = record.get("all_letters") or ["A", "B", "C", "D"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]
    gen_kwargs = dict(suffix_ids=suffix_ids, letter_ids=letter_ids,
                      trace_answer=trace_answer, batch_size=args.batch_size)

    prefix_ids, sentences, _, _, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=0, min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    prefix_ids = prefix_ids.to(device)

    # token -> sentence map over the prefix (needed by the scorer)
    t2s_prefix = torch.full((prefix_len,), -1, dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        t2s_prefix[s.start:s.end + 1] = idx

    # ---- Thought Anchors leave-one-out scores (one forward per sentence)
    print(f"Scoring {num_sents} sentences (prefix {prefix_len} tokens)...")
    scores = compute_suppression_scores(
        model, prefix_ids, sentences, t2s_prefix, args.sentence_gap,
        backend="sdpa",
    )
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

    # ---- filters: prompt sentences are never maskable, matching training
    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode,
                                    device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    prompt_filter = build_prompt_filter(num_prompt_sentences, num_sents,
                                        device=device)
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter)
    valid_filter = ~combined_filter.bool()

    handles = install_clean_sdpa_forward(model)
    token_to_sent = torch.full((prefix_len + args.horizon + 8,), -1,
                               dtype=torch.long, device=device)
    token_to_sent[:prefix_len] = t2s_prefix

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.bank_path))[0]
    for tsp in args.target_sparsities:
        # top-k on the raw Thought Anchors scores (ranking is monotonic, so
        # no log-alpha conversion is needed)
        binary = build_binary_mask(scores_t, "top_k", tsp, valid_filter)
        _install(model, layers, binary, num_heads, token_to_sent,
                 combined_filter)
        set_seed(args.seed)
        rolls = _generate_and_grade(model, tokenizer, prefix_ids,
                                    args.n_rollouts, args.horizon,
                                    args.temperature, device, **gen_kwargs)
        _clear(model, layers)
        out = {
            "method": "thought_anchors",
            "bank_path": args.bank_path,
            "data_path": data_path,
            "prompt_index": prompt_index,
            "analysis_sentence_step": step,
            "target_sparsity": tsp,
            "prefix_len": prefix_len,
            "horizon": args.horizon,
            "temperature": args.temperature,
            "trace_answer": trace_answer,
            "gold_letter": gold_letter,
            "num_frozen_prompt_sentences": num_prompt_sentences,
            "ta_scores": scores,
            "variants": {"thought_anchors": {
                "summary": _summarize(rolls, trace_answer, gold_letter),
                "rollouts": rolls,
            }},
        }
        out_path = os.path.join(
            args.output_dir,
            f"{stem}_thought_anchors_tsp{int(round(tsp * 100)):02d}"
            f"_rollout_eval.json",
        )
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"tsp={tsp}: {out['variants']['thought_anchors']['summary']}")
        print(f"wrote {out_path}")
    remove_handles(handles)


if __name__ == "__main__":
    main()
