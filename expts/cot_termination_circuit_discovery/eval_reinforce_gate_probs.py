"""Rollout-evaluate the REINFORCE pilot's learned gates, in one process.

The REINFORCE pilot (reinforce_termination.py) saved its result as
``final_gate_probs.json`` — a sentence-pair matrix of gate keep
probabilities — rather than a mask file in the NodeMask schema, so
eval_termination_rollouts.py cannot read it.  This script closes that gap
without writing any mask file: it loads the keep-probability matrix, keeps
the top (1 - target_sparsity) fraction of pairs (matched-target
convention, ranking directly on the probabilities), installs the binary
mask, and generates and grades rollouts with exactly the machinery and
seed of eval_termination_rollouts.py, so the output JSON is directly
comparable with the off-policy learnt masks' evaluations on the same
prompt.

Usage (one GPU):
    uv run python -m expts.cot_termination_circuit_discovery.eval_reinforce_gate_probs \
        --gate_probs results/snp_sweep/gradient_check/reinforce_aqua_p080_s57/final_gate_probs.json \
        --n_rollouts 16 \
        --output results/snp_sweep/gradient_check/reinforce_aqua_p080_s57/rollout_eval.json
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
    _generate_and_grade,
    _summarize,
    PROBE_SUFFIX_TEXT,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_probs", required=True,
                    help="final_gate_probs.json from reinforce_termination.py")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_rollouts", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sentence_gap", type=int, default=0)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_sparsity", type=float, default=None,
                    help="Defaults to the pilot's own target_sparsity.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(args.gate_probs) as f:
        gp = json.load(f)
    bank_path = gp["bank_path"]
    target_sparsity = (args.target_sparsity
                       if args.target_sparsity is not None
                       else float(gp["target_sparsity"]))
    with open(bank_path) as f:
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
    keep_probs = torch.tensor(gp["edge_keep_probs"], dtype=torch.float32)
    if keep_probs.shape != (num_sents, num_sents):
        raise ValueError(
            f"edge_keep_probs shape {tuple(keep_probs.shape)} != "
            f"({num_sents}, {num_sents}) sentences of this prefix"
        )

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    prefix_ids = prefix_ids.to(device)
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

    # rank the keep probabilities directly (monotonic, no conversion needed)
    binary = build_binary_mask(keep_probs.to(device), "top_k",
                               target_sparsity, valid_filter)

    token_to_sent = torch.full((prefix_len + args.horizon + 8,), -1,
                               dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        token_to_sent[s.start:s.end + 1] = idx

    _install(model, layers, binary, num_heads, token_to_sent, combined_filter)
    set_seed(args.seed)
    rolls = _generate_and_grade(model, tokenizer, prefix_ids, args.n_rollouts,
                                args.horizon, args.temperature, device,
                                **gen_kwargs)
    _clear(model, layers)
    remove_handles(handles)

    out = {
        "method": "reinforce_gate_probs",
        "gate_probs_path": args.gate_probs,
        "bank_path": bank_path,
        "data_path": data_path,
        "prompt_index": prompt_index,
        "analysis_sentence_step": step,
        "target_sparsity": target_sparsity,
        "final_keep_frac_of_gates": float(keep_probs.mean().item()),
        "prefix_len": prefix_len,
        "horizon": args.horizon,
        "temperature": args.temperature,
        "trace_answer": trace_answer,
        "gold_letter": gold_letter,
        "variants": {"reinforce": {
            "summary": _summarize(rolls, trace_answer, gold_letter),
            "rollouts": rolls,
        }},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print("reinforce:", out["variants"]["reinforce"]["summary"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
