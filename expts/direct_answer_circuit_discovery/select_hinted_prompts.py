"""Final prompt selection for the hint-removal experiment (one GPU, HF).

For every record in the hinted collection, picks the analysis sentence
step, runs the direct-answer probe at (a) an early prefix — a few
sentences into the reasoning, before any substantive deliberation — and
(b) the analysis point, and applies the selection criteria from
notes/termination_and_hint_masks.md §2.2:

    P_early(hint letter)   < --early_saturation_max   (copying guard)
    P_analysis(hint letter) > --analysis_p_hint_min    (switch visible at probe)

Also records whether the trace verbalizes the hint before the analysis
point (regex on the decoded reasoning prefix).

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.select_hinted_prompts \
        --data_path data/collection/qwen3_8b/hinted_gpqa_aqua.json \
        --output results/hint_removal/selection.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer

from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.direct_answer_circuit_discovery.probe import (
    build_answer_probe, answer_probs_from_logits, DEFAULT_SUFFIX,
)

HINT_REGEX = re.compile(r"professor|stanford", re.IGNORECASE)


def probe_at(model, tokenizer, probe, data_path, prompt_index, step,
             sentences_after_prefix, device):
    prefix_ids, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=sentences_after_prefix,
        min_sentence_length=10, sentence_chunk=1,
    )
    prefix_len = prefix_ids.shape[-1]
    cont = probe.make_continuation(device)
    full = torch.cat([prefix_ids.to(device), cont], dim=-1)
    with torch.no_grad():
        logits = model(full).logits
    p = answer_probs_from_logits(logits, probe, prefix_len).cpu()
    del logits, full
    return p, prefix_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--max_step", type=int, default=50)
    ap.add_argument("--reasoning_frac", type=float, default=0.6,
                    help="Analysis step = min(max_step, n_prompt + "
                    "frac * n_reasoning sentences).")
    ap.add_argument("--min_reasoning_sentences", type=int, default=25)
    ap.add_argument("--sentences_after_prefix", type=int, default=5)
    ap.add_argument("--early_sentences", type=int, default=3,
                    help="Early probe at n_prompt + this many sentences.")
    ap.add_argument("--early_saturation_max", type=float, default=0.9)
    ap.add_argument("--analysis_p_hint_min", type=float, default=0.5)
    ap.add_argument("--max_prompts", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.data_path) as f:
        records = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device

    rows = []
    for pi, rec in enumerate(records):
        full = torch.tensor(
            list(rec["prompt_token_ids"]) + list(rec["output_token_ids"])
        )
        from utils.cot_analysis import (
            split_tokens_into_sentences, remove_bos_from_sentences,
            chunk_sentences,
        )
        sents = split_tokens_into_sentences(full, tokenizer,
                                            min_sentence_length=10)
        sents = remove_bos_from_sentences(sents)
        sents = chunk_sentences(sents, 1)
        prompt_len = len(rec["prompt_token_ids"])
        n_sent = len(sents)
        n_prompt = sum(1 for s in sents if s.start < prompt_len)
        n_reason = n_sent - n_prompt
        if n_reason < args.min_reasoning_sentences:
            print(f"p{pi:02d}: only {n_reason} reasoning sentences, skip")
            continue
        step = min(args.max_step,
                   n_prompt + int(round(args.reasoning_frac * n_reason)))
        step = max(step, n_prompt + 15)
        if step + args.sentences_after_prefix >= n_sent:
            step = n_sent - args.sentences_after_prefix - 1
        if step <= n_prompt + 5:
            print(f"p{pi:02d}: trace too short after adjustment, skip")
            continue

        letters = [" " + l for l in rec["all_letters"]]
        probe = build_answer_probe(tokenizer, suffix=DEFAULT_SUFFIX,
                                   answer_letters=letters)
        hint_idx = rec["all_letters"].index(rec["hint_letter"])
        target_letter = rec["control_modal_answer"]
        if target_letter not in rec["all_letters"]:
            print(f"p{pi:02d}: control modal {target_letter!r} not a letter, skip")
            continue
        target_idx = rec["all_letters"].index(target_letter)

        early_step = n_prompt + args.early_sentences
        p_early, _ = probe_at(model, tokenizer, probe, args.data_path, pi,
                              early_step, 0, device)
        p_analysis, prefix_len = probe_at(
            model, tokenizer, probe, args.data_path, pi, step,
            args.sentences_after_prefix, device)

        cut = sents[step - 1].end + 1
        reasoning_text = tokenizer.decode(full[prompt_len:cut])
        m = HINT_REGEX.search(reasoning_text)
        verbalizes = bool(m)

        row = {
            "prompt_index": pi,
            "source_data_path": rec["source_data_path"],
            "source_prompt_index": rec["source_prompt_index"],
            "analysis_sentence_step": step,
            "n_sentences": n_sent,
            "n_prompt_sentences": n_prompt,
            "prefix_len_analysis": prefix_len,
            "hint_letter": rec["hint_letter"],
            "target_letter": target_letter,
            "correct_letter": rec["correct_letter"],
            "all_letters": rec["all_letters"],
            "switch_rate": rec["switch_rate"],
            "control_accuracy": rec["control_accuracy"],
            "p_early": p_early.tolist(),
            "p_early_hint": float(p_early[hint_idx]),
            "p_analysis": p_analysis.tolist(),
            "p_analysis_hint": float(p_analysis[hint_idx]),
            "p_analysis_target": float(p_analysis[target_idx]),
            "verbalizes_hint": verbalizes,
            "hint_mention": (
                reasoning_text[max(0, m.start() - 80):m.end() + 80]
                if m else None
            ),
        }
        row["passes"] = (
            row["p_early_hint"] < args.early_saturation_max
            and row["p_analysis_hint"] > args.analysis_p_hint_min
            and target_letter != rec["hint_letter"]
        )
        rows.append(row)
        print(f"p{pi:02d} step={step} P_early(hint)={row['p_early_hint']:.2f} "
              f"P_analysis(hint)={row['p_analysis_hint']:.2f} "
              f"verbalizes={verbalizes} passes={row['passes']}")

    passing = [r for r in rows if r["passes"]]
    # balance across source datasets
    by_src = {}
    for r in passing:
        by_src.setdefault(r["source_data_path"], []).append(r)
    selected, idx = [], 0
    while len(selected) < args.max_prompts:
        advanced = False
        for src in sorted(by_src):
            lst = by_src[src]
            if idx < len(lst) and len(selected) < args.max_prompts:
                selected.append(lst[idx])
                advanced = True
        if not advanced:
            break
        idx += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "all_rows": rows,
                   "selected": selected}, f, indent=2)
    nv = sum(r["verbalizes_hint"] for r in selected)
    print(f"Selected {len(selected)} prompts "
          f"({nv} verbalize the hint, {len(selected) - nv} silent) "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
