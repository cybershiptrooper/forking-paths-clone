"""Pick the prompts whose decision changes with the demographic cue, store
one biased trace per prompt in collection format, and probe it.

Steps
1. For every (setting, profile) read P(A) of the neutral variant (cue
   sentence says the optional section was left blank) and of every cue
   variant from the raw rollouts.
2. A cue variant is a candidate when its modal answer differs from the
   neutral modal answer by at least --min_delta in P(A) and at least
   --min_biased_rate of its rollouts give the biased answer.
3. For each candidate pick a stored trace: a terminated rollout with the
   biased answer, preferring one whose reasoning (text before </think>)
   does not match the cue's words (``verbalizes_cue`` is recorded).
4. Write a collection-format record per candidate (the same fields the
   GPQA collection has, plus the cue metadata) and, on GPU, probe
   P(letter) at an early prefix and at the analysis point
   (``--reasoning_frac`` of the reasoning sentences). Keep candidates
   whose analysis-point probe gives P(biased) >= --p_biased_min.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.select_bias_prompts \
        --rollouts results/prompt_bias/rollouts_raw.json \
        --data_output data/collection/qwen3_8b/bias_decisions.json \
        --selection_output results/prompt_bias/bias_selection.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

import torch
from transformers import AutoTokenizer

from utils.cot_analysis import split_tokens_into_sentences


def rates(rollouts):
    ans = [r["answer"] for r in rollouts]
    c = collections.Counter(ans)
    n = len(ans)
    return {k: v / n for k, v in c.items()}, c


def reasoning_text(text):
    return text.split("</think>")[0]


def mentions(text, words):
    t = " " + text.lower() + " "
    hits = [w for w in words if w in t]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--min_delta", type=float, default=0.25)
    ap.add_argument("--min_biased_rate", type=float, default=0.5)
    ap.add_argument("--max_neutral_biased_rate", type=float, default=0.4)
    ap.add_argument("--reasoning_frac", type=float, default=0.6)
    ap.add_argument("--min_reasoning_sentences", type=int, default=12)
    ap.add_argument("--sentences_after_prefix", type=int, default=5)
    ap.add_argument("--early_sentences", type=int, default=3)
    ap.add_argument("--p_biased_min", type=float, default=0.5)
    ap.add_argument("--no_probe", action="store_true")
    ap.add_argument("--data_output", required=True)
    ap.add_argument("--selection_output", required=True)
    args = ap.parse_args()

    with open(args.rollouts) as f:
        variants = json.load(f)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    by_key = {(v["setting"], v["profile_id"], v["axis"], v["value"]): v for v in variants}

    # ---- 1./2. candidates -------------------------------------------------
    table, candidates = [], []
    for v in variants:
        neutral = by_key[(v["setting"], v["profile_id"], "neutral", "none")]
        pr_n, _ = rates(neutral["rollouts"])
        pr_v, cnt_v = rates(v["rollouts"])
        row = dict(uid=v["uid"], setting=v["setting"], profile_id=v["profile_id"],
                   axis=v["axis"], value=v["value"], p_A=pr_v.get("A", 0.0),
                   p_A_neutral=pr_n.get("A", 0.0), delta_p_A=pr_v.get("A", 0.0) - pr_n.get("A", 0.0))
        table.append(row)
        if v["axis"] == "neutral":
            continue
        neutral_modal = max(pr_n, key=pr_n.get)
        biased = "B" if neutral_modal == "A" else "A"
        if (abs(row["delta_p_A"]) >= args.min_delta
                and pr_v.get(biased, 0.0) >= args.min_biased_rate
                and pr_n.get(biased, 0.0) <= args.max_neutral_biased_rate):
            candidates.append((v, neutral, biased, neutral_modal, row))
    print(f"{len(candidates)} candidate variants of {len(variants)}")

    # ---- 3. stored trace + collection record -------------------------------
    records, sel = [], []
    for v, neutral, biased, neutral_modal, row in candidates:
        pool = [r for r in v["rollouts"]
                if r["answer"] == biased and r["terminated"] and r["finish_reason"] == "stop"]
        if not pool:
            continue
        scored = []
        for r in pool:
            hits = mentions(reasoning_text(r["text"]), v["cue_regex"])
            scored.append((len(hits) > 0, len(r["token_ids"]), r, hits))
        scored.sort(key=lambda x: (x[0], x[1]))
        verbal, _, base, hits = scored[0]
        n_silent = sum(1 for s in scored if not s[0])
        others = [r for r in v["rollouts"] if r is not base]
        # sentence layout of prompt + trace
        full = torch.tensor(v["prompt_token_ids"] + base["token_ids"])
        sents = split_tokens_into_sentences(full, tok, 10)
        n_prompt = sum(1 for s in sents if s.start < len(v["prompt_token_ids"]))
        think_end = next((i for i, s in enumerate(sents)
                          if "</think>" in tok.decode(full[s.start:s.end + 1])), len(sents))
        n_reason = think_end - n_prompt
        if n_reason < args.min_reasoning_sentences:
            print(f"skip {v['uid']}: only {n_reason} reasoning sentences")
            continue
        step = n_prompt + max(args.min_reasoning_sentences - 5,
                              int(round(args.reasoning_frac * n_reason)))
        step = min(step, think_end - args.sentences_after_prefix)
        cue_sentence_idx = next((i for i in range(n_prompt)
                                 if v["cue_sentence"].strip()[:40] in tok.decode(full[sents[i].start:sents[i].end + 1])), None)
        rec = dict(
            prompt=v["prompt"], prompt_token_ids=v["prompt_token_ids"],
            question=v["question"], question_with_choices=v["question_with_choices"],
            output_token_ids=base["token_ids"], output_text=base["text"],
            clean_answer=base["answer"], raw_answer=base["raw_answer"],
            finish_reason=base["finish_reason"],
            alternate_texts=[r["text"] for r in others],
            alternate_answers=[r["answer"] for r in others],
            alternate_finish_reasons=[r["finish_reason"] for r in others],
            all_sampled_answers=[r["answer"] for r in v["rollouts"]],
            all_letters=v["all_letters"], all_answers=v["all_answers"],
            correct_letter=neutral_modal, correct_answer=v["all_answers"][v["all_letters"].index(neutral_modal)],
            dataset_name=v["dataset_name"], dataset_type=v["dataset_type"],
            base_answer_type="stored",
            # cue metadata
            uid=v["uid"], setting=v["setting"], profile_id=v["profile_id"],
            axis=v["axis"], value=v["value"], cue_sentence=v["cue_sentence"], cue_regex=v["cue_regex"],
            biased_letter=biased, target_letter=neutral_modal,
            p_A=row["p_A"], p_A_neutral=row["p_A_neutral"],
            neutral_answers=[r["answer"] for r in neutral["rollouts"]],
            verbalizes_cue=bool(verbal), cue_mentions=hits, n_silent_biased_rollouts=n_silent,
            n_biased_rollouts=len(pool),
        )
        records.append(rec)
        sel.append(dict(
            tag=v["uid"], data_path=args.data_output, prompt_index=len(records) - 1,
            analysis_sentence_step=int(step), sentences_after_prefix=args.sentences_after_prefix,
            all_letters=v["all_letters"], target_letter=neutral_modal, biased_letter=biased,
            verbalizes_cue=bool(verbal), cue_mentions=hits, n_prompt_sentences=n_prompt,
            cue_sentence_idx=cue_sentence_idx, n_reasoning_sentences=n_reason,
            setting=v["setting"], profile_id=v["profile_id"], axis=v["axis"], value=v["value"],
            p_A=row["p_A"], p_A_neutral=row["p_A_neutral"], delta_p_A=row["delta_p_A"],
            n_silent_biased_rollouts=n_silent, n_biased_rollouts=len(pool),
            trace_tokens=len(base["token_ids"]),
        ))
    os.makedirs(os.path.dirname(args.data_output) or ".", exist_ok=True)
    with open(args.data_output, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} records -> {args.data_output}")

    # ---- 4. probe ----------------------------------------------------------
    if not args.no_probe and sel:
        from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
        from expts.direct_answer_circuit_discovery.probe import (
            build_answer_probe, answer_probs_from_logits, DEFAULT_SUFFIX)
        model, _ = load_model_eager(args.model_name, device="cuda")
        dev = next(model.parameters()).device
        for s in sel:
            letters = [" " + l for l in s["all_letters"]]
            probe = build_answer_probe(tok, suffix=DEFAULT_SUFFIX, answer_letters=letters)
            bi = s["all_letters"].index(s["biased_letter"])
            ti = s["all_letters"].index(s["target_letter"])
            for name, step in [("early", s["n_prompt_sentences"] + args.early_sentences),
                               ("analysis", s["analysis_sentence_step"])]:
                prefix_ids, _, _, _, _, _ = _build_prefix(
                    tokenizer=tok, prompt=None, data_path=args.data_output,
                    prompt_index=s["prompt_index"], base_answer_type="stored",
                    analysis_timestep=None, analysis_sentence_step=step,
                    sentences_after_prefix=args.sentences_after_prefix if name == "analysis" else 0,
                    min_sentence_length=10, sentence_chunk=1)
                full = torch.cat([prefix_ids.to(dev), probe.make_continuation(dev)], dim=-1)
                with torch.no_grad():
                    logits = model(full).logits
                p = answer_probs_from_logits(logits, probe, prefix_ids.shape[-1]).cpu().tolist()
                s[f"p_{name}"] = p
                s[f"p_{name}_biased"] = p[bi]
                s[f"p_{name}_target"] = p[ti]
                if name == "analysis":
                    s["prefix_len_analysis"] = int(prefix_ids.shape[-1])
            s["passes"] = s["p_analysis_biased"] >= args.p_biased_min
            print(f"{s['tag']:34s} verbal={s['verbalizes_cue']!s:5s} dP(A)={s['delta_p_A']:+.2f} "
                  f"P_early(biased)={s['p_early_biased']:.2f} P_analysis(biased)={s['p_analysis_biased']:.2f} "
                  f"step={s['analysis_sentence_step']} passes={s['passes']}")
    os.makedirs(os.path.dirname(args.selection_output) or ".", exist_ok=True)
    with open(args.selection_output, "w") as f:
        json.dump({"args": vars(args), "table": table, "selected": sel}, f, indent=1)
    print(f"Wrote selection -> {args.selection_output}")


if __name__ == "__main__":
    main()
