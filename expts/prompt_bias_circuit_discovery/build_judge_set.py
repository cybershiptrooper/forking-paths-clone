"""Build a judge set: traces with a cued (biased) answer and matched traces
with the reference answer, in collection format plus a selection list.

Modes
- resume: cue variants from --selection (bias_selection_passing.json);
  positives = terminated rollouts of the variant with the cued answer
  (silent ones first); negatives = rollouts of the neutral variant of the
  same profile with the reference answer.
- discrim: candidates from the public-collection summary (|delta| >= min_delta);
  positives = rollouts of the candidate variant with the shifted answer;
  negatives = rollouts of the white-man baseline of the same question and
  fill type with the baseline answer.
- bbq: items with at least --bbq_min_stereotyped stereotyped answers;
  positives = stereotyped-answer rollouts; negatives = unknown-answer
  rollouts of the same item.

Every record gets an analysis point at --reasoning_frac of its reasoning
sentences (at least --min_reasoning_sentences), ``target_letter`` = the
letter the trace did not give (positives: the reference answer; negatives:
the cued answer), and ``is_positive``.
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


def reasoning(text):
    return text.split("</think>")[0]


def mentions(text, words):
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return [w for w in words if w.lower() in t]


def layout(tok, prompt_ids, out_ids, frac, min_reason, k_after):
    full = torch.tensor(prompt_ids + out_ids)
    sents = split_tokens_into_sentences(full, tok, 10)
    n_prompt = sum(1 for s in sents if s.start < len(prompt_ids))
    think_end = next((i for i, s in enumerate(sents) if "</think>" in tok.decode(full[s.start:s.end + 1])), len(sents))
    n_reason = think_end - n_prompt
    if n_reason < min_reason:
        return None
    step = n_prompt + max(min_reason - k_after, int(round(frac * n_reason)))
    step = min(step, think_end - k_after)
    return n_prompt, n_reason, int(step)


def make_record(v, r, others, reference_letter, cued_letter, is_positive, extra):
    return dict(
        prompt=v["prompt"], prompt_token_ids=v["prompt_token_ids"], question=v["question"],
        question_with_choices=v["question_with_choices"], output_token_ids=r["token_ids"], output_text=r["text"],
        clean_answer=r["answer"], raw_answer=r.get("raw_answer", ""), finish_reason=r["finish_reason"],
        alternate_texts=[o["text"] for o in others][:4], alternate_answers=[o["answer"] for o in others],
        alternate_finish_reasons=[o["finish_reason"] for o in others],
        all_sampled_answers=[o["answer"] for o in v["rollouts"]], all_letters=v["all_letters"], all_answers=v["all_answers"],
        correct_letter=reference_letter, correct_answer=v["all_answers"][v["all_letters"].index(reference_letter)],
        dataset_name=v["dataset_name"], dataset_type="multiple choice", base_answer_type="stored",
        uid=v["uid"], cue_regex=v.get("cue_regex", []), cue_sentence=v.get("cue_sentence", ""),
        reference_letter=reference_letter, cued_letter=cued_letter, is_positive=is_positive,
        target_letter=(reference_letter if is_positive else cued_letter), **extra,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["resume", "discrim", "bbq"], required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--selection", default=None, help="resume: bias_selection_passing.json")
    ap.add_argument("--summary", default=None, help="discrim/bbq: public_collection_summary.json")
    ap.add_argument("--min_delta", type=float, default=0.25)
    ap.add_argument("--bbq_min_stereotyped", type=int, default=2)
    ap.add_argument("--k_pos", type=int, default=2)
    ap.add_argument("--k_neg", type=int, default=2)
    ap.add_argument("--neg_source", choices=["baseline", "same"], default="baseline",
                    help="baseline: negatives from the neutral / white-man variant; same: "
                    "negatives from the cue variant itself, rollouts that gave the reference answer "
                    "(same prompt as the positives, so the judge cannot read the label off the prompt).")
    ap.add_argument("--max_positives", type=int, default=40)
    ap.add_argument("--reasoning_frac", type=float, default=0.6)
    ap.add_argument("--min_reasoning_sentences", type=int, default=12)
    ap.add_argument("--sentences_after_prefix", type=int, default=5)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--data_output", required=True)
    ap.add_argument("--selection_output", required=True)
    ap.add_argument("--tag_prefix", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    V = json.load(open(args.rollouts))
    by_uid = {v["uid"]: v for v in V}

    pairs = []  # (cue variant, baseline variant, cued_letter, reference_letter, extra)
    if args.mode == "resume":
        sel = json.load(open(args.selection))
        for s in sel:
            v = by_uid[s["tag"]]
            base = by_uid[f"{v['setting']}_p{v['profile_id']:02d}_neutral_none"]
            pairs.append((v, base, s["biased_letter"], s["target_letter"], dict(axis=v["axis"], value=v["value"])))
    elif args.mode == "discrim":
        summ = json.load(open(args.summary))["discrim"]
        cands = [dict(c, fill=fill) for fill in ["explicit", "implicit"] for c in summ[fill]["candidates"]]
        cands.sort(key=lambda c: -abs(c["delta"]))
        for c in cands:
            v = by_uid[c["uid"]]
            base = by_uid[f"discrim_{c['fill']}_q{c['qid']:02d}_white_male"]
            cued = "A" if c["delta"] > 0 else "B"
            pairs.append((v, base, cued, "B" if cued == "A" else "A",
                          dict(axis="race_gender", value=f"{c['race']}_{c['gender']}", fill_type=c["fill"], delta=c["delta"])))
    else:
        for v in V:
            n_st = sum(r["answer"] == v["stereotyped_letter"] for r in v["rollouts"])
            n_un = sum(r["answer"] == v["unknown_letter"] for r in v["rollouts"])
            if n_st >= args.bbq_min_stereotyped and n_un >= 1:
                pairs.append((v, v, v["stereotyped_letter"], v["unknown_letter"],
                              dict(axis=v["axis"], value=v["question_polarity"], stereotyped_groups=v.get("stereotyped_groups", []))))

    records, sel_out = [], []
    n_pos = 0
    for v, base, cued, ref, extra in pairs:
        if n_pos >= args.max_positives:
            break
        pos = [r for r in v["rollouts"] if r["answer"] == cued and r["terminated"] and r["finish_reason"] == "stop"]
        neg_src = v if args.neg_source == "same" else base
        neg = [r for r in neg_src["rollouts"] if r["answer"] == ref and r["terminated"] and r["finish_reason"] == "stop"]
        words = v.get("cue_regex", []) or [g for g in extra.get("stereotyped_groups", [])]
        pos.sort(key=lambda r: (len(mentions(reasoning(r["text"]), words)) > 0, len(r["token_ids"])))
        neg.sort(key=lambda r: len(r["token_ids"]))
        for group, rs, is_pos, src in [("pos", pos[:args.k_pos], True, v), ("neg", neg[:args.k_neg], False, neg_src)]:
            for j, r in enumerate(rs):
                lay = layout(tok, src["prompt_token_ids"], r["token_ids"], args.reasoning_frac,
                             args.min_reasoning_sentences, args.sentences_after_prefix)
                if lay is None:
                    continue
                n_prompt, n_reason, step = lay
                rec = make_record(src, r, [o for o in src["rollouts"] if o is not r], ref, cued, is_pos,
                                  dict(extra, mentions_cue=mentions(reasoning(r["text"]), words), source_uid=src["uid"],
                                       neg_source=args.neg_source))
                records.append(rec)
                tag = f"{args.tag_prefix}_{len(records) - 1:03d}_{'pos' if is_pos else 'neg'}"
                sel_out.append(dict(tag=tag, data_path=args.data_output, prompt_index=len(records) - 1,
                                    analysis_sentence_step=step, sentences_after_prefix=args.sentences_after_prefix,
                                    all_letters=src["all_letters"], target_letter=rec["target_letter"],
                                    biased_letter=cued, reference_letter=ref, is_positive=is_pos,
                                    verbalizes_cue=len(rec["mentions_cue"]) > 0, cue_mentions=rec["mentions_cue"],
                                    n_prompt_sentences=n_prompt, n_reasoning_sentences=n_reason,
                                    trace_tokens=len(r["token_ids"]), source_uid=src["uid"], pair_uid=v["uid"], **extra))
                if is_pos:
                    n_pos += 1
    os.makedirs(os.path.dirname(args.data_output) or ".", exist_ok=True)
    json.dump(records, open(args.data_output, "w"))
    json.dump(sel_out, open(args.selection_output, "w"), indent=1)
    c = collections.Counter((s["is_positive"], s["verbalizes_cue"]) for s in sel_out)
    print(f"{len(records)} records -> {args.data_output}; (positive, mentions cue) counts: {dict(c)}")


if __name__ == "__main__":
    main()
