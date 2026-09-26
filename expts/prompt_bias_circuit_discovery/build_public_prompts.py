"""Build collection prompts from two public bias datasets, in the same
format as ``build_bias_prompts.py`` so ``collect_bias_rollouts.py`` and the
selection code apply unchanged.

- Anthropic discrim-eval (Tamkin et al. 2023; parquet files under
  data/external/discrim_eval/): 70 decision questions, each filled with a
  demographic profile, explicit (the attribute is stated) or implicit (the
  attribute is carried by a name and pronouns). We keep age 60 and take the
  baseline of the paper (white man) plus every race for both binary genders:
  10 variants per question and fill type. The yes/no question becomes a
  two-option multiple-choice prompt (A) Yes, B) No).
- BBQ (Parrish et al. 2022; parquet files under data/external/bbq/):
  ambiguous-context items only, where the unbiased answer is the
  "unknown" option and the stereotyped answer is ``target_label``. Three
  options, letters A/B/C in the dataset's order.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.build_public_prompts \
        --discrim_output results/prompt_bias/discrim_prompts.json \
        --bbq_output results/prompt_bias/bbq_prompts.json --bbq_per_category 60
"""

from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer

from utils.prompt_utils import MC_INSTRUCTION_PROMPT

RACE_WORDS = {"white": ["white", "caucasian"], "Black": ["black", "african"], "Asian": ["asian"],
              "Hispanic": ["hispanic", "latin"], "Native American": ["native american", "indigenous"]}
GENDER_WORDS = {"male": ["male", " man ", " he ", " his ", " him "], "female": ["female", " woman ", " she ", " her "]}


def _as_list(v):
    if v is None:
        return []
    try:
        return [str(x) for x in list(v)] if not isinstance(v, str) else [v]
    except TypeError:
        return [str(v)]


def fmt(tok, question_with_choices):
    content = MC_INSTRUCTION_PROMPT.format(question=question_with_choices)
    formatted = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True)
    ids = tok(formatted, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == formatted
    return formatted, ids


def build_discrim(tok, races, genders, age):
    out = []
    for fill in ["explicit", "implicit"]:
        df = pd.read_parquet(f"data/external/discrim_eval/{fill}.parquet")
        df = df[(df["age"] == age) & (df["gender"].isin(genders)) & (df["race"].isin(races))]
        for _, r in df.iterrows():
            qid = int(r["decision_question_id"])
            axis_value = f"{r['race']}_{r['gender']}".replace(" ", "")
            qwc = r["filled_template"].strip() + "\n\nChoices:\nA) Yes\nB) No"
            formatted, ids = fmt(tok, qwc)
            regex = RACE_WORDS[r["race"]] + GENDER_WORDS[r["gender"]] + ["race", "ethnic", "gender", "demograph"]
            out.append(dict(
                uid=f"discrim_{fill}_q{qid:02d}_{axis_value}", setting=f"discrim_{fill}", profile_id=qid,
                axis="race_gender", value=axis_value, race=r["race"], gender=r["gender"], age=age,
                fill_type=fill, cue_sentence="", cue_regex=regex,
                question=r["filled_template"].strip(), question_with_choices=qwc,
                prompt=formatted, prompt_token_ids=ids, all_letters=["A", "B"], all_answers=["Yes", "No"],
                dataset_name=f"discrim_eval_{fill}", dataset_type="multiple choice",
            ))
    return out


def build_bbq(tok, per_category, seed):
    rng = random.Random(seed)
    out = []
    for cat in ["race_ethnicity", "gender_identity", "nationality", "ses", "sexual_orientation", "disability_status"]:
        path = f"data/external/bbq/{cat}.parquet"
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            continue
        df = pd.read_parquet(path)
        df = df[df["context_condition"] == "ambig"]
        idx = list(df.index)
        rng.shuffle(idx)
        for i in idx[:per_category]:
            r = df.loc[i]
            answers = [r["ans0"], r["ans1"], r["ans2"]]
            qwc = f"{r['context'].strip()} {r['question'].strip()}\n\nChoices:\n" + "\n".join(
                f"{l}) {a}" for l, a in zip("ABC", answers))
            formatted, ids = fmt(tok, qwc)
            unknown_idx = int(r["answer_label"])
            target_idx = int(r["target_label"])
            out.append(dict(
                uid=f"bbq_{cat}_{int(r['example_id']):05d}", setting=f"bbq_{cat}", profile_id=int(r["example_id"]),
                axis=cat, value=str(r["question_polarity"]), cue_sentence="", cue_regex=[],
                question=f"{r['context'].strip()} {r['question'].strip()}", question_with_choices=qwc,
                prompt=formatted, prompt_token_ids=ids, all_letters=["A", "B", "C"], all_answers=answers,
                unknown_letter="ABC"[unknown_idx], stereotyped_letter="ABC"[target_idx],
                question_polarity=str(r["question_polarity"]),
                stereotyped_groups=_as_list(r["additional_metadata"].get("stereotyped_groups")),
                dataset_name=f"bbq_{cat}", dataset_type="multiple choice",
            ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--discrim_output", required=True)
    ap.add_argument("--bbq_output", required=True)
    ap.add_argument("--races", nargs="+", default=["white", "Black", "Asian", "Hispanic", "Native American"])
    ap.add_argument("--genders", nargs="+", default=["male", "female"])
    ap.add_argument("--age", type=float, default=60.0)
    ap.add_argument("--bbq_per_category", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    d = build_discrim(tok, args.races, args.genders, args.age)
    b = build_bbq(tok, args.bbq_per_category, args.seed)
    for path, rows in [(args.discrim_output, d), (args.bbq_output, b)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"Wrote {len(rows)} prompts -> {path}")
    print(d[0]["prompt"][-700:])
    print(b[0]["prompt"][-500:])


if __name__ == "__main__":
    main()
