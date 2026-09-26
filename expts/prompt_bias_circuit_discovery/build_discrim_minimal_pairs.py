"""Minimal-pair prompts from Anthropic's discrimination evaluation.

The public dataset's *implicit* fills are separate paraphrases per variant
(the Native-American-male fill of question 9 drops the credit score and the
down payment that the white-male fill states), so a difference in the
answer between two implicit fills is not attributable to the name. This
builder derives every variant from one base text per (question, gender):
the dataset's explicit white-male / white-female fill at age 60.

- explicit: the race word is substituted in place ("60-year-old white male"
  -> "60-year-old Black male"); the pair differs only in that word.
- implicit: the race word is deleted and " named <NAME>" is appended to
  the noun phrase that contained it ("a 60-year-old male mechanic named
  DeAndre Jackson"); the pair differs only in the name. Two names per
  race and gender. The white names are the implicit baseline.

Every variant is verified: the word-level diff against its base must
consist of the race word, the name and "named" only, otherwise the
question is dropped. Gender is kept in the text ("male"/"female") and
gender comparisons use the dataset's own male/female bases, which differ
in pronouns.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.build_discrim_minimal_pairs \
        --output results/prompt_bias_v2/discrim_mp_prompts.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re

import pandas as pd
import spacy
from transformers import AutoTokenizer

from utils.prompt_utils import MC_INSTRUCTION_PROMPT

RACES = ["white", "Black", "Asian", "Hispanic", "Native American"]
NAMES = {
    ("white", "male"): ["Tyler Lewis", "Connor Walsh"],
    ("white", "female"): ["Amanda Taylor", "Emily Schmidt"],
    ("Black", "male"): ["DeAndre Jackson", "DeShawn Washington"],
    ("Black", "female"): ["Lakisha Robinson", "Ebony Booker"],
    ("Asian", "male"): ["Chang Chen", "Wei Zhang"],
    ("Asian", "female"): ["Mei Yu", "Ling Wang"],
    ("Hispanic", "male"): ["Juan Gomez", "Luis Hernandez"],
    ("Hispanic", "female"): ["Sofia Ramos", "Maria Rodriguez"],
    ("Native American", "male"): ["Tawa Altaha", "Takoda Begay"],
    ("Native American", "female"): ["Nahimana Dosela", "Aiyana Yazzie"],
}
RACE_WORDS = {"white": ["white", "caucasian"], "Black": ["black", "african"], "Asian": ["asian"],
              "Hispanic": ["hispanic", "latin"], "Native American": ["native american", "indigenous"]}
GENDER_WORDS = {"male": ["male", " man ", " he ", " his ", " him "], "female": ["female", " woman ", " she ", " her "]}


def fmt(tok, question_with_choices):
    content = MC_INSTRUCTION_PROMPT.format(question=question_with_choices)
    formatted = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True)
    ids = tok(formatted, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == formatted
    return formatted, ids


def normalise(text):
    """Drop the dataset's "a(n)" hack (the phrase always continues with
    "60", so "a" is right), collapse double spaces, and put the race word
    before the gender word ("female white" -> "white female", the order the
    male fills use) so the explicit variants read naturally."""
    text = re.sub(r"\b[Aa]\(n\) ", lambda m: m.group(0)[0] + " ", text)
    text = re.sub(r"[ ]{2,}", " ", text.strip())
    text = re.sub(r"\b(female|male) (white|White)\b", r"\2 \1", text)
    return text


def word_diff(a, b):
    return [x for x in difflib.ndiff(a.split(), b.split()) if x[0] in "+-"]


def allowed_diff(diff, allowed_words):
    """True when, after stripping punctuation and cancelling words that
    merely moved (e.g. "trainer." -> "trainer" + "Lewis."), every
    remaining changed word is in ``allowed_words``."""
    strip = lambda w: w.strip(".,;:?!'\"()")
    removed = [strip(x[2:]) for x in diff if x[0] == "-"]
    added = [strip(x[2:]) for x in diff if x[0] == "+"]
    for w in list(removed):
        if w in added:
            removed.remove(w); added.remove(w)
    return all(w == "" or w in allowed_words for w in removed + added)


def implicit_text(nlp, base, name):
    """Delete the race word and append ' named NAME' to its noun phrase."""
    doc = nlp(base)
    tok_white = [t for t in doc if t.text.lower() == "white"]
    if len(tok_white) != 1:
        return None, "race word count != 1"
    t = tok_white[0]
    chunk = next((c for c in doc.noun_chunks if c.start <= t.i < c.end), None)
    if chunk is None:
        return None, "no noun chunk"
    if any(x.tag_ == "POS" for x in chunk):
        return None, "possessive noun phrase"
    # the chunk must end before a verb/relative clause, otherwise spaCy has
    # cut the noun phrase short (e.g. "a 60-year-old white male" + "mechanic")
    end = chunk.end
    while end < len(doc) and doc[end].pos_ in ("NOUN", "PROPN") and not doc[end].is_punct:
        end += 1
    text = base[:t.idx] + base[t.idx + len(t.text):]
    text = re.sub(r"[ ]{2,}", " ", text)
    shift = len(t.text) + (1 if base[t.idx + len(t.text):t.idx + len(t.text) + 1] == " " else 0)
    insert_at = doc[end - 1].idx + len(doc[end - 1].text) - shift
    text = text[:insert_at] + f" named {name}" + text[insert_at:]
    return text, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--age", type=float, default=60.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet("data/external/discrim_eval/explicit.parquet")
    df = df[df["age"] == args.age]

    out, dropped = [], []
    for qid in sorted(df["decision_question_id"].unique()):
        variants = []
        ok = True
        for gender in ["male", "female"]:
            sel = df[(df.decision_question_id == qid) & (df.race == "white") & (df.gender == gender)]
            if len(sel) != 1:
                ok = False; dropped.append((int(qid), f"no white {gender} base")); break
            base = normalise(sel.filled_template.iloc[0])
            if len(re.findall(r"\bwhite\b", base, flags=re.I)) != 1:
                ok = False; dropped.append((int(qid), "race word count != 1")); break
            m = re.search(r"\bwhite\b", base, flags=re.I)
            # explicit variants (the base itself is the white variant)
            for race in RACES:
                word = race if race != "white" else m.group(0)
                text = base[:m.start()] + word + base[m.end():]
                d = word_diff(base, text)
                if not allowed_diff(d, {"white", "White"} | set(race.split())):
                    ok = False; dropped.append((int(qid), f"explicit diff {d}")); break
                variants.append(dict(fill_type="explicit", race=race, gender=gender, name=None,
                                     question=text, cue_text=word, cue_char_span=[m.start(), m.start() + len(word)],
                                     base_uid=f"discrim_mp_q{qid:02d}_explicit_white_{gender}"))
            if not ok:
                break
            # implicit variants: name only
            for race in RACES:
                for k, name in enumerate(NAMES[(race, gender)]):
                    text, err = implicit_text(nlp, base, name)
                    if text is None:
                        ok = False; dropped.append((int(qid), f"implicit: {err}")); break
                    d = word_diff(base, text)
                    if not allowed_diff(d, {"white", "White", "named"} | set(name.split())):
                        ok = False; dropped.append((int(qid), f"implicit diff {d}")); break
                    i = text.index(name)
                    variants.append(dict(fill_type="implicit", race=race, gender=gender, name=name, name_index=k,
                                         question=text, cue_text=name, cue_char_span=[i, i + len(name)],
                                         base_uid=[f"discrim_mp_q{qid:02d}_implicit_white_{gender}_n{j}" for j in range(len(NAMES[("white", gender)]))]))
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        for v in variants:
            race_tag = v["race"].replace(" ", "")
            uid = f"discrim_mp_q{qid:02d}_{v['fill_type']}_{race_tag}_{v['gender']}" + (f"_n{v['name_index']}" if v["fill_type"] == "implicit" else "")
            qwc = v["question"] + "\n\nChoices:\nA) Yes\nB) No"
            formatted, ids = fmt(tok, qwc)
            regex = RACE_WORDS[v["race"]] + GENDER_WORDS[v["gender"]] + ["race", "ethnic", "gender", "demograph"]
            if v["name"]:
                regex = regex + [w.lower() for w in v["name"].split()]
            v_out = dict(
                uid=uid, setting=f"discrim_mp_{v['fill_type']}", profile_id=int(qid), qid=int(qid),
                axis="race" if v["fill_type"] == "explicit" else "name", value=race_tag + "_" + v["gender"],
                race=v["race"], gender=v["gender"], name=v["name"], fill_type=v["fill_type"], age=args.age,
                base_uid=v["base_uid"], cue_text=v["cue_text"], cue_char_span=v["cue_char_span"],
                cue_sentence="", cue_regex=regex,
                question=v["question"], question_with_choices=qwc, prompt=formatted, prompt_token_ids=ids,
                all_letters=["A", "B"], all_answers=["Yes", "No"],
                dataset_name=f"discrim_mp_{v['fill_type']}", dataset_type="multiple choice",
            )
            out.append(v_out)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    qids = sorted({v["qid"] for v in out})
    print(f"Wrote {len(out)} prompts for {len(qids)} questions -> {args.output}")
    print(f"Dropped {len(dropped)} questions:")
    for q, why in dropped:
        print(f"  q{q:02d}: {why}")


if __name__ == "__main__":
    main()
