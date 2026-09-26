"""Render the generation prompts for the selected questions.

Writes ``data/external_compression/prompts_rendered.json`` with, per
question: the user message, the full prompt string (chat template +
``<think>\n``), and the letter set.  Run this BEFORE generation; the
rendered prompts are what ``generate.py`` feeds to vLLM, and 1-2 of them
are read by hand as the pre-generation checkpoint.

Usage:
    uv run python -m expts.external_compression.prep [--show QUESTION_ID]
"""

from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer

from expts.external_compression.common import (
    DATA_DIR,
    MODEL_NAME,
    build_prompt_str,
    build_user_msg,
    letter_token_ids,
    list_spec_ids,
    load_spec,
)

OUT_PATH = os.path.join(DATA_DIR, "prompts_rendered.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default=None, help="print one rendered prompt in full")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rendered = {}
    for qid in list_spec_ids():
        spec = load_spec(qid)
        choices = spec["choices"]
        assert choices, f"{qid}: empty choices list"
        user_msg = build_user_msg(spec["question_text"], choices)
        prompt_str = build_prompt_str(tokenizer, user_msg)
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        ans_ids = letter_token_ids(tokenizer, letters)
        assert len(set(ans_ids)) == len(ans_ids), f"{qid}: letter token collision"
        rendered[qid] = {
            "question_id": qid,
            "n_sentences_in_cot_theirs": int(spec["n_sentences_in_cot"]),
            "num_choices": len(choices),
            "letters": letters,
            "letter_token_ids": ans_ids,
            "user_msg": user_msg,
            "prompt_str": prompt_str,
            "prompt_tokens": len(tokenizer.encode(prompt_str, add_special_tokens=False)),
        }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rendered, f, indent=2)
    print(f"Wrote {len(rendered)} rendered prompts to {OUT_PATH}")
    for qid, r in rendered.items():
        print(f"  {qid:35s} choices={r['num_choices']} prompt_tokens={r['prompt_tokens']} "
              f"(their trace length prior: {r['n_sentences_in_cot_theirs']})")

    if args.show:
        r = rendered[args.show]
        print("\n" + "=" * 80)
        print(f"FULL RENDERED PROMPT — {args.show}")
        print("=" * 80)
        print(r["prompt_str"])
        print("=" * 80)


if __name__ == "__main__":
    main()
