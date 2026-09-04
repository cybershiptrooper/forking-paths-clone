"""Pick a base path from a data-collection record (for sentence-step analysis).

The collection JSON produced by `expts/forking_paths/data_collection_new.py`
stores ONE chosen base path per record (with `output_token_ids`) plus a list
of `alternate_texts` (strings only, no token ids). To support
`base_answer_type` selection at learn time we sometimes need to pick an
alternate, which means retokenizing its text.

TODO: extend `expts/forking_paths/data_collection_new.py` to also persist
`alternate_token_ids` (and `alternate_finish_reasons` if useful), then drop
the retokenization branch below — it is lossy on edge cases like leading-space
tokens, BOS handling, or UTF-8 boundary differences between the original
generation tokenizer and the one loaded here.
"""

from collections import Counter


def select_base_from_record(record: dict, base_answer_type: str, tokenizer):
    """Return token IDs for the chosen base path from a collection record.

    base_answer_type:
        'stored'    — use record['output_token_ids'] (path selected at collection time)
        'correct'   — pick a path whose clean_answer == correct_answer
        'incorrect' — pick a path whose clean_answer != correct_answer
        'mode'      — pick a path matching the mode answer across base + alternates

    Prefers the stored base whenever it satisfies the criterion (no
    retokenization needed). Otherwise picks the first matching alternate
    text and retokenizes it, with a WARNING.
    """
    if base_answer_type == "stored":
        return list(record["output_token_ids"])

    correct = record.get("correct_answer") or record.get("correct_letter")
    base_answer = record.get("clean_answer")
    alternate_answers = record.get("alternate_answers") or []
    alternate_texts = record.get("alternate_texts") or []
    if len(alternate_texts) != len(alternate_answers):
        raise ValueError(
            f"alternate_texts ({len(alternate_texts)}) and alternate_answers "
            f"({len(alternate_answers)}) length mismatch in record. Re-run "
            f"data_collection_new.py with --return_alternate_texts."
        )

    # Each candidate: (kind, answer, payload). For 'base' the payload is
    # token_ids; for 'alternate' it is the raw text (lossy retokenization).
    candidates = [("base", base_answer, list(record["output_token_ids"]))]
    for alt_ans, alt_text in zip(alternate_answers, alternate_texts):
        candidates.append(("alternate", alt_ans, alt_text))

    if base_answer_type == "correct":
        matches = [c for c in candidates if c[1] == correct]
    elif base_answer_type == "incorrect":
        matches = [c for c in candidates if c[1] != correct]
    elif base_answer_type == "mode":
        mode_ans = Counter(c[1] for c in candidates).most_common(1)[0][0]
        matches = [c for c in candidates if c[1] == mode_ans]
    else:
        raise ValueError(
            f"unknown base_answer_type: {base_answer_type!r} "
            f"(expected 'stored', 'correct', 'incorrect', or 'mode')"
        )

    if not matches:
        raise ValueError(
            f"No path satisfies base_answer_type={base_answer_type!r} "
            f"(correct={correct!r}, base={base_answer!r}, "
            f"alternates={alternate_answers!r})"
        )

    base_match = next((c for c in matches if c[0] == "base"), None)
    if base_match is not None:
        print(
            f"  base_answer_type={base_answer_type!r}: stored base path already "
            f"matches (answer={base_match[1]!r}), no retokenization needed."
        )
        return base_match[2]

    chosen = matches[0]
    print(
        f"  WARNING: base_answer_type={base_answer_type!r} selected an alternate "
        f"path (answer={chosen[1]!r}); retokenizing its text. This is lossy on "
        f"edge cases (leading-space tokens, BOS handling). "
        f"TODO: persist alternate_token_ids in data_collection_new.py to remove "
        f"this retokenization."
    )
    return tokenizer.encode(chosen[2], add_special_tokens=False)
