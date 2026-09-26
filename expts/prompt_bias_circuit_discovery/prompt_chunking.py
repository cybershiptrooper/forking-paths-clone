"""Chunk the prompt of a collection record into units for the sentence-pair
mask, and check that the demographic cue does not share a unit with other
task information.

Chunk methods for the question text (the instruction line, each answer
option and the chat-template tail are always their own chunks):

- ``sentence``: split at sentence punctuation and newlines (the resume
  prompts put every fact on its own line, so this is one fact per chunk).
- ``content``: one chunk per content word (NOUN, PROPN, ADJ, NUM, VERB by
  spaCy), with the function words and punctuation that follow it attached;
  named entities (a person's name, "Native American", "$150,000") are kept
  as one chunk. The middle ground between sentences and tokens.
- ``llm``: an LLM splits the text into the smallest units of information,
  each an exact substring; a person's name or a stated attribute must be
  its own unit. Validated by exact reconstruction; falls back to
  ``content`` when the segmentation does not reconstruct the text.
- ``token``: one chunk per token.

Every chunk list covers the prompt tokens contiguously and is stored on the
record as ``prompt_chunk_spans`` (inclusive token spans), which
``_build_prefix`` uses in place of the sentence splitter for the prompt.

The confound check: the cue tokens are the tokens that differ from the
baseline variant (resume: the demographic sentence; discrim minimal pairs:
the race word or the name) or, for BBQ, the group words of the two people.
A cue chunk is *clean* when the text of the chunk outside the cue contains
no content word.
"""

from __future__ import annotations

import json
import re
from typing import Optional

CONTENT_POS = {"NOUN", "PROPN", "ADJ", "NUM", "VERB"}
ENT_LABELS = {"PERSON", "NORP", "GPE", "ORG", "MONEY", "DATE", "CARDINAL", "PERCENT", "QUANTITY", "LOC", "FAC", "LANGUAGE"}
# words that describe the person's role rather than a fact about them; they
# are reported separately so a cue chunk like "the candidate named X" is
# not flagged for "candidate"
ROLE_WORDS = {"candidate", "applicant", "borrower", "patient", "customer", "user", "client", "employee", "person",
              "student", "defendant", "claimant", "recipient", "requester", "traveler", "visitor", "subject",
              "tenant", "owner", "holder", "inventor", "author", "photographer", "journalist", "contractor",
              "recruit", "entrepreneur", "athlete", "caller", "fellow", "professor", "male", "female", "man", "woman",
              "boy", "girl", "kid", "kids", "friend", "friends", "guy", "guys", "named", "name", "application", "submitted",
              "resume", "email"}  # template words of the Karvonen hiring prompt ("Candidate resume is: Name: ...")

_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# --------------------------------------------------------------------------
# token / char alignment
# --------------------------------------------------------------------------
def token_offsets(tok, formatted, ids):
    enc = tok(formatted, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == list(ids), "prompt_token_ids do not match the formatted prompt"
    return enc["offset_mapping"]


def char_spans_to_token_spans(spans, offsets, lo, hi, text):
    """Assign each token in [lo, hi] to the span containing its first
    non-space character; tokens not covered join the previous chunk.
    Returns a list of (start, end) token spans, in order, covering [lo, hi]."""
    out = []
    for i in range(lo, hi + 1):
        cs, ce = offsets[i]
        piece = text[cs:ce]
        cs_eff = cs + (len(piece) - len(piece.lstrip())) if piece.strip() else cs
        j = None
        for k, (s, e) in enumerate(spans):
            if s <= cs_eff < e:
                j = k
                break
        if out and (j is None or j == out[-1][2]):
            out[-1][1] = i
        else:
            out.append([i, i, j])
    return [(s, e) for s, e, _ in out]


# --------------------------------------------------------------------------
# question-text segmentation
# --------------------------------------------------------------------------
def seg_sentence(text):
    spans, pos = [], 0
    for m in re.finditer(r".+?(?:(?<=[.!?])\s+|\n+|$)", text, flags=re.S):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def seg_token(text, offsets_in_text):
    return [(s, e) for s, e in offsets_in_text if e > s]


def seg_content(text):
    doc = nlp()(text)
    anchors = []
    ent_of = {}
    for ent in doc.ents:
        if ent.label_ in ENT_LABELS:
            for t in ent:
                ent_of[t.i] = ent
    i = 0
    while i < len(doc):
        t = doc[i]
        if t.i in ent_of:
            ent = ent_of[t.i]
            anchors.append(ent.start_char)
            i = ent.end
            continue
        if t.pos_ in CONTENT_POS and not t.is_punct:
            anchors.append(t.idx)
        i += 1
    anchors = sorted(set(anchors))
    if not anchors or anchors[0] > 0:
        anchors = [0] + anchors
    spans = [(a, b) for a, b in zip(anchors, anchors[1:] + [len(text)])]
    return spans


LLM_SEG_PROMPT = """Split the text below into the smallest units of information, in order. Rules:
1. Every unit must be an exact, contiguous substring of the text; the units concatenated in order must reproduce the text exactly (keep all whitespace and punctuation inside the units).
2. Each unit states one fact, quantity, attribute or question; do not merge two facts into one unit.
3. A person's name, and any stated demographic attribute (race, ethnicity, gender, age, nationality, religion, disability), must be its own unit, separate from the person's role, occupation and every other fact. E.g. "The borrower is a 60-year-old white male mechanic who" -> ["The borrower is a ", "60-year-old ", "white ", "male ", "mechanic ", "who"].
4. Output only a JSON list of strings.

Text:
{text}"""


def seg_llm(text, client, model, cache, cache_path, fallback_model="google/gemini-2.5-flash"):
    """LLM segmentation with exact-reconstruction validation. ``model`` is
    tried twice (the second time with a note that the first answer did not
    reconstruct the text), then ``fallback_model`` twice (the primary model
    with mandatory reasoning sometimes exhausts its token budget on short
    texts), then the content-word chunking."""
    from expts.prompt_bias_circuit_discovery.openrouter_client import chat
    norm = lambda s: re.sub(r"\s+", "", s)

    def parse(out):
        m = re.search(r"\[.*\]", out, flags=re.S)
        if not m:
            return None
        try:
            units = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not all(isinstance(u, str) for u in units) or norm("".join(units)) != norm(text):
            return None
        spans, pos = [], 0
        for u in units:
            u2 = u.strip()
            if not u2:
                continue
            k = text.find(u2, pos)
            if k < 0:
                return None
            spans.append((k, k + len(u2)))
            pos = k + len(u2)
        return spans or None

    for mdl in [model, fallback_model]:
        if not mdl:
            continue
        for attempt in range(2):
            msgs = [{"role": "user", "content": LLM_SEG_PROMPT.format(text=text)}]
            if attempt:
                msgs[0]["content"] += "\n\n(Second attempt: your previous answer did not reconstruct the text exactly. Copy substrings verbatim.)"
            try:
                out = chat(client, mdl, msgs, max_tokens=4000 if mdl == model else 2000, temperature=0.0, retries=2,
                           cache=cache, cache_path=cache_path)
            except Exception:  # noqa: BLE001
                break
            spans = parse(out)
            if spans:
                return spans, ("llm" if mdl == model else f"llm(fallback {mdl})")
    return seg_content(text), "content(fallback)"


# --------------------------------------------------------------------------
# the record-level chunker
# --------------------------------------------------------------------------
def split_at(spans, cuts):
    """Insert boundaries at every char position in ``cuts``."""
    out = []
    for s, e in spans:
        inner = sorted({c for c in cuts if s < c < e})
        pos = s
        for c in inner:
            out.append((pos, c)); pos = c
        out.append((pos, e))
    return out


def chunk_prompt(tok, record, method, llm=None, forced_spans=None):
    """Return (chunks, info). ``chunks``: list of dicts with inclusive token
    ``start``/``end`` over ``record['prompt_token_ids']``, ``kind`` in
    {instruction, question, choices_header, choice, tail} and ``text``.

    ``forced_spans``: char spans inside record['question'] (the known cue:
    a name or an attribute word) that must start and end a chunk under the
    ``content`` and ``llm`` methods, since spaCy's entity detection misses
    unusual names. The ``token`` method ignores it."""
    formatted, ids = record["prompt"], record["prompt_token_ids"]
    offsets = token_offsets(tok, formatted, ids)
    q = record["question"]
    q0 = formatted.index(q)
    q1 = q0 + len(q)
    qwc = record["question_with_choices"]
    choices_text = qwc[len(q):]
    c0 = q1
    c1 = q1 + len(choices_text)
    assert formatted[c0:c1] == choices_text
    # fixed regions: [0, q0) instruction (+ template head), question, choices, tail
    spans = [(0, q0)]
    kinds = ["instruction"]
    info = {"method": method}
    if method == "sentence":
        qs = seg_sentence(q)
    elif method == "content":
        qs = seg_content(q)
    elif method == "token":
        qs = [(s - q0, e - q0) for i, (s, e) in enumerate(offsets) if s >= q0 and e <= q1 and e > s]
    elif method == "llm":
        assert llm is not None, "method=llm needs llm=(client, model, cache, cache_path)"
        qs, used = seg_llm(q, *llm)
        info["llm_status"] = used
    else:
        raise ValueError(method)
    if forced_spans and method in ("content", "llm", "sentence"):
        qs = split_at(qs, [c for s, e in forced_spans for c in (s, e)])
    spans += [(q0 + s, q0 + e) for s, e in qs]
    kinds += ["question"] * len(qs)
    # choices: header line, then each option line
    for m in re.finditer(r".+?(?:\n|$)", choices_text):
        if m.group(0).strip():
            spans.append((c0 + m.start(), c0 + m.end()))
            kinds.append("choices_header" if m.group(0).strip().startswith("Choices") else "choice")
    spans.append((c1, len(formatted)))
    kinds.append("tail")
    tspans = char_spans_to_token_spans(spans, offsets, 0, len(ids) - 1, formatted)
    # char_spans_to_token_spans keeps only spans that own a token; recover kinds by first token's char
    chunks = []
    for s, e in tspans:
        cs, ce = offsets[s]
        piece = formatted[cs:ce]
        cs = cs + (len(piece) - len(piece.lstrip())) if piece.strip() else cs
        k = next((kinds[j] for j, (a, b) in enumerate(spans) if a <= cs < b), "question")
        chunks.append(dict(start=s, end=e, kind=k, text=tok.decode(ids[s:e + 1]),
                           char_start=offsets[s][0], char_end=offsets[e][1]))
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(ids) - 1
    assert all(chunks[i]["end"] + 1 == chunks[i + 1]["start"] for i in range(len(chunks) - 1))
    return chunks, info


# --------------------------------------------------------------------------
# cue location and the confound check
# --------------------------------------------------------------------------
def minimal_pair_diff(record, baseline_record):
    """Word-level diff of record['question'] against the baseline: list of
    (char_start, char_end, replaced_text) runs in the record's question."""
    import difflib
    q, b = record["question"], baseline_record["question"]
    wq = [(m.start(), m.end()) for m in re.finditer(r"\S+", q)]
    wb = [(m.start(), m.end()) for m in re.finditer(r"\S+", b)]
    sm = difflib.SequenceMatcher(None, [b[s:e] for s, e in wb], [q[s:e] for s, e in wq], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        span = (wq[j1][0], wq[j2 - 1][1]) if j2 > j1 else (wq[j1][0] if j1 < len(wq) else len(q),) * 2
        out.append(dict(tag=tag, span=span, new=q[span[0]:span[1]], old=" ".join(b[s:e] for s, e in wb[i1:i2])))
    return out


UNKNOWN_ANSWERS = {"not answerable", "unknown", "can't be determined", "cannot be determined", "not enough info",
                   "not enough information", "undetermined", "can't answer", "not known", "cannot answer",
                   "not specified", "can not be determined", "can not answer"}


def cue_char_spans(record):
    """Char spans of the cue inside record['question']: the builder's
    ``cue_char_span`` (discrim minimal pairs), else the demographic sentence
    (resume prompts), else the group words and person descriptors of BBQ."""
    q = record["question"]
    if record.get("cue_char_span"):
        return [tuple(record["cue_char_span"])]
    if record.get("cue_sentence") and record["cue_sentence"] in q:
        i = q.index(record["cue_sentence"])
        return [(i, i + len(record["cue_sentence"]))]
    words = list(record.get("stereotyped_groups", []) or [])
    for a in record.get("all_answers", []):
        a2 = re.sub(r"^(The|the|A|An|a|an)\s+", "", a).strip()
        if a2.lower() not in UNKNOWN_ANSWERS:
            words.append(a2)
    spans = []
    for w in set(words):
        if not w:
            continue
        for m in re.finditer(r"\b" + re.escape(w) + r"\b", q, flags=re.I):
            spans.append((m.start(), m.end()))
    if not spans:
        # BBQ items whose answer descriptors paraphrase the context ("The well
        # off one" vs "a well off person"): fall back to the descriptors'
        # content words ("welfare", "rich", "poor") found in the context.
        for w in set(words):
            for t in nlp()(w):
                if t.pos_ in ("NOUN", "ADJ", "PROPN") and len(t.text) > 2 and t.text.lower() not in ROLE_WORDS:
                    for m in re.finditer(r"\b" + re.escape(t.text) + r"\b", q, flags=re.I):
                        spans.append((m.start(), m.end()))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


def cue_token_set(tok, record, cue_spans_in_q):
    formatted, ids = record["prompt"], record["prompt_token_ids"]
    offsets = token_offsets(tok, formatted, ids)
    q0 = formatted.index(record["question"])
    abs_spans = [(q0 + s, q0 + e) for s, e in cue_spans_in_q]
    toks = set()
    for i, (cs, ce) in enumerate(offsets):
        for s, e in abs_spans:
            if cs < e and ce > s and ce > cs:
                toks.add(i)
    return toks


def _content_words(text):
    doc = nlp()(text)
    words = [t.text for t in doc if t.pos_ in CONTENT_POS and not t.is_punct and not t.is_space]
    role = [w for w in words if w.lower() in ROLE_WORDS]
    other = [w for w in words if w.lower() not in ROLE_WORDS]
    return other, role


def confound_report(tok, record, chunks, cue_spans_in_q):
    """For every chunk that overlaps the cue: the content words of the
    chunk's text outside the cue (computed on characters, so a name split
    across tokens is never counted as residual)."""
    formatted = record["prompt"]
    q0 = formatted.index(record["question"])
    abs_cue = [(q0 + s, q0 + e) for s, e in cue_spans_in_q]
    cue_toks = cue_token_set(tok, record, cue_spans_in_q)
    rows = []
    for ci, c in enumerate(chunks):
        inside = [i for i in range(c["start"], c["end"] + 1) if i in cue_toks]
        if not inside:
            continue
        cs, ce = c["char_start"], c["char_end"]
        pieces, pos = [], cs
        for s, e in sorted(abs_cue):
            if e <= cs or s >= ce:
                continue
            if s > pos:
                pieces.append(formatted[pos:s])
            pos = max(pos, e)
        if pos < ce:
            pieces.append(formatted[pos:ce])
        rest = " ".join(pieces)
        rest = re.sub(r"<\|im_(start|end)\|>|\buser\b|\bassistant\b", " ", rest)
        other, role = _content_words(rest)
        rows.append(dict(chunk=ci, kind=c["kind"], text=c["text"], n_cue_tokens=len(inside),
                         n_tokens=c["end"] - c["start"] + 1, residual_content=other, residual_role=role,
                         clean=len(other) == 0))
    return dict(n_chunks=len(chunks), n_cue_tokens=len(cue_toks), cue_chunks=rows,
                all_clean=all(r["clean"] for r in rows) and bool(rows))
