"""Answer-guard variants of a termination bank's boundary data.

The boundary-hazard objectives reward stopping only at paragraph breaks
whose forced answer probe agrees with the *trace's own* answer
(``eligible``), or weight each stopping term by the probe's probability
of that answer (``probe_p_trace``; see build_boundary_data.py).  On the
10 termination prompts the trace's answer equals the correct answer, so
this guard always promotes the correct answer.  This script writes two
alternative guards for the same bank so the trainers can be run unchanged
on them (they only read ``eligible`` / ``probe_p_trace`` from the file):

- ``per_chain``: each sampled continuation's *own* final answer is the
  answer to preserve for that continuation.  ``eligible[b]`` is true
  where the clean probe at boundary b returns that chain's answer, and
  ``probe_p_trace[b]`` holds the probe's probability of that answer.
  The chain answer is the bank's graded answer for terminated chains and
  the probe label at the last recorded boundary for chains that hit the
  horizon (the model's answer if forced to stop there).
- ``no_guard``: every boundary is eligible and every weight is 1 — no
  preference over answers; only stopping sooner is rewarded.

The full per-letter probe distribution at every boundary is recomputed
(the original file stores only the trace-answer probability) and saved as
``probe_probs`` in both outputs; the recomputed argmax labels are checked
against the stored ``probe_labels``.

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.build_boundary_data_variants \
        --boundary_data_path results/cot_termination/early_2200/boundary_data/aqua_p008_s78.json \
        --output_dir results/cot_termination/early_2200/boundary_data_variants
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re

import torch

from utils.utils import set_seed, clear_cuda
from utils.circuit_eval import install_clean_sdpa_forward

from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.cot_termination_circuit_discovery.answer_bank_utils import (
    flatten_bank_candidates,
)
from expts.cot_termination_circuit_discovery.build_boundary_data import (
    THINK_END_ID, PROBE_SUFFIX_TEXT, _probe_at_boundary,
)


def _bank_candidate_answers(bank):
    """Graded answer letter of every flattened candidate (None if the
    candidate did not terminate)."""
    out = []
    for c in bank["candidates"]:
        variants = c.get("continuation_token_ids_variants") or [
            c["continuation_token_ids"]
        ]
        m = re.search(r"probe=([A-Z])", c.get("answer_text") or "")
        ans = m.group(1) if (c.get("terminated") and m) else None
        out.extend([ans] * len(variants))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary_data_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    bd = json.load(open(args.boundary_data_path))
    bank = json.load(open(bd["bank_path"]))
    stem = os.path.splitext(os.path.basename(args.boundary_data_path))[0]
    all_letters = bank.get("all_letters") or ["A", "B", "C", "D"]
    trace_answer = bd["trace_answer"]

    model, tokenizer = load_model_eager(args.model_name, device="cuda")
    install_clean_sdpa_forward(model)
    device = next(model.parameters()).device
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]

    prefix_ids, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=bd["data_path"],
        prompt_index=int(bd["prompt_index"]), base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=int(bd["analysis_sentence_step"]),
        sentences_after_prefix=int(bank.get("sentences_after_prefix", 0)),
        min_sentence_length=int(bd.get("min_sentence_length", 10)),
        sentence_chunk=1,
    )
    token_lists, _, _ = flatten_bank_candidates(bank)
    bank_answers = _bank_candidate_answers(bank)
    assert len(token_lists) == len(bd["candidates"]) == len(bank_answers)

    n_label_mismatch = 0
    chain_answers = []
    for c, ids, bank_ans in zip(bd["candidates"], token_lists, bank_answers):
        terminated = THINK_END_ID in ids
        content = ids[: ids.index(THINK_END_ID)] if terminated else list(ids)
        probs_all, labels = [], []
        for j in c["boundaries"]:
            label, probs = _probe_at_boundary(
                model, prefix_ids, content, j, suffix_ids, letter_ids, device,
            )
            labels.append(label)
            probs_all.append(probs)
        n_label_mismatch += sum(
            1 for a, b in zip(labels, c["probe_labels"]) if a != b
        )
        c["probe_probs"] = probs_all
        c["probe_labels_recomputed"] = labels
        chain_ans = bank_ans if bank_ans is not None else (
            labels[-1] if labels else trace_answer
        )
        c["chain_answer"] = chain_ans
        c["chain_answer_source"] = (
            "bank_graded" if bank_ans is not None else "last_boundary_probe"
        )
        chain_answers.append(chain_ans)
        clear_cuda()
    n_b = sum(len(c["boundaries"]) for c in bd["candidates"])
    print(f"{stem}: {len(bd['candidates'])} candidates, {n_b} boundaries, "
          f"{n_label_mismatch} recomputed probe labels differ from stored")
    print(f"  trace answer {trace_answer}; chain answers {chain_answers}")

    os.makedirs(args.output_dir, exist_ok=True)

    per_chain = copy.deepcopy(bd)
    per_chain["guard_variant"] = "per_chain_answer"
    for c in per_chain["candidates"]:
        ans = c["chain_answer"]
        c["eligible"] = [lab == ans for lab in c["probe_labels"]]
        c["probe_p_trace"] = [p.get(ans, 0.0) for p in c["probe_probs"]]
    n_any = sum(1 for c in per_chain["candidates"] if any(c["eligible"]))
    print(f"  per_chain: {n_any}/{len(per_chain['candidates'])} candidates "
          f"with >=1 eligible boundary; eligible fraction "
          f"{sum(sum(c['eligible']) for c in per_chain['candidates']) / n_b:.2f} "
          f"(original {sum(sum(c['eligible']) for c in bd['candidates']) / n_b:.2f})")
    p = os.path.join(args.output_dir, f"{stem}_perchain.json")
    json.dump(per_chain, open(p, "w"))
    print(f"  saved {p}")

    no_guard = copy.deepcopy(bd)
    no_guard["guard_variant"] = "no_answer_guard"
    for c in no_guard["candidates"]:
        c["eligible"] = [True] * len(c["boundaries"])
        c["probe_p_trace"] = [1.0] * len(c["boundaries"])
    p = os.path.join(args.output_dir, f"{stem}_noguard.json")
    json.dump(no_guard, open(p, "w"))
    print(f"  saved {p}")


if __name__ == "__main__":
    main()
