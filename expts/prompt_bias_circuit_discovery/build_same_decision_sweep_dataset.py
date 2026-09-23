"""Dataset for the analysis-point sweep on the same-decision admission set.

One example per input (a unique prompt with its own on-policy rollout):
the first stored admit rollout under the cue name of every positive and
every negative input of the same-decision judge set. Positives are ordered
by attributable share (largest first), negatives by |admit-rate difference|
(smallest first), and the two classes are interleaved so that any prefix of
the dataset is balanced.

Adds the fields ``eval_analysis_point_sweep.py`` needs: ``prompt_chunk_spans``
(sentence chunking of the prompt; the sweep removes reads of every prompt
chunk together, so the chunk granularity does not matter), ``n_prompt_chunks``,
``analysis_timestep`` (the ``</think>`` position) and ``mask_target_letter``
(the trace's own answer, always the admit letter A).

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.build_same_decision_sweep_dataset \
        --judge_set results/prompt_bias_v2/judge_sets_same_decision/qwen3_8b_admission_same_decision.json \
        --out results/prompt_bias_v2/same_decision_sweep/dataset_qwen3_8b_black.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os

from transformers import AutoTokenizer

from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_set", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--model_tag", default="qwen3_8b")
    ap.add_argument("--n_pos", type=int, default=None, help="keep at most this many positives (default all)")
    ap.add_argument("--n_neg", type=int, default=None, help="keep at most this many negatives (default all)")
    ap.add_argument("--group", default="black", help="name group of the judge set (used when the records lack the field)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    think_id = tok.convert_tokens_to_ids("</think>")
    J = json.load(open(args.judge_set))
    first = {}
    for r in J:  # first stored admit rollout of every input
        if r["uid"] not in first or r["rollout_index"] < first[r["uid"]]["rollout_index"]:
            first[r["uid"]] = r
    pos = sorted([r for r in first.values() if r["is_positive"]], key=lambda r: -(r["attributable_white"] or 0))
    neg = sorted([r for r in first.values() if not r["is_positive"]], key=lambda r: abs(r["delta_white"]))
    pos = pos[:args.n_pos] if args.n_pos else pos
    neg = neg[:args.n_neg] if args.n_neg else neg
    order = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos):
            order.append(pos[i])
        if i < len(neg):
            order.append(neg[i])
    dataset = []
    for r in order:
        rec = dict(r)
        chunks, _ = chunk_prompt(tok, rec, "sentence")
        rec["prompt_chunk_spans"] = [[c["start"], c["end"]] for c in chunks]
        rec["prompt_chunk_kinds"] = [c["kind"] for c in chunks]
        rec["n_prompt_chunks"] = len(chunks)
        rec["chunk_method"] = "sentence"
        ids = rec["output_token_ids"]
        assert ids.count(think_id) == 1, rec["tag"]
        rec["analysis_timestep"] = ids.index(think_id)
        rec["mask_target_letter"] = rec["clean_answer"]
        rec["group"] = rec.get("group") or args.group
        rec["subset"] = rec["group"]
        rec["pool_tag"] = rec["tag"]
        rec["example_id"] = len(dataset)
        rec["tag"] = f"{args.model_tag}_same_decision_{rec['group']}_{len(dataset):03d}_{'pos' if rec['is_positive'] else 'neg'}"
        dataset.append(rec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dataset, open(args.out, "w"))
    n_pos = sum(r["is_positive"] for r in dataset)
    print(f"{len(dataset)} examples: {n_pos} positives, {len(dataset) - n_pos} negatives; one rollout per input;",
          "prompt chunks:", collections.Counter(r["n_prompt_chunks"] for r in dataset).most_common(3),
          "; reasoning tokens median", sorted(r["analysis_timestep"] for r in dataset)[len(dataset) // 2], "->", args.out)


if __name__ == "__main__":
    main()
