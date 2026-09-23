"""Dataset and training configs for the "reduce P(trace's answer)" mask
experiment on the same-decision admission set.

Prompts: ``--n_pos`` inputs whose admit rate the Black name raised the most
(largest attributable share) and ``--n_neg`` inputs where the name did
nothing (smallest |difference|, Black-name admit rate near 0.5). From each
input, under the Black name, the first stored admit (A) rollout and the
first stored reject (B) rollout. Every trace is one example.

Per example the prompt is chunked into sentences (the applicant's name its
own chunk at every mention) with the chat-template start ``<|im_start|>user\\n``
left outside every chunk, and the first two reasoning tokens (``<think>\\n``)
outside every sentence: tokens outside a sentence are never masked
(attention sinks stay readable). The answer-option chunks are listed as
``frozen_key_sentences`` so that their reads are never masked either.

Configs (subnetwork probing, region trainer): pool in {prompt_to_trace,
trace_to_trace} x cut in {half of the reasoning sentences, the whole
reasoning up to </think>} x sparsity in ``--sparsities`` (fraction of the
pool removed), objective ``answer_probe_reward_gap`` towards the *other*
letter, i.e. lower P(trace's answer) at the probe.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.build_reduce_mask_experiment --name reduce_masks_qwen3_8b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import torch
import yaml
from transformers import AutoTokenizer

from utils.cot_analysis import split_tokens_into_sentences
from expts.prompt_bias_circuit_discovery.gen_sweep_configs import CANONICAL
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt
from expts.direct_answer_circuit_discovery.learn import split_with_prompt_chunks

SINK_TOKENS = 3  # <|im_start|> user \n
THINK_TOKENS = 2  # <think> \n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/prompt_bias_v2/analysis/admission_same_decision_dataset_selection.json")
    ap.add_argument("--rollouts", default="results/prompt_bias_v2/rollouts_confirm_admission_full/qwen3_8b_admission_full_confirm_shard*of8.json")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_pos", type=int, default=4)
    ap.add_argument("--n_neg", type=int, default=4)
    ap.add_argument("--sparsities", nargs="+", type=float, default=[0.2, 0.8])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    think_id = tok.convert_tokens_to_ids("</think>")
    sel = json.load(open(args.selection))
    R = {}
    for f in glob.glob(args.rollouts):
        for r in json.load(open(f)):
            R[r["uid"]] = r
    pos = sorted(sel["positives"], key=lambda it: -(it.get("attributable") or 0))
    neg = sorted(sel["negatives"], key=lambda it: (abs(it["delta"]), abs(it["p1"] - 0.5)))
    chosen = []
    for cls, items, n in [("pos", pos, args.n_pos), ("neg", neg, args.n_neg)]:
        k = 0
        for it in items:
            rec = R[it["uid"]]
            a = [x for x in rec["rollouts"] if x["answer"] == "A"]; b = [x for x in rec["rollouts"] if x["answer"] == "B"]
            if len(a) >= 4 and len(b) >= 4:
                chosen.append((cls, it, rec, a[0], b[0])); k += 1
            if k >= n:
                break
    dataset = []
    for cls, it, rec, ra, rb in chosen:
        fa, la = rec["name"].split()[0], rec["name"].split()[-1]
        forced = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(rec['name'])}\b|\b{re.escape(fa)}\b|\b{re.escape(la)}\b", rec["question"])]
        chunks, _ = chunk_prompt(tok, rec, "sentence", forced_spans=forced)
        assert tok.decode(rec["prompt_token_ids"][:SINK_TOKENS]) == "<|im_start|>user\n"
        chunks[0]["start"] = SINK_TOKENS  # the chat-template start stays outside every chunk (never masked)
        spans = [[c["start"], c["end"]] for c in chunks]; kinds = [c["kind"] for c in chunks]
        frozen = [j for j, k in enumerate(kinds) if k in ("choices_header", "choice")]
        for letter, x in [("A", ra), ("B", rb)]:
            ids = x["token_ids"]
            assert ids[:THINK_TOKENS] == [151667, 198] and ids.count(think_id) == 1
            t_think = ids.index(think_id)
            n_r = len(split_tokens_into_sentences(torch.tensor(ids[THINK_TOKENS:t_think]), tok, 10))
            d = dict(uid=rec["uid"], base_uid=it["baseline_uid"], family="blindspot_admission", case=f"reduce_{cls}", axis="name", value=rec["value"],
                     race=rec["race"], gender=rec["gender"], name=rec["name"], question=rec["question"], question_with_choices=rec["question_with_choices"],
                     prompt=rec["prompt"], prompt_token_ids=rec["prompt_token_ids"], all_letters=rec["all_letters"], all_answers=rec["all_answers"],
                     cue_char_span=rec.get("cue_char_span"), output_text=x["text"], output_token_ids=ids, clean_answer=x["answer"],
                     is_positive=(cls == "pos"), p_black=it["p1"], p_white=it["p0"], delta_white=it["delta"], attributable_white=it.get("attributable"),
                     prompt_chunk_spans=spans, prompt_chunk_kinds=kinds, n_prompt_chunks=len(chunks), chunk_method="sentence", reasoning_skip_tokens=THINK_TOKENS,
                     frozen_key_sentences=frozen, analysis_timestep=t_think, n_reasoning_sentences=n_r,
                     analysis_sentence_step_half=len(chunks) + max(1, round(0.5 * n_r)),
                     mask_target_letter=letter, other_letter=("B" if letter == "A" else "A"), example_id=len(dataset),
                     tag=f"qwen3_8b_reduce_{cls}_{len(dataset):03d}_{letter}")
            dataset.append(d)
    res = f"results/prompt_bias_v2/reduce_masks/{args.name}"
    root = f"expts/prompt_bias_circuit_discovery/configs/{args.name}"
    os.makedirs(f"{res}/masks", exist_ok=True); os.makedirs(root, exist_ok=True)
    data_path = f"{res}/dataset.json"
    json.dump(dataset, open(data_path, "w"))
    # sanity: the sentence split of the first example with the skip and the excluded sink
    d0 = dataset[0]
    full = torch.tensor(d0["prompt_token_ids"] + d0["output_token_ids"][:d0["analysis_timestep"]])
    sents = split_with_prompt_chunks(full, tok, len(d0["prompt_token_ids"]), d0["prompt_chunk_spans"], 10, skip_reasoning_tokens=THINK_TOKENS)
    covered = set()
    for s in sents:
        covered.update(range(s.start, s.end + 1))
    outside = sorted(set(range(full.shape[0])) - covered)
    print("tokens outside every sentence (never masked):", [(i, tok.decode([int(full[i])])) for i in outside])
    print("prompt chunks:", d0["n_prompt_chunks"], "frozen (options):", d0["frozen_key_sentences"], [d0["prompt_chunk_kinds"][j] for j in d0["frozen_key_sentences"]],
          "| reasoning sentences:", d0["n_reasoning_sentences"], "| half cut at sentence step", d0["analysis_sentence_step_half"])
    n = 0
    for d in dataset:
        i = d["example_id"]
        common = dict(model_name=args.model_name, data_path=data_path, prompt_index=i, sentences_after_prefix=0, answer_letters=list(d["all_letters"]),
                      probe_suffix=PROBE_SUFFIX, objective="answer_probe_reward_gap", target_letter=d["other_letter"], output_dir=f"{res}/masks")
        for pool, region in [("p2t", "prompt_to_trace"), ("t2t", "trace_to_trace")]:
            for cut in ["half", "think"]:
                for sp in args.sparsities:
                    name = f"ex{i:03d}_{pool}_{cut}_s{int(round(sp * 100)):02d}"
                    cfg = dict(CANONICAL, **common, learnable_region=region, target_sparsity=sp, file_name=name)
                    if pool == "p2t":
                        cfg["frozen_key_sentences"] = d["frozen_key_sentences"]
                    if cut == "think":
                        cfg["analysis_timestep"] = d["analysis_timestep"]
                    else:
                        cfg["analysis_sentence_step"] = d["analysis_sentence_step_half"]
                    yaml.dump(cfg, open(f"{root}/{name}.yaml", "w"), default_flow_style=False, sort_keys=False)
                    n += 1
    print(f"{len(dataset)} examples ({sum(d['is_positive'] for d in dataset)} from effect inputs) -> {data_path}; {n} configs -> {root}")
    for cls, it, rec, ra, rb in chosen:
        print(f"  {cls} {rec['uid']}: admit rate Black {it['p1']:.2f} White {it['p0']:.2f} attributable {it.get('attributable')}")


if __name__ == "__main__":
    main()
