"""Collect the model's own reasoning on hinted prompts (vLLM, one GPU).

Samples 16 rollouts per hinted prompt at the same temperature the
original collection used (0.6), parses each rollout's final answer with
the same LLM-parser protocol as the collection pipeline
(``utils.answer_utils.parse_answer``), computes the switch rate toward
the hinted (wrong) letter, and emits collection-format records — one
per prompt whose switch rate clears --min_switch_rate — with the first
switched rollout stored as the base path.  A per-candidate report
covering ALL candidates (switched or not) is written alongside.

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.collect_hinted_rollouts \
        --candidates results/hint_removal/hinted_candidates.json \
        --output data/collection/qwen3_8b/hinted_gpqa_aqua.json \
        --report results/hint_removal/collection_report.json
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_rollouts", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--min_switch_rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from utils.answer_utils import parse_answer

    with open(args.candidates) as f:
        candidates = json.load(f)["candidates"]
    print(f"{len(candidates)} hinted candidates")

    llm = LLM(model=args.model_name, gpu_memory_utilization=0.9,
              max_model_len=args.max_tokens + 2048, seed=args.seed)
    sp = SamplingParams(n=args.n_rollouts, temperature=args.temperature,
                        max_tokens=args.max_tokens, seed=args.seed)

    prompts = [{"prompt_token_ids": c["hinted_prompt_token_ids"]}
               for c in candidates]
    outs = llm.generate(prompts, sp)

    # Flatten for the answer parser (same protocol as collection).
    parse_inputs = []
    for cand, out in zip(candidates, outs):
        for comp in out.outputs:
            parse_inputs.append({
                "question": cand["question"],
                "output_text": comp.text,
                "all_letters": cand["all_letters"],
                "all_answers": cand["all_answers"],
                "dataset_type": cand["dataset_type"],
            })
    parsed = parse_answer(llm, parse_inputs)

    records, report = [], []
    k = 0
    for cand, out in zip(candidates, outs):
        rollouts = []
        for comp in out.outputs:
            rollouts.append({
                "token_ids": list(comp.token_ids),
                "text": comp.text,
                "answer": parsed[k]["clean_answer"],
                "raw_answer": parsed[k]["raw_answer"],
                "finish_reason": comp.finish_reason,
                "terminated": "</think>" in comp.text,
            })
            k += 1
        answers = [r["answer"] for r in rollouts]
        n_switch = sum(a == cand["hint_letter"] for a in answers)
        switch_rate = n_switch / len(rollouts)
        base = next(
            (r for r in rollouts
             if r["answer"] == cand["hint_letter"] and r["terminated"]
             and r["finish_reason"] == "stop"),
            None,
        )
        rep = {
            "source_data_path": cand["source_data_path"],
            "source_prompt_index": cand["source_prompt_index"],
            "hint_letter": cand["hint_letter"],
            "correct_letter": cand["correct_letter"],
            "control_accuracy": cand["control_accuracy"],
            "control_modal_answer": cand["control_modal_answer"],
            "hinted_answers": answers,
            "switch_rate": switch_rate,
            "kept": bool(base) and switch_rate >= args.min_switch_rate,
        }
        report.append(rep)
        print(f"p{cand['source_prompt_index']:03d} "
              f"({os.path.basename(cand['source_data_path'])}): "
              f"hint={cand['hint_letter']} switch={switch_rate:.2f} "
              f"kept={rep['kept']}")
        if not rep["kept"]:
            continue
        others = [r for r in rollouts if r is not base]
        records.append({
            # collection-format fields consumed by _build_prefix & eval
            "prompt": cand["hinted_prompt"],
            "prompt_token_ids": cand["hinted_prompt_token_ids"],
            "question": cand["question"],
            "question_with_choices": cand["question_with_choices"],
            "output_token_ids": base["token_ids"],
            "output_text": base["text"],
            "clean_answer": base["answer"],
            "raw_answer": base["raw_answer"],
            "finish_reason": base["finish_reason"],
            "alternate_texts": [r["text"] for r in others],
            "alternate_answers": [r["answer"] for r in others],
            "alternate_finish_reasons": [r["finish_reason"] for r in others],
            "all_sampled_answers": answers,
            "all_letters": cand["all_letters"],
            "all_answers": cand["all_answers"],
            "correct_letter": cand["correct_letter"],
            "correct_answer": cand["correct_answer"],
            "dataset_name": cand["dataset_name"],
            "dataset_type": cand["dataset_type"],
            "base_answer_type": "stored",
            # hint-experiment metadata
            "hint_letter": cand["hint_letter"],
            "control_answer_counts": cand["control_answer_counts"],
            "control_accuracy": cand["control_accuracy"],
            "control_modal_answer": cand["control_modal_answer"],
            "source_data_path": cand["source_data_path"],
            "source_prompt_index": cand["source_prompt_index"],
            "switch_rate": switch_rate,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "report": report}, f, indent=2)
    print(f"Kept {len(records)}/{len(candidates)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
