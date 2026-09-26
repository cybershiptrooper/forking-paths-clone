"""Generate one rollout per selected question with vLLM (Qwen3-32B).

Matches their verification-rollout generation: temperature 0.7, max 16384
tokens (their ``forced_response/task.py:296-297``), prompt = chat template +
``"<think>\n"`` (their ``build_thinking_prompt`` with empty cot_prefix).

The CoT is the generated text up to (excluding) ``</think>`` if present,
else the full generation.  Sentences are their regex splits.

Writes ``data/external_compression/rollouts/<qid>.json``:
    {question_id, cot_text, sentences, n_sentences, finish_reason,
     closed_think, gen_params}

Usage (on a GPU node):
    uv run python -m expts.external_compression.generate [--tp 2] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os

from expts.external_compression.common import (
    DATA_DIR,
    GEN_MAX_TOKENS,
    GEN_TEMPERATURE,
    MODEL_NAME,
    ROLLOUT_DIR,
    split_cot_into_sentences,
)

RENDERED_PATH = os.path.join(DATA_DIR, "prompts_rendered.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=24576)
    args = parser.parse_args()

    with open(RENDERED_PATH) as f:
        rendered = json.load(f)

    # Never regenerate an existing rollout — only fill missing ones.
    rendered = {
        q: r for q, r in rendered.items()
        if not os.path.exists(os.path.join(ROLLOUT_DIR, f"{q}.json"))
    }
    if not rendered:
        print("all rollouts already exist; nothing to do")
        return
    print(f"generating {len(rendered)} missing rollouts: {sorted(rendered)}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=0.92,
    )
    params = SamplingParams(
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
    )

    qids = sorted(rendered.keys())
    prompts = [rendered[q]["prompt_str"] for q in qids]
    outputs = llm.generate(prompts, params)

    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    for qid, out in zip(qids, outputs):
        gen_text = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        closed_think = "</think>" in gen_text
        cot_text = gen_text.split("</think>")[0] if closed_think else gen_text
        sentences = split_cot_into_sentences(cot_text)
        rec = {
            "question_id": qid,
            "cot_text": cot_text,
            "sentences": sentences,
            "n_sentences": len(sentences),
            "finish_reason": finish_reason,
            "closed_think": closed_think,
            "gen_tokens": len(out.outputs[0].token_ids),
            "gen_params": {
                "model": MODEL_NAME,
                "temperature": GEN_TEMPERATURE,
                "max_tokens": GEN_MAX_TOKENS,
                "seed": args.seed,
            },
        }
        path = os.path.join(ROLLOUT_DIR, f"{qid}.json")
        assert not os.path.exists(path), f"{path} appeared mid-run"
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        print(
            f"{qid:35s} gen_tokens={rec['gen_tokens']:6d} "
            f"sentences={rec['n_sentences']:4d} closed_think={closed_think} "
            f"finish={finish_reason}"
        )


if __name__ == "__main__":
    main()
