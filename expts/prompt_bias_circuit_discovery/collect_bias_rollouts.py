"""Sample the model's own reasoning on every prompt variant (vLLM, one GPU).

For each prompt in the builder's output, samples ``--n_rollouts``
completions at the collection temperature, parses the final letter with
the collection pipeline's LLM parser (``utils.answer_utils.parse_answer``)
and stores every rollout (token ids, text, parsed letter, finish reason).
The selection script turns this raw file into collection-format records.

By default, every prompt receives a disjoint block of sampling seeds, indexed
before sharding. Use a different nonnegative ``--seed`` for each collection
phase (for example, screening and confirmation), and keep the concatenated
prompt order fixed across shards. ``--seed_scheme legacy`` reproduces the old
shared parent seed; its streams overlap across prompts and may overlap phases.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.collect_bias_rollouts \
        --prompts results/prompt_bias/prompts.json \
        --output results/prompt_bias/rollouts_raw.json \
        --report results/prompt_bias/rollouts_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os


PROMPT_SEED_STRIDE = 10_000
ROOT_SEED_STRIDE = 1_000_000_000
MAX_PROMPTS = ROOT_SEED_STRIDE // PROMPT_SEED_STRIDE


def sampling_parent_seeds(root_seed: int, n_all: int, n_rollouts: int,
                          seed_scheme: str = "disjoint") -> list[int]:
    """Allocate vLLM parent seeds; child sample i uses parent_seed + i.

    Global prompt indices refer to the concatenated, unsharded prompt files.
    Distinct root seeds reserve disjoint billion-seed collection blocks.
    """
    if seed_scheme not in ("disjoint", "legacy"):
        raise ValueError(f"Unknown seed scheme: {seed_scheme}")
    if root_seed < 0:
        raise ValueError("--seed must be nonnegative")
    if not 0 <= n_all <= MAX_PROMPTS:
        raise ValueError(f"The collection must contain at most {MAX_PROMPTS} prompts")
    if not 1 <= n_rollouts <= PROMPT_SEED_STRIDE:
        raise ValueError(f"--n_rollouts must be between 1 and {PROMPT_SEED_STRIDE}")
    first_parent = ((root_seed + 1) * ROOT_SEED_STRIDE
                    if seed_scheme == "disjoint" else root_seed)
    last_parent = (first_parent + max(n_all - 1, 0) * PROMPT_SEED_STRIDE
                   if seed_scheme == "disjoint" else first_parent)
    if last_parent + n_rollouts - 1 >= 2**63:
        raise ValueError("The allocated sampling seeds must be smaller than 2**63")
    if seed_scheme == "legacy":
        return [root_seed] * n_all
    return [first_parent + i * PROMPT_SEED_STRIDE for i in range(n_all)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, nargs="+",
                    help="One or more prompt files (concatenated in order).")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--shard", type=int, default=0, help="Take prompts i with i %% n_shards == shard.")
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=None, help="default: max_tokens + 1024 (short prompts)")
    ap.add_argument("--n_rollouts", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42,
                    help="Nonnegative collection root seed; use different roots for different phases.")
    ap.add_argument("--seed_scheme", choices=("disjoint", "legacy"), default="disjoint",
                    help="Disjoint per-prompt blocks (default), or legacy shared sampling streams.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    prompts = []
    for path in args.prompts:
        with open(path) as f:
            prompts.extend(json.load(f))
    n_all = len(prompts)
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        ap.error("--n_shards must be positive and --shard must be in [0, n_shards)")
    try:
        all_parent_seeds = sampling_parent_seeds(args.seed, n_all, args.n_rollouts,
                                                args.seed_scheme)
    except ValueError as exc:
        ap.error(str(exc))
    global_indices = [i for i in range(n_all) if i % args.n_shards == args.shard]
    parent_seeds = [all_parent_seeds[i] for i in global_indices]
    prompts = [prompts[i] for i in global_indices]
    print(f"{len(prompts)} of {n_all} prompt variants (shard {args.shard}/{args.n_shards})")

    from vllm import LLM, SamplingParams
    from utils.answer_utils import parse_answer

    llm = LLM(model=args.model_name, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len or (args.max_tokens + 1024), seed=args.seed,
              tensor_parallel_size=args.tensor_parallel_size)
    sp = [SamplingParams(n=args.n_rollouts, temperature=args.temperature,
                         max_tokens=args.max_tokens, seed=parent_seed)
          for parent_seed in parent_seeds]
    outs = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts], sp)

    parse_inputs = []
    for p, out in zip(prompts, outs):
        for comp in out.outputs:
            parse_inputs.append({
                "question": p["question"], "output_text": comp.text,
                "all_letters": p["all_letters"], "all_answers": p["all_answers"],
                "dataset_type": p["dataset_type"],
            })
    parsed = parse_answer(llm, parse_inputs)

    records, report = [], []
    k = 0
    for p, out, global_index, parent_seed in zip(prompts, outs, global_indices, parent_seeds):
        rollouts = []
        for comp in out.outputs:
            rollouts.append({
                "token_ids": list(comp.token_ids), "text": comp.text,
                "answer": parsed[k]["clean_answer"], "raw_answer": parsed[k]["raw_answer"],
                "finish_reason": comp.finish_reason,
                "terminated": "</think>" in comp.text,
                "sample_index": comp.index,
                "sampling_seed": parent_seed + comp.index,
            })
            k += 1
        answers = [r["answer"] for r in rollouts]
        counts = collections.Counter(answers)
        rec = dict(p)
        rec["global_prompt_index"] = global_index
        rec["sampling_parent_seed"] = parent_seed
        rec["sampling_seed_scheme"] = args.seed_scheme
        rec["rollouts"] = rollouts
        records.append(rec)
        rep = {k2: p[k2] for k2 in ("uid", "setting", "profile_id", "axis", "value")}
        rep["answer_counts"] = dict(counts)
        rep["p_A"] = counts.get("A", 0) / len(answers)
        rep["mean_tokens"] = sum(len(r["token_ids"]) for r in rollouts) / len(rollouts)
        rep["n_terminated"] = sum(r["terminated"] for r in rollouts)
        report.append(rep)
        print(f"{p['uid']:36s} P(A)={rep['p_A']:.2f} counts={dict(counts)} "
              f"tokens={rep['mean_tokens']:.0f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f)
    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "report": report}, f, indent=1)
    print(f"Wrote {len(records)} variants -> {args.output}")


if __name__ == "__main__":
    main()
