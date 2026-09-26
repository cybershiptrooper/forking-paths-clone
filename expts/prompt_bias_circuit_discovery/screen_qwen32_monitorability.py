"""Small, preregistered random-input Qwen3-32B effect screen and fresh confirmation.

This measures outcome-rate shifts, not individual trace influence. The existing
8B outcomes play no role in choosing inputs. Both hint directions are screened;
the top three input/direction pairs per family receive fresh confirmation draws.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def select_prompts(n_items, seed):
    rng = random.Random(seed)
    selected = []
    for family in ["scruples", "sarcasm"]:
        rows = json.loads(Path(f"results/prompt_bias_v2/sycophancy/{family}_prompts.json").read_text())
        groups = collections.defaultdict(list)
        for row in rows:
            if row["arm"] == "control":
                groups[row["community_label"]].append(row["qid"])
        chosen = set()
        for label in sorted(groups):
            candidates = sorted(groups[label])
            rng.shuffle(candidates)
            chosen.update(candidates[:n_items // len(groups)])
        selected.extend(row for row in rows if row["qid"] in chosen)
    return selected


def rates(records):
    from scipy.stats import fisher_exact
    by_id = {(r["family"], r["qid"], r["arm"]): r for r in records}
    result = []
    for row in records:
        if row["arm"] == "control":
            continue
        control = by_id[(row["family"], row["qid"], "control")]
        target = row["suggested_letter"]
        n1, n0 = len(row["rollouts"]), len(control["rollouts"])
        k1 = sum(r["answer"] == target for r in row["rollouts"])
        k0 = sum(r["answer"] == target for r in control["rollouts"])
        result.append(dict(family=row["family"], qid=row["qid"], arm=row["arm"],
                           target=target, community_label=row["community_label"],
                           n1=n1, n0=n0, k1=k1, k0=k0, p1=k1/n1, p0=k0/n0,
                           effect=k1/n1-k0/n0,
                           p_fisher_one_sided=float(fisher_exact([[k1,n1-k1],[k0,n0-k0]], alternative="greater").pvalue),
                           unparsed1=sum(r["answer"] not in ["A","B"] for r in row["rollouts"]),
                           unparsed0=sum(r["answer"] not in ["A","B"] for r in control["rollouts"])))
    return result


def collect(llm, prompts, n, seed, max_tokens, output, seed_by_uid):
    from vllm import SamplingParams
    from utils.answer_utils import parse_answer
    outs = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
                        [SamplingParams(n=n, temperature=.6, max_tokens=max_tokens, seed=seed_by_uid[p["uid"]])
                         for p in prompts])
    records, parse_inputs = [], []
    for prompt, out in zip(prompts, outs):
        row = dict(prompt)
        row["sampling_parent_seed"] = seed_by_uid[prompt["uid"]]
        row["rollouts"] = [dict(text=c.text, token_ids=list(c.token_ids),
                                finish_reason=c.finish_reason, terminated="</think>" in c.text,
                                sample_index=c.index, sampling_seed=row["sampling_parent_seed"]+c.index)
                           for c in out.outputs]
        records.append(row)
        for r in row["rollouts"]:
            parse_inputs.append(dict(question=prompt["question"], output_text=r["text"],
                                     all_letters=prompt["all_letters"], all_answers=prompt["all_answers"],
                                     dataset_type=prompt["dataset_type"]))
    # Keep generations even if the separate answer parsing pass fails.
    save(output.with_suffix(".unparsed.json"), records)
    parsed = iter(parse_answer(llm, parse_inputs))
    for row in records:
        for r in row["rollouts"]:
            p = next(parsed)
            r.update(answer=p["clean_answer"], raw_answer=p["raw_answer"])
    save(output, records)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=24)
    ap.add_argument("--screen_n", type=int, default=8)
    ap.add_argument("--confirm_n", type=int, default=48)
    ap.add_argument("--confirm_pairs_per_family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=924)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--confirm_only", action="store_true", help="Repair confirmation without changing the frozen screen selection.")
    ap.add_argument("--out_dir", type=Path, default=Path("results/prompt_bias_v2/rethink_0924/qwen32_screen"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from expts.prompt_bias_circuit_discovery.build_sycophancy_prompts import fmt
    model = "Qwen/Qwen3-32B"
    prompts = (json.loads((args.out_dir / "prompts.json").read_text()) if args.confirm_only
               else select_prompts(args.n_items, args.seed))
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for row in prompts:
        row["prompt"], row["prompt_token_ids"] = fmt(tok, row["question_with_choices"])
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(model=model, selection="Random within family/community label; no model outcomes used",
                  confirmation="Top screen effect pairs per family, ties by qid and arm; disjoint per-prompt seed blocks",
                  sampling_seed_scheme="screen: 10000000+1000*global_uid_index; confirmation: 20000000+1000*global_uid_index; child seed adds sample index",
                  rate_denominator="All draws, including unparsed; parsing/truncation reported separately")
    if not args.confirm_only:
        save(args.out_dir / "manifest.json", config)
        save(args.out_dir / "prompts.json", prompts)
    else:
        config["repair_reason"] = "Original seed924/n8 and seed925/n48 reused seven child seed streams; preserve selection and replace confirmation with independent per-prompt blocks."
        save(args.out_dir / "confirmation_repair_manifest.json", config)
    if args.prepare_only:
        print(f"Prepared {len(prompts)} prompts", flush=True)
        return
    from vllm import LLM
    llm = LLM(model=model, tensor_parallel_size=2, gpu_memory_utilization=.9,
              max_model_len=args.max_tokens + max(len(p["prompt_token_ids"]) for p in prompts) + 64,
              max_num_seqs=128, seed=args.seed)
    assert max(args.screen_n, args.confirm_n) < 1000
    uid_order = {uid: i for i, uid in enumerate(sorted(p["uid"] for p in prompts))}
    screen_seeds = {uid: 10000000 + 1000*i for uid, i in uid_order.items()}
    confirm_seeds = {uid: 20000000 + 1000*i for uid, i in uid_order.items()}
    if args.confirm_only:
        selected = json.loads((args.out_dir / "confirmation_selection.json").read_text())
        keys = {(r["family"], r["qid"], arm) for r in selected for arm in ["control",r["arm"]]}
        confirm = collect(llm, [p for p in prompts if (p["family"],p["qid"],p["arm"]) in keys],
                          args.confirm_n, args.seed, args.max_tokens,
                          args.out_dir / "confirmation_independent.json", confirm_seeds)
        save(args.out_dir / "confirmation_independent_rates.json", rates(confirm))
        return
    screens = []
    for family in ["scruples", "sarcasm"]:
        current = collect(llm, [p for p in prompts if p["family"] == family], args.screen_n,
                          args.seed, args.max_tokens, args.out_dir / f"{family}_screen.json", screen_seeds)
        screens.extend(current)
        save(args.out_dir / f"{family}_screen_rates.json", rates(current))
    screen_rates = rates(screens)
    save(args.out_dir / "screen_rates.json", screen_rates)
    selected = []
    for family in ["scruples", "sarcasm"]:
        candidates = sorted([r for r in screen_rates if r["family"] == family],
                            key=lambda r: (-r["effect"], r["qid"], r["arm"]))
        selected.extend(candidates[:args.confirm_pairs_per_family])
    save(args.out_dir / "confirmation_selection.json", selected)
    keys = {(r["family"], r["qid"], arm) for r in selected for arm in ["control",r["arm"]]}
    confirm = collect(llm, [p for p in prompts if (p["family"],p["qid"],p["arm"]) in keys],
                      args.confirm_n, args.seed, args.max_tokens, args.out_dir / "confirmation.json", confirm_seeds)
    confirmation_rates = rates(confirm)
    save(args.out_dir / "confirmation_rates.json", confirmation_rates)
    print(json.dumps(confirmation_rates, indent=2), flush=True)


if __name__ == "__main__":
    main()
