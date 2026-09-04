"""Stopping-point screening and instance-manifest construction.

Per question, candidate prefix lengths N in {30, 50, 70, ...} (step 20,
always including n_sentences - 5 as the final point).  At each N:

- baseline: prompt + first N sentences (joined with " ") + probe;
- deletion: prompt + last 5 of those N sentences + probe;
- deletion KL = KL(baseline || deletion) over the letter distribution.

Grid points with deletion KL >= 0.1 pass the filter (their construction).
The manifest fills each bucket (<50, 50-100, 100-200, 200+ rankable
sentences, where rankable = N - 5) with up to --per_bucket passing
instances, preferring distinct questions within a bucket.

Outputs:
    data/external_compression/screening.json   (all grid points, KLs, dists)
    data/external_compression/instances.json   (selected instances)

Usage (GPU):
    uv run python -m expts.external_compression.screen
    uv run python -m expts.external_compression.screen --manifest_only
"""

from __future__ import annotations

import argparse
import json
import os

from expts.external_compression.common import (
    DATA_DIR,
    DELETION_KL_THRESHOLD,
    K_KEEP,
    MODEL_NAME,
    bucket_of,
    encode_forced,
    kl_from_distributions,
    letter_distribution,
    list_spec_ids,
    load_rollout,
    BUCKET_NAMES,
)

SCREENING_PATH = os.path.join(DATA_DIR, "screening.json")
INSTANCES_PATH = os.path.join(DATA_DIR, "instances.json")
GRID_START = 30
GRID_STEP = 20


def grid_for(n_sentences: int, fine_below: int = 0, fine_step: int = 10) -> list:
    top = n_sentences - K_KEEP
    if top < GRID_START:
        return [top] if top >= 10 else []
    pts = set(range(GRID_START, top + 1, GRID_STEP))
    pts.add(top)
    if fine_below:
        pts.update(range(GRID_START, min(top, fine_below) + 1, fine_step))
    return sorted(pts)


def forced_letter_distribution(model, tokenizer, prompt_str, sentences, answer_ids):
    import torch
    ids = encode_forced(tokenizer, prompt_str, sentences)
    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        try:
            logits = model(input_ids, logits_to_keep=1).logits
        except TypeError:
            logits = model(input_ids, num_logits_to_keep=1).logits
    return letter_distribution(logits[0, -1], answer_ids), len(ids)


def run_screening(fine_below: int = 0, fine_step: int = 10) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(DATA_DIR, "prompts_rendered.json")) as f:
        rendered = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    records = []
    done = set()
    if os.path.exists(SCREENING_PATH):
        with open(SCREENING_PATH) as f:
            records = json.load(f)
        done = {(r["question_id"], r["prefix_length"]) for r in records}
        print(f"loaded {len(records)} existing grid points; appending new ones")

    for qid in list_spec_ids():
        if not os.path.exists(os.path.join(DATA_DIR, "rollouts", f"{qid}.json")):
            print(f"{qid}: no rollout yet, skipping")
            continue
        roll = load_rollout(qid)
        sents = roll["sentences"]
        n = len(sents)
        r = rendered[qid]
        prompt_str = r["prompt_str"]
        answer_ids = r["letter_token_ids"]
        pts = [N for N in grid_for(n, fine_below, fine_step) if (qid, N) not in done]
        print(f"{qid}: n_sentences={n}, new grid points={pts}")
        for N in pts:
            base_dist, base_tokens = forced_letter_distribution(
                model, tokenizer, prompt_str, sents[:N], answer_ids,
            )
            del_dist, _ = forced_letter_distribution(
                model, tokenizer, prompt_str, sents[N - K_KEEP:N], answer_ids,
            )
            kl = kl_from_distributions(base_dist, del_dist)
            records.append({
                "question_id": qid,
                "prefix_length": N,
                "num_rankable": N - K_KEEP,
                "n_sentences": n,
                "deletion_kl": kl,
                "passes": kl >= DELETION_KL_THRESHOLD,
                "baseline_dist": base_dist,
                "deletion_dist": del_dist,
                "baseline_prefix_tokens": base_tokens,
                "letters": r["letters"],
            })
            print(
                f"  N={N:4d} rankable={N - K_KEEP:4d} tokens={base_tokens:6d} "
                f"deletion_KL={kl:.4f} {'PASS' if kl >= DELETION_KL_THRESHOLD else 'fail'}"
            )

    with open(SCREENING_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} grid points to {SCREENING_PATH}")


def build_manifest(per_bucket: int) -> None:
    with open(SCREENING_PATH) as f:
        records = json.load(f)
    passing = [r for r in records if r["passes"]]

    # Merge with an existing manifest: previously selected instances are
    # kept (jobs may already reference them); we only fill bucket deficits.
    existing = []
    if os.path.exists(INSTANCES_PATH):
        with open(INSTANCES_PATH) as f:
            existing = json.load(f)
    existing_ids = {r["instance_id"] for r in existing}
    existing_per_bucket = {name: 0 for name in BUCKET_NAMES}
    for r in existing:
        existing_per_bucket[r["bucket"]] += 1

    by_bucket = {name: [] for name in BUCKET_NAMES}
    for r in passing:
        if f"{r['question_id']}_pl{r['prefix_length']}" in existing_ids:
            continue
        by_bucket[bucket_of(r["num_rankable"])].append(r)

    selected = list(existing)
    for name in BUCKET_NAMES:
        pool = by_bucket[name]
        per_bucket_deficit = max(0, per_bucket - existing_per_bucket[name])
        # Round-robin over questions: sort each question's points by
        # rankable size descending, take one per question per round.
        by_q = {}
        for r in sorted(pool, key=lambda x: -x["num_rankable"]):
            by_q.setdefault(r["question_id"], []).append(r)
        picked = []
        round_idx = 0
        while len(picked) < per_bucket_deficit and any(by_q.values()):
            for q in sorted(by_q.keys()):
                if by_q[q] and len(picked) < per_bucket_deficit:
                    # Spread within a question: alternate ends of its list.
                    lst = by_q[q]
                    picked.append(lst.pop(0 if round_idx % 2 == 0 else -1))
            round_idx += 1
        for r in picked:
            selected.append({
                "instance_id": f"{r['question_id']}_pl{r['prefix_length']}",
                "question_id": r["question_id"],
                "prefix_length": r["prefix_length"],
                "num_rankable": r["num_rankable"],
                "bucket": name,
                "deletion_kl": r["deletion_kl"],
                "baseline_dist": r["baseline_dist"],
                "letters": r["letters"],
            })
        print(f"bucket {name:8s}: {existing_per_bucket[name]} existing + "
              f"{len(picked)} new (pool had {len(pool)} passing points)")

    with open(INSTANCES_PATH, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Wrote {len(selected)} instances to {INSTANCES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--per_bucket", type=int, default=10)
    parser.add_argument("--fine_below", type=int, default=0,
                        help="also screen every --fine_step sentences up to this N")
    parser.add_argument("--fine_step", type=int, default=10)
    args = parser.parse_args()
    if not args.manifest_only:
        run_screening(args.fine_below, args.fine_step)
    build_manifest(args.per_bucket)


if __name__ == "__main__":
    main()
