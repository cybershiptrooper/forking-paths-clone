"""Build termination candidate banks from the scan outputs (CPU only).

For each selected (prompt, analysis step) pair, splits the scan's
sampled continuations into a training bank and a held-out bank
(stratified by cluster so every cluster present in the data is
represented in both halves), and writes a bank JSON in the schema
consumed by ``learn.py``'s ``candidate_*`` objectives:

    cluster 0 — terminated within the horizon with the correct answer
                (the target cluster)
    cluster 1 — terminated, wrong or unparseable answer
    cluster 2 — did not terminate within the horizon

The clean model's per-candidate log-probabilities are NOT stored: the
subnetwork-probing trainer computes them itself at the start of
training.

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.build_termination_bank \
        --scan_dir results/cot_termination/scan \
        --selection results/cot_termination/selection.json \
        --n_train 16 \
        --output_dir results/cot_termination/banks
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

PROBE_SUFFIX = " </think> I think the answer is"
NUM_CLUSTERS = 3
TARGET_CLUSTER = 0


def stratified_split(samples, n_train, rng):
    """Split samples into (train, heldout), stratified by cluster_id."""
    by_cluster = {}
    for s in samples:
        by_cluster.setdefault(s["cluster_id"], []).append(s)
    train, heldout = [], []
    frac = n_train / len(samples)
    for cid in sorted(by_cluster):
        group = list(by_cluster[cid])
        rng.shuffle(group)
        k = int(round(frac * len(group)))
        # every non-empty cluster contributes at least one to each half
        # (when it has at least 2 members)
        k = max(1, min(k, len(group) - 1)) if len(group) >= 2 else len(group)
        train.extend(group[:k])
        heldout.extend(group[k:])
    return train, heldout


def to_candidate(sample):
    """Terminated candidates are truncated at their ``</think>`` token —
    the termination event is fully contained in the truncated sequence,
    the post-think prose is irrelevant to the reward, and shorter
    candidates make teacher-forcing cheaper."""
    ids = sample["token_ids"]
    if sample["terminated"]:
        ids = ids[: sample["think_pos"] + 1]
    return {
        "continuation_token_ids": ids,
        "cluster_id": sample["cluster_id"],
        "count": 1,
        "is_correct": sample["cluster_id"] == TARGET_CLUSTER,
        "grade_method": "probe" if sample["terminated"] else "none",
        "answer_text": (
            f"probe={sample['probe_label']} regex={sample['regex_answer']}"
            if sample["terminated"] else "(not terminated)"
        ),
        "terminated": sample["terminated"],
        "n_tokens": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    with open(args.selection) as f:
        selected = json.load(f)["selected"]

    scans = {}
    for path in glob.glob(os.path.join(args.scan_dir, "scan_*.json")):
        with open(path) as f:
            scan = json.load(f)
        scans[scan["data_path"]] = {
            (r["prompt_index"], r["analysis_sentence_step"]): r
            for r in scan["results"]
        }

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    for sel in selected:
        key = (sel["prompt_index"], sel["analysis_sentence_step"])
        rec = scans[sel["data_path"]][key]
        rng = random.Random(args.seed + sel["prompt_index"])
        train, heldout = stratified_split(rec["samples"], args.n_train, rng)
        n_target_train = sum(s["cluster_id"] == TARGET_CLUSTER for s in train)
        if n_target_train == 0:
            print(f"SKIP p{sel['prompt_index']} step{key[1]}: "
                  f"no target-cluster candidate in train half")
            continue
        stem = os.path.splitext(os.path.basename(sel["data_path"]))[0]
        name = f"{stem}_p{sel['prompt_index']:03d}_s{key[1]}"
        bank = {
            "bank_type": "termination",
            "data_path": sel["data_path"],
            "prompt_index": sel["prompt_index"],
            "analysis_sentence_step": key[1],
            "sentences_after_prefix": 0,
            "probe_suffix": PROBE_SUFFIX,
            "horizon": max(s["n_tokens"] for s in rec["samples"]),
            "num_clusters": NUM_CLUSTERS,
            "target_cluster": TARGET_CLUSTER,
            "gold_answer_normalized": rec.get("correct_letter") or "",
            "trace_answer": rec.get("trace_answer"),
            "all_letters": rec.get("all_letters"),
            "clean_fraction_correct": (
                rec["n_target_cluster"] / rec["n_samples"]
            ),
            "clean_fraction_terminated": rec["f_terminated"],
            "candidates": [to_candidate(s) for s in train],
            "heldout_candidates": [to_candidate(s) for s in heldout],
        }
        out_path = os.path.join(args.output_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(bank, f)
        counts = [sum(s["cluster_id"] == c for s in train) for c in range(3)]
        print(f"{name}: train clusters {counts}, heldout {len(heldout)}")
        manifest.append({
            "bank_path": out_path,
            "data_path": sel["data_path"],
            "prompt_index": sel["prompt_index"],
            "analysis_sentence_step": key[1],
            "prefix_len": sel["prefix_len"],
            "f_terminated": sel["f_terminated"],
        })
    man_path = os.path.join(args.output_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} banks + {man_path}")


if __name__ == "__main__":
    main()
