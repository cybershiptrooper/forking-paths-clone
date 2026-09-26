"""Prepare 64 retrospective, intervention-only monitor candidates without fitting.

Run:
    python -m expts.prompt_bias_circuit_discovery.build_monitor_pilot_candidates

Select 16 complete matched pairs per split by a fixed hash, after excluding all
current admission mask-development inputs and their partners. Preserve the
previously hash-selected admit trace. Selection never consults token lengths,
mask outcomes, judge predictions, rates, or the sign of either pair member's
label. Matching upstream used labels and intervention rates; these candidates
are retrospective and are not newly untouched confirmatory examples.

Actor packets use an explicit field allowlist. Susceptibility labels, rates,
splits, source IDs, and pair IDs live only in separate analysis artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

SEED = 20260924
ACTOR_FIELDS = ("question", "question_with_choices", "prompt", "prompt_token_ids",
                "output_text", "output_token_ids", "all_letters", "all_answers")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def base_id(row):
    return row["uid"].split("_")[2]


def quantiles(values):
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered)-1)*q
        lo = int(index); hi = min(lo+1, len(ordered)-1)
        return ordered[lo]+(index-lo)*(ordered[hi]-ordered[lo])
    return {"n": len(ordered), "min": ordered[0], "p25": percentile(.25),
            "median": percentile(.50), "p75": percentile(.75), "p90": percentile(.90),
            "p95": percentile(.95), "max": ordered[-1], "mean": statistics.mean(ordered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/prompt_bias_v2"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pairs_per_split", type=int, default=16)
    parser.add_argument("--out", type=Path,
                        default=Path("results/prompt_bias_v2/rethink_0924/monitor_pilot_candidates"))
    args = parser.parse_args()
    sources = {
        "matched_and_original_splits": args.root / "rethink_0924/shortcut_split_manifest.json",
        "mask_development": args.root / "reduce_masks/reduce_masks_qwen3_8b/dataset.json",
        "id_actor_traces": args.root / "judge_sets_same_decision/qwen3_8b_admission_same_decision.json",
        "ood_actor_traces": args.root / "judge_sets_same_decision/qwen3_8b_admission_same_decision_nativeamerican.json",
    }
    split_source = read(sources["matched_and_original_splits"])
    matched = split_source["matched_gender_p1_0.10"]
    development_ids = {base_id(row) for row in read(sources["mask_development"])}
    assert len(development_ids) == 8, "Revisit the exclusion specification if the development cohort changes."
    black179 = {row["input"] for row in split_source["original"]["id"]}
    assert len(black179) == 179
    actor_source = {scope: {row["tag"]: row for row in read(sources[f"{scope}_actor_traces"])}
                    for scope in ("id", "ood")}
    packets = []; label_rows = []; provenance = []; availability = {}; selections = {}
    for scope in ("id", "ood"):
        groups = defaultdict(list)
        for row in matched[scope]:
            groups[row["group"]].append(row)
        eligible = []; excluded = []
        for group, members in groups.items():
            assert len(members) == 2 and sorted(r["label"] for r in members) == [0, 1]
            member_ids = sorted(r["input"] for r in members)
            if scope == "id" and development_ids.intersection(member_ids):
                excluded.append({"source_group": group, "inputs": member_ids,
                                 "development_inputs": sorted(development_ids.intersection(member_ids))})
                continue
            if scope == "ood":
                assert not black179.intersection(member_ids), "OOD must exclude every Black179 application."
            pair_hash = digest(f"{args.seed}|pair|{scope}|"+"|".join(member_ids))
            eligible.append((pair_hash, group, members))
        eligible.sort(key=lambda item: item[0])
        assert len(eligible) >= args.pairs_per_split, f"Not enough whole {scope} pairs."
        chosen = eligible[:args.pairs_per_split]
        availability[scope] = {"original_pairs": len(groups), "excluded_pairs": len(excluded),
                               "eligible_pairs": len(eligible), "selected_pairs": len(chosen),
                               "selected_inputs": 2*len(chosen), "excluded_pair_records": excluded}
        selections[scope] = []
        for pair_hash, source_group, members in chosen:
            pair_id = "pair_"+digest(f"{args.seed}|analysis_pair|{pair_hash}")[:20]
            selected_pair = {"pair_id": pair_id, "selection_hash": pair_hash,
                             "source_group": source_group, "inputs": sorted(r["input"] for r in members),
                             "sample_ids": []}
            for selected in members:
                assert selected["input"] not in development_ids
                source = actor_source[scope][selected["chosen_tag"]]
                assert base_id(source) == selected["input"]
                assert int(source["is_positive"]) == selected["label"]
                assert source["clean_answer"] == "A"
                sample_id = "sample_"+digest(f"{args.seed}|sample|{selected['input']}|{source['tag']}")[:24]
                packet = {"sample_id": sample_id, **{key: source[key] for key in ACTOR_FIELDS}}
                assert set(packet) == {"sample_id", *ACTOR_FIELDS}
                packets.append(packet)
                label_rows.append({"sample_id": sample_id, "split": scope, "pair_id": pair_id,
                                   "input_effect_label": selected["label"],
                                   "p_intervention": selected["p1"], "p_control": selected["p0"],
                                   "estimated_effect": selected["p1"]-selected["p0"],
                                   "label_interpretation": "Large-effect versus small-estimated-effect input; not an identified label of individual causal influence."})
                provenance.append({"sample_id": sample_id, "split": scope, "pair_id": pair_id,
                                   "source_input": selected["input"], "source_tag": source["tag"],
                                   "source_file": str(sources[f"{scope}_actor_traces"]),
                                   "prompt_tokens": len(source["prompt_token_ids"]),
                                   "output_tokens": len(source["output_token_ids"]),
                                   "total_actor_tokens": len(source["prompt_token_ids"])+len(source["output_token_ids"])})
                selected_pair["sample_ids"].append(sample_id)
            selections[scope].append(selected_pair)
    assert len({row["sample_id"] for row in packets}) == len(packets) == 4*args.pairs_per_split
    assert len({row["source_input"] for row in provenance}) == len(packets)
    for scope in ("id", "ood"):
        labels = [row["input_effect_label"] for row in label_rows if row["split"] == scope]
        assert len(labels) == 2*args.pairs_per_split and sum(labels) == args.pairs_per_split
    # Combine scopes and classes; neither packet ID nor order encodes a label or split.
    packets.sort(key=lambda row: digest(f"{args.seed}|packet_order|{row['sample_id']}"))
    label_rows.sort(key=lambda row: row["sample_id"])
    provenance.sort(key=lambda row: row["sample_id"])
    lengths = {scope: {key: quantiles([row[key] for row in provenance if scope == "all" or row["split"] == scope])
                       for key in ("prompt_tokens", "output_tokens", "total_actor_tokens")}
               for scope in ("id", "ood", "all")}
    manifest = {
        "status": "Historical preparation only; not eligible for judge calls, mask fitting, training, or optimization launch.",
        "ineligibility_reason": "The seed-overlap audit invalidates the historical fresh-confirmation premise of these labels. After removing reused child seeds, only 7 near-OOD pairs satisfy the retained-bank proxy rules and exact-gender/p1-matching criterion, below the 16 required here. This manifest preserves the original candidate selection, actor packets, and evaluation labels for provenance; it does not authorize launching that cohort. Evidence: rethink_0924/seed_overlap/nativeamerican_summary.json (original_native58_disjoint, retained bank, original 1133-test family).",
        "scope": "Retrospective intervention-only, same-answer input-susceptibility pilot; not a fresh confirmatory test.",
        "planned_mask_pool": "P_to_R; one recipe must be frozen before any later monitor test.",
        "seed": args.seed, "pairs_per_split": args.pairs_per_split,
        "selection": "Sort complete matched pairs by SHA256(seed|pair|scope|sorted base IDs); take first N. Never use lengths, model predictions, mask outcomes, rates, or member label signs to rank pairs.",
        "trace_selection": "Preserve chosen_tag from shortcut_split_manifest.json, already selected by SHA256(seed:base_id) among four stored admits.",
        "upstream_matching": "Exact gender, estimated intervention p1 caliper 0.10; upstream matching did use proxy labels and rates.",
        "development_inputs": sorted(development_ids),
        "development_inputs_absent_from_matched_id_pool": sorted(development_ids-{r["input"] for r in matched["id"]}),
        "ood_excludes_all_black179_inputs": sorted(black179),
        "sources": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for key, path in sources.items()},
        "availability": availability, "selected_pairs": selections, "samples": provenance,
        "token_lengths": lengths, "output_length_note": "Output token counts include the original final-answer suffix.",
        "actor_packet_fields": ["sample_id", *ACTOR_FIELDS],
        "information_separation": "Only actor_traces.json is eligible as monitor input; evaluation_labels.json and split_source_manifest.json are analysis-only. Actual factual prompt content is retained by design.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "actor_traces.json").write_text(json.dumps(packets, ensure_ascii=False)+"\n")
    (args.out / "evaluation_labels.json").write_text(json.dumps(label_rows, indent=2)+"\n")
    (args.out / "split_source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"out": str(args.out), "n_packets": len(packets), "availability": availability,
                      "output_token_lengths": {scope: values["output_tokens"] for scope, values in lengths.items()}}, indent=2))


if __name__ == "__main__":
    main()
