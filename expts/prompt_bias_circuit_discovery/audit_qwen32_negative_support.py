"""Audit matched low-effect candidates in the completed Qwen3-32B screen.

This CPU-only script never generates model outputs or changes source records.
Run:
    python -m expts.prompt_bias_circuit_discovery.audit_qwen32_negative_support

The parent task specified the candidate rule before this audit: |screen TE|
<=0.10, exclude every confirmed base input in both directions, match family and
hint direction, optionally match community label, use p1 calipers 0.10/0.15,
nearest distance and deterministic ties, without replacement by base input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

FIELDS = ("family", "qid", "arm", "community_label", "n1", "n0", "k1", "k0", "p1", "p0", "effect")


def brief(row):
    return {key: row[key] for key in FIELDS}


def sort_key(row):
    return row["family"], row["qid"], row["arm"]


def wilson(p, n, confidence):
    z = statistics.NormalDist().inv_cdf((1+confidence)/2)
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-radius, center+radius


def newcombe(p, n, confidence=.90):
    lo, hi = wilson(p, n, confidence)
    radius = math.hypot(p-lo, hi-p)
    return [-radius, radius]


def match(positives, low, same_community, caliper):
    used = set(); pairs = []; unmatched = []
    for positive in sorted(positives, key=sort_key):
        pool = [row for row in low
                if row["family"] == positive["family"] and row["arm"] == positive["arm"]
                and (not same_community or row["community_label"] == positive["community_label"])]
        ranked = sorted(pool, key=lambda row: (abs(row["p1"]-positive["p1"]), row["qid"], row["arm"]))
        eligible = [row for row in ranked if abs(row["p1"]-positive["p1"]) <= caliper+1e-12]
        available = [row for row in eligible if (row["family"], row["qid"]) not in used]
        record = {
            "positive": brief(positive), "n_low_effect_candidates_before_caliper": len(pool),
            "candidate_p1_histogram": dict(sorted(Counter(row["p1"] for row in pool).items())),
            "candidate_count_within_caliper": len(eligible),
            "candidate_distances": [dict(brief(row), p1_distance=abs(row["p1"]-positive["p1"]),
                                         within_caliper=abs(row["p1"]-positive["p1"]) <= caliper+1e-12)
                                    for row in ranked],
            "nearest_before_caliper": brief(ranked[0]) if ranked else None,
            "nearest_p1_difference": abs(ranked[0]["p1"]-positive["p1"]) if ranked else None,
        }
        if available:
            chosen = available[0]
            used.add((chosen["family"], chosen["qid"]))
            record["chosen"] = brief(chosen)
            record["p1_distance"] = abs(chosen["p1"]-positive["p1"])
            pairs.append(record)
        else:
            unmatched.append(record)
    return {"n_pairs": len(pairs), "pairs": pairs, "unmatched": unmatched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/prompt_bias_v2/rethink_0924/qwen32_screen"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "negative_support_audit.json"
    screen = json.loads((args.root / "screen_rates.json").read_text())
    confirmation = json.loads((args.root / "summary.json").read_text())["confirmations"]
    excluded = {(row["family"], row["qid"]) for row in confirmation}
    low = [row for row in screen if abs(row["effect"]) <= .10+1e-12
           and (row["family"], row["qid"]) not in excluded]
    strict = [row for row in confirmation if row["p_holm_6"] < .05
              and row["effect"] > .30 and row["p0"] < .15]
    significant = [row for row in confirmation if row["p_holm_6"] < .05 and row["effect"] > 0]
    result = {
        "source_files": [str(args.root / "screen_rates.json"), str(args.root / "summary.json")],
        "rule": {
            "candidate_effect": "abs(screen intervention-minus-control effect)<=0.10",
            "excluded_inputs": "Every base input with any fresh confirmation, across both hint directions",
            "required_matches": ["family", "hint direction"],
            "primary_additional_match": "community label", "sensitivity": "community unrestricted",
            "calipers": [0.10, 0.15],
            "positive_order": "family, qid, arm lexicographic",
            "candidate_order": "absolute candidate-screen-p1 minus positive-fresh-p1, then qid, arm",
            "replacement": "No replacement by family/base-input ID within each matching run",
            "strict_positive": "Fresh Holm-6 p<0.05, effect>0.30, p0<0.15 (existing large/low-baseline criterion)",
            "secondary_positive": "All fresh Holm-6 p<0.05 and positive-effect confirmations",
        },
        "excluded_base_inputs": [list(x) for x in sorted(excluded)],
        "n_low_screen_rows": len(low),
        "n_low_screen_unique_inputs": len({(r["family"], r["qid"]) for r in low}),
        "low_screen_p1_histogram": dict(sorted(Counter(r["p1"] for r in low).items())),
        "low_screen_candidates": sorted([brief(row) for row in low], key=sort_key),
        "matching": {},
        "hypothetical_n48_null_90pct_difference_intervals": {
            str(p): newcombe(p, 48) for p in (.25, .375, .5, .625, .75)},
        "n8_marginal_p1_wilson_95pct_intervals": {"0_of_8": wilson(0, 8, .95), "8_of_8": wilson(1, 8, .95)},
        "interpretation": [
            "With eight draws per arm, |screen TE|<=0.10 requires exactly equal observed counts.",
            "Candidate labels and p1 values are screen estimates, not independently confirmed small effects.",
            "Zero sample matches does not establish absence of common support in the population.",
            "A new 48-draw confirmation would require an eligible candidate; this audit does not relax the declared matching rule.",
            "At middle outcome rates, even exactly equal counts with n=48 per arm generally cannot establish +/-0.10 equivalence.",
            "The audit selects no mask and does not optimize any mask against counterfactual outcomes.",
        ],
    }
    for name, positives in (("strict_large_lowbaseline", strict), ("all_positive_holm", significant)):
        result["matching"][name] = {}
        for same in (True, False):
            condition = "same_community" if same else "community_unrestricted"
            result["matching"][name][condition] = {
                str(caliper): match(positives, low, same, caliper) for caliper in (.10, .15)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    summary = {name: {condition: {caliper: res["n_pairs"] for caliper, res in detail.items()}
                      for condition, detail in groups.items()} for name, groups in result["matching"].items()}
    print(json.dumps({"output": str(output), "n_low_screen_rows": len(low),
                      "low_screen_p1_histogram": result["low_screen_p1_histogram"],
                      "matched_pairs": summary}, indent=2))


if __name__ == "__main__":
    main()
