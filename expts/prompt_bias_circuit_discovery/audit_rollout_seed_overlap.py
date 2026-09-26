"""Audit historical admission RNG reuse without changing any existing selection.

Run from the repository root with:
  .venv/bin/python -m expts.prompt_bias_circuit_discovery.audit_rollout_seed_overlap

vLLM 0.11.0 assigns n-way child sample j the seed parent_seed+j. The
historical screen and confirmation therefore reuse seed streams even though
their parent seeds differ. This script records source metadata, checks exact
text/token repetition on a fixed bounded sample, and reanalyses the original
338-input confirmation universe after removing every overlapping child seed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, fisher_exact


def read(path):
    return json.loads(Path(path).read_text())


def bh(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q, 1)
    return out


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def ranges(values):
    values = sorted(set(values))
    out = []
    for value in values:
        if out and value == out[-1][1] + 1:
            out[-1][1] = value
        else:
            out.append([value, value])
    return out


def load_reports(paths, stage):
    mapping, reports = {}, []
    for path in paths:
        data = read(path)
        args = data["args"]
        uids = [r["uid"] for r in data["report"] if r["uid"].startswith("blindspot_admission_")]
        report = {"stage": stage, "report_path": str(path), "raw_path": args["output"],
                  "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "args": args, "n_admission_variants": len(uids),
                  "child_seed_range_inclusive": [args["seed"], args["seed"] + args["n_rollouts"] - 1]}
        reports.append(report)
        for uid in uids:
            if uid in mapping:
                raise ValueError(f"Ambiguous {stage} source for {uid}")
            mapping[uid] = report
    return mapping, reports


def statistics(one, zero, seeds):
    paired = [s for s in seeds if one[s] in ("A", "B") and zero[s] in ("A", "B")]
    valid_one = [one[s] for s in seeds if one[s] in ("A", "B")]
    valid_zero = [zero[s] for s in seeds if zero[s] in ("A", "B")]
    k1, k0 = valid_one.count("A"), valid_zero.count("A")
    n1, n0 = len(valid_one), len(valid_zero)
    n11 = sum(one[s] == "A" and zero[s] == "A" for s in paired)
    n10 = sum(one[s] == "A" and zero[s] == "B" for s in paired)
    n01 = sum(one[s] == "B" and zero[s] == "A" for s in paired)
    n00 = len(paired) - n11 - n10 - n01
    return {"k1": k1, "k0": k0, "n1": n1, "n0": n0,
            "p1": k1 / n1, "p0": k0 / n0, "delta": k1 / n1 - k0 / n0,
            "fisher_p": float(fisher_exact([[k1, n1-k1], [k0, n0-k0]]).pvalue),
            "mcnemar_p": float(binomtest(n10, n10+n01, .5).pvalue) if n10+n01 else 1.,
            "paired_counts": {"both_admit": n11, "only_intervention_admit": n10,
                              "only_control_admit": n01, "neither_admit": n00},
            "n_valid_pairs": len(paired), "child_seed_ranges_inclusive": ranges(seeds)}


def describe(rows):
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "mean_p1": float(np.mean([r["p1"] for r in rows])),
            "mean_p0": float(np.mean([r["p0"] for r in rows])),
            "mean_delta": float(np.mean([r["delta"] for r in rows])),
            "median_delta": float(np.median([r["delta"] for r in rows])),
            "p1_range": [min(r["p1"] for r in rows), max(r["p1"] for r in rows)],
            "gender": dict(Counter(r["gender"] for r in rows)),
            "p0_le_0.15": sum(r["p0"] <= .15 for r in rows),
            "uids": [r["uid"] for r in rows]}


def maximum_matching(positive, negative, caliper=.10):
    positive, negative = sorted(positive, key=lambda r: r["uid"]), sorted(negative, key=lambda r: r["uid"])
    if not positive or not negative:
        return {"n_pairs": 0, "pairs": []}
    cost = np.full((len(positive), len(negative)+len(positive)), 1000.)
    cost[:, :len(negative)] = 1e6
    for i, p in enumerate(positive):
        for j, n in enumerate(negative):
            d = abs(p["p1"]-n["p1"])
            if p["gender"] == n["gender"] and d <= caliper+1e-12:
                cost[i, j] = d
    pairs = []
    for i, j in zip(*linear_sum_assignment(cost)):
        if j < len(negative) and cost[i, j] < 1000:
            p, n = positive[i], negative[j]
            pairs.append({"positive_uid": p["uid"], "negative_uid": n["uid"], "gender": p["gender"],
                          "positive_p1": p["p1"], "negative_p1": n["p1"], "p1_distance": float(cost[i, j])})
    return {"n_pairs": len(pairs), "rule": "maximum cardinality, then minimum total absolute p1 gap; exact gender and p1 gap <=0.10", "pairs": pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/prompt_bias_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/prompt_bias_v2/rethink_0924/seed_overlap"))
    args = parser.parse_args()
    root = args.root
    historical_path = root / "analysis/bias_rates_qwen3_8b_admission_full_confirm.json"
    selection_path = root / "analysis/admission_same_decision_dataset_selection.json"
    historical, selection = read(historical_path)["rows"], read(selection_path)
    initial, initial_reports = load_reports(sorted((root / "rollouts_papers").glob("qwen3_8b_papers_shard*of4_report.json")), "initial_300_screen")
    later, later_reports = load_reports(sorted((root / "rollouts_papers_full").glob("qwen3_8b_admission_full_shard*of8_report.json")), "later_2200_screen")
    assert not (set(initial) & set(later)), "Initial and later screens unexpectedly overlap"
    screen = {**initial, **later}
    confirmation, confirmation_reports = load_reports(sorted((root / "rollouts_confirm_admission_full").glob("qwen3_8b_admission_full_confirm_shard*of8_report.json")), "confirmation_338")
    needed = {uid for r in historical for uid in (r["uid"], r["baseline_uid"])}
    assert needed == set(confirmation) and needed <= set(screen)
    assert len(historical) == 338 and len(needed) == 676
    # Fixed bounded audit: first five historical positive input IDs in the later screen.
    bounded = sorted((r for r in selection["positives"] if r["uid"] in later), key=lambda r: r["uid"])[:5]
    bounded_uids = {uid for r in bounded for uid in (r["uid"], r["baseline_uid"])}
    outcomes, full_bounded, provenance = {}, {}, []
    grouped = defaultdict(list)
    for uid in sorted(needed):
        grouped[confirmation[uid]["raw_path"]].append(uid)
        s, c = screen[uid], confirmation[uid]
        ss = set(range(s["args"]["seed"], s["args"]["seed"]+s["args"]["n_rollouts"]))
        cs = set(range(c["args"]["seed"], c["args"]["seed"]+c["args"]["n_rollouts"]))
        provenance.append({"uid": uid, "screen_stage": s["stage"], "screen_raw_path": s["raw_path"],
                           "screen_report_path": s["report_path"], "confirmation_raw_path": c["raw_path"],
                           "confirmation_report_path": c["report_path"],
                           "screen_child_seed_ranges": ranges(ss), "confirmation_child_seed_ranges": ranges(cs),
                           "overlapping_child_seeds": sorted(ss & cs), "retained_child_seeds": sorted(cs-ss)})
    prov = {r["uid"]: r for r in provenance}
    for path, uids in grouped.items():
        rows = read(path)
        for r in rows:
            if r["uid"] not in uids:
                continue
            seed = confirmation[r["uid"]]["args"]["seed"]
            assert len(r["rollouts"]) == confirmation[r["uid"]]["args"]["n_rollouts"]
            outcomes[r["uid"]] = {seed+j: rollout["answer"] for j, rollout in enumerate(r["rollouts"])}
            if r["uid"] in bounded_uids:
                full_bounded[r["uid"]] = r
        del rows
    exact = []
    for path in sorted({screen[uid]["raw_path"] for uid in bounded_uids}):
        rows = read(path)
        for r in rows:
            uid = r["uid"]
            if uid not in bounded_uids:
                continue
            c = full_bounded[uid]
            screen_seed, confirm_seed = screen[uid]["args"]["seed"], confirmation[uid]["args"]["seed"]
            checks = []
            for j, rollout in enumerate(r["rollouts"]):
                k = screen_seed+j-confirm_seed
                other = c["rollouts"][k]
                checks.append({"screen_index": j, "confirmation_index": k, "child_seed": screen_seed+j,
                               "same_tokens": rollout["token_ids"] == other["token_ids"],
                               "same_text": rollout["text"] == other["text"],
                               "same_answer": rollout["answer"] == other["answer"],
                               "n_tokens": len(rollout["token_ids"])})
            cross_matches = [[i, j] for i, one in enumerate(r["rollouts"])
                             for j, two in enumerate(c["rollouts"])
                             if one["token_ids"] == two["token_ids"] and one["text"] == two["text"]]
            exact.append({"uid": uid, "screen_raw_path": path, "confirmation_raw_path": confirmation[uid]["raw_path"],
                          "same_prompt": r["prompt"] == c["prompt"],
                          "same_prompt_tokens": r["prompt_token_ids"] == c["prompt_token_ids"],
                          "checks": checks, "all_exact_cross_index_matches": cross_matches})
        del rows
    old_class = {r["uid"]: label for label in ("positives", "negatives") for r in selection[label]}
    reanalysis = []
    for old in historical:
        uid, control = old["uid"], old["baseline_uid"]
        all_seeds = sorted(set(outcomes[uid]) & set(outcomes[control]))
        kept = sorted(set(prov[uid]["retained_child_seeds"]) & set(prov[control]["retained_child_seeds"]))
        full = statistics(outcomes[uid], outcomes[control], all_seeds)
        assert (full["k1"], full["n1"], full["k0"], full["n0"]) == (old["k1"], old["n1"], old["k0"], old["n0"])
        reanalysis.append({"uid": uid, "baseline_uid": control, "gender": old["gender"],
                           "historical_selection": old_class.get(uid),
                           "historical_fisher_q": old["q"], "historical_64": full,
                           "retained": statistics(outcomes[uid], outcomes[control], kept)})
    harmonic = sum(1/j for j in range(1, len(reanalysis)+1))
    for bank in ("historical_64", "retained"):
        for test in ("fisher", "mcnemar"):
            p = [r[bank][test+"_p"] for r in reanalysis]
            q, h = bh(p), holm(p)
            for r, bq, hp in zip(reanalysis, q, h):
                r[bank][test+"_bh_q"] = float(bq)
                r[bank][test+"_by_q"] = float(min(1, bq*harmonic))
                r[bank][test+"_holm_p"] = float(hp)
    old_range = [min(r["p1"] for r in selection["positives"]), max(r["p1"] for r in selection["positives"])]
    summaries = {}
    for bank in ("historical_64", "retained"):
        rows = [{**r[bank], "uid": r["uid"], "gender": r["gender"], "historical_selection": r["historical_selection"]} for r in reanalysis]
        analyses = {}
        for test in ("fisher", "mcnemar"):
            positives = [r for r in rows if r["delta"] >= .1 and r[test+"_bh_q"] < .05]
            new_range = [min(r["p1"] for r in positives), max(r["p1"] for r in positives)] if positives else None
            negatives_fixed = [r for r in rows if abs(r["delta"]) < .1 and r[test+"_p"] > .2 and old_range[0] <= r["p1"] <= old_range[1]]
            negatives_new = [r for r in rows if abs(r["delta"]) < .1 and r[test+"_p"] > .2 and new_range and new_range[0] <= r["p1"] <= new_range[1]]
            analyses[test] = {
                "positive_bh": describe(positives), "positive_by": describe([r for r in rows if r["delta"] >= .1 and r[test+"_by_q"] < .05]),
                "positive_holm": describe([r for r in rows if r["delta"] >= .1 and r[test+"_holm_p"] < .05]),
                "small_effect_old_p1_range": describe(negatives_fixed), "small_effect_new_p1_range": describe(negatives_new),
                "n_original63_retained_positive_bh": sum(r["historical_selection"] == "positives" for r in positives),
                "n_original116_retained_small_effect_old_range": sum(r["historical_selection"] == "negatives" for r in negatives_fixed),
                "n_original116_retained_small_effect_new_range": sum(r["historical_selection"] == "negatives" for r in negatives_new),
                "gender_p1_0.10_matching_new_cohorts": maximum_matching(positives, negatives_new)}
        summaries[bank] = {"all338": describe(rows), "original_cohorts": {
            label: describe([r for r in rows if r["historical_selection"] == label]) for label in ("positives", "negatives")}, "tests": analyses}
    checks = [c for r in exact for c in r["checks"]]
    shifted = [r["retained"]["delta"]-r["historical_64"]["delta"] for r in reanalysis if r["historical_selection"] == "positives"]
    summary = {
        "scope": "Retrospective source and seed audit of the original 338 admission inputs; no source data or existing selections changed.",
        "source_files": {"historical_rates": str(historical_path), "historical_selection": str(selection_path)},
        "child_seed_rule": "vLLM 0.11.0 n-way sample j uses parent seed+j; index j follows stored CompletionOutput order.",
        "screen_reports": initial_reports+later_reports, "confirmation_reports": confirmation_reports,
        "n_screen_variants_by_stage": dict(Counter(r["stage"] for r in screen.values())),
        "n_confirmation_variants_by_screen_stage": dict(Counter(r["screen_stage"] for r in provenance)),
        "n_retained_seed_count_by_variant": dict(Counter(len(r["retained_child_seeds"]) for r in provenance)),
        "bounded_exact_repeat_audit": {"selection_rule": "First five lexicographically sorted historical positive UIDs present in the later 2200-input screen, both arms.",
            "n_inputs": len(bounded), "n_variants": len(exact), "n_compared_rollouts": len(checks),
            "n_same_tokens": sum(c["same_tokens"] for c in checks), "n_same_text": sum(c["same_text"] for c in checks),
            "n_same_answer": sum(c["same_answer"] for c in checks), "n_same_prompts": sum(r["same_prompt"] for r in exact),
            "n_same_prompt_tokens": sum(r["same_prompt_tokens"] for r in exact),
            "n_all_exact_cross_index_matches": sum(len(r["all_exact_cross_index_matches"]) for r in exact)},
        "historical_positive_p1_range": old_range,
        "original63_delta_change": {"mean": float(np.mean(shifted)), "median": float(np.median(shifted)), "min": min(shifted), "max": max(shifted)},
        "summaries": summaries,
        "rules": {"positive": "delta >= 0.10 and adjusted two-sided p < 0.05 across all original 338 inputs",
                  "small_effect_candidate": "abs(delta)<0.10 and unadjusted two-sided p>0.20 and p1 within specified positive range; NOT an established causal negative",
                  "fisher": "two-sided independent-arm Fisher for historical comparability; arms use common RNG streams, so paired test is preferred",
                  "mcnemar": "exact two-sided binomial test of discordant A/B outcomes paired by child seed; invalid outputs excluded jointly",
                  "multiplicity": "BH, BY, and Holm over original 338; BY and Holm tolerate arbitrary dependence across valid p-values; BH dependence assumptions are not guaranteed",
                  "matching": "exploratory feasibility only, fresh retained-bank label rules; no cross-domain OOD or individual-rollout causal labels inferred"},
        "limitations": [
            "Child seeds also repeat across different inputs. Shared RNG pairs the arms and induces dependence across inputs.",
            "Dropping seeds removes literal overlap with each input's screen. It does not make this retrospective analysis an untouched confirmation after prior inspection of the entire 64-draw bank.",
            "Initial 300-screen jobs set max_model_len=16384; later screen and confirmation metadata use None, which their collector resolves to 9216. The exact-repeat check covers five later-screen inputs only.",
            "Marginal effect-selected inputs and low-effect candidates do not identify whether any one trace was causally influenced.",
            "A paired test corrects the within-input shared-stream design; it does not repair arbitrary data-dependent selection or all historical analysis choices."]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary.json", summary), ("variant_provenance.json", provenance),
                       ("bounded_exact_matches.json", exact), ("admission_338_reanalysis.json", reanalysis)):
        (args.out_dir/name).write_text(json.dumps(data, indent=2)+"\n")
    concise = {"exact_repeat": summary["bounded_exact_repeat_audit"],
               "source_counts": summary["n_confirmation_variants_by_screen_stage"],
               "retained_draw_counts": summary["n_retained_seed_count_by_variant"],
               "original63_delta_change": summary["original63_delta_change"], "tests": {}}
    for test, data in summaries["retained"]["tests"].items():
        concise["tests"][test] = {"positive_bh": data["positive_bh"]["n"], "positive_by": data["positive_by"]["n"],
                                 "positive_holm": data["positive_holm"]["n"], "old63_still_positive_bh": data["n_original63_retained_positive_bh"],
                                 "old116_still_small_effect": data["n_original116_retained_small_effect_old_range"],
                                 "small_effect_new_range": data["small_effect_new_p1_range"]["n"],
                                 "matched_pairs": data["gender_p1_0.10_matching_new_cohorts"]["n_pairs"]}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
