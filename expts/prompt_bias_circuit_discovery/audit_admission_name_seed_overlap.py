"""Extend the seed-overlap audit to the original 1,133 name contrasts.

  .venv/bin/python -m expts.prompt_bias_circuit_discovery.audit_admission_name_seed_overlap

Native American statistics retain the original 1,133-test multiplicity family.
No historical selection or actor data is overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from .audit_rollout_seed_overlap import (bh, describe, holm, load_reports,
                                        maximum_matching, read, statistics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/prompt_bias_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/prompt_bias_v2/rethink_0924/seed_overlap"))
    args = parser.parse_args()
    root = args.root
    history_path = root/"analysis/bias_rates_qwen3_8b_admission_all_confirm.json"
    history = read(history_path)["rows"]
    native = read(root/"analysis/admission_same_decision_selection_nativeamerican.json")
    black = read(root/"analysis/admission_same_decision_dataset_selection.json")
    exclude_ids = {r["uid"].split("_")[2] for group in ("positives", "negatives") for r in black[group]}
    selected = {r["uid"]: label for label in ("positives", "negatives") for r in native[label]}
    confirmation, creports = load_reports(sorted(root.glob("rollouts_confirm_admission_*/*shard*of[0-9]_report.json")), "all_name_confirmation")
    initial, ireports = load_reports(sorted((root/"rollouts_papers").glob("qwen3_8b_papers_shard*of4_report.json")), "initial_300_screen")
    later, lreports = load_reports(sorted((root/"rollouts_papers_full").glob("qwen3_8b_admission_full_shard*of8_report.json")), "later_2200_screen")
    names, nreports = load_reports(sorted((root/"rollouts_papers_full").glob("qwen3_8b_admission_names_screen_shard*of8_report.json")), "name_screen")
    assert not (set(initial)&set(later) or set(initial)&set(names) or set(later)&set(names))
    screen = {**initial, **later, **names}
    needed = {u for r in history for u in (r["uid"], r["baseline_uid"])}
    assert len(history)==1133 and len(needed)==1607
    assert needed <= set(confirmation) and needed <= set(screen)
    outcomes, provenance = {}, {}
    for uid in sorted(needed):
        s, c = screen[uid], confirmation[uid]
        ss = set(range(s["args"]["seed"], s["args"]["seed"]+s["args"]["n_rollouts"]))
        cs = set(range(c["args"]["seed"], c["args"]["seed"]+c["args"]["n_rollouts"]))
        provenance[uid] = {"uid": uid, "screen_report": s["report_path"], "screen_raw": s["raw_path"],
                           "confirmation_report": c["report_path"], "confirmation_raw": c["raw_path"],
                           "overlap_seeds": sorted(ss&cs), "retained_seeds": sorted(cs-ss)}
    for path in sorted({confirmation[uid]["raw_path"] for uid in needed}):
        records = read(path)
        for r in records:
            uid = r["uid"]
            if uid in needed:
                seed = confirmation[uid]["args"]["seed"]
                assert len(r["rollouts"]) == confirmation[uid]["args"]["n_rollouts"]
                outcomes[uid] = {seed+j: rollout["answer"] for j, rollout in enumerate(r["rollouts"])}
        del records
    rows = []
    for old in history:
        uid, control = old["uid"], old["baseline_uid"]
        full = statistics(outcomes[uid], outcomes[control], sorted(set(outcomes[uid])&set(outcomes[control])))
        assert (full["k1"],full["n1"],full["k0"],full["n0"]) == (old["k1"],old["n1"],old["k0"],old["n0"])
        retained = statistics(outcomes[uid], outcomes[control], sorted(set(provenance[uid]["retained_seeds"])&set(provenance[control]["retained_seeds"])))
        rows.append({"uid": uid,"baseline_uid": control,"race": old["race"],"gender": old["gender"],
                     "historical_native_class": selected.get(uid),"disjoint_from_all_black179": uid.split("_")[2] not in exclude_ids,
                     "historical_q": old["q"],"historical_64": full,"retained": retained})
    harmonic = sum(1/j for j in range(1,len(rows)+1))
    for bank in ("historical_64","retained"):
        for test in ("fisher","mcnemar"):
            ps = [r[bank][test+"_p"] for r in rows]
            for r,q,h in zip(rows,bh(ps),holm(ps)):
                r[bank][test+"_bh_q"] = float(q)
                r[bank][test+"_by_q"] = min(1.,float(q)*harmonic)
                r[bank][test+"_holm_p"] = float(h)
    assert max(abs(r["historical_q"]-r["historical_64"]["fisher_bh_q"]) for r in rows) < 1e-12
    old_range = [min(r["p1"] for r in native["positives"]),max(r["p1"] for r in native["positives"])]
    summaries = {}
    for scope in ("all_original_native78","original_native58_disjoint","all_native122","all_native_disjoint"):
        scoped = [r for r in rows if r["race"]=="Native American"
                  and (not scope.startswith("all_original") and not scope.startswith("original_") or r["historical_native_class"])
                  and (not scope.endswith("disjoint") or r["disjoint_from_all_black179"])]
        summary = {"n":len(scoped),"banks":{}}
        for bank in ("historical_64","retained"):
            current = [{**r[bank],"uid":r["uid"],"gender":r["gender"],"historical_class":r["historical_native_class"]} for r in scoped]
            detail = {"all":describe(current),"original_classes":{k:describe([r for r in current if r["historical_class"]==k]) for k in ("positives","negatives")},"tests":{}}
            for test in ("fisher","mcnemar"):
                pos = [r for r in current if r["delta"]>=.1 and r[test+"_bh_q"]<.05]
                neg = [r for r in current if abs(r["delta"])<.1 and r[test+"_p"]>.2 and old_range[0]<=r["p1"]<=old_range[1]]
                detail["tests"][test] = {"positive_bh":describe(pos),
                    "positive_by":describe([r for r in current if r["delta"]>=.1 and r[test+"_by_q"]<.05]),
                    "positive_holm":describe([r for r in current if r["delta"]>=.1 and r[test+"_holm_p"]<.05]),
                    "small_effect_fixed_historical_p1_range":describe(neg),
                    "old_positive_still_positive_bh":sum(r["historical_class"]=="positives" for r in pos),
                    "old_negative_still_small_effect":sum(r["historical_class"]=="negatives" for r in neg),
                    "gender_p1_0.10_matching":maximum_matching(pos,neg),
                    "original_classes_only_stable_matching":maximum_matching([r for r in pos if r["historical_class"]=="positives"],[r for r in neg if r["historical_class"]=="negatives"])}
            summary["banks"][bank]=detail
        for label in ("positives","negatives"):
            old = [r for r in scoped if r["historical_native_class"]==label]
            summary[label+"_point_stability"]={"n":len(old),"retained_delta_ge_0.1":sum(r["retained"]["delta"]>=.1 for r in old),
                "retained_abs_delta_lt_0.1":sum(abs(r["retained"]["delta"])<.1 for r in old),
                "retained_positive_sign":sum(r["retained"]["delta"]>0 for r in old)}
        summaries[scope]=summary
    result={"scope":"Retrospective seed-disjoint subset of historical name bank, using all original 1133 contrasts for multiplicity correction.",
        "source_rates":str(history_path),"screen_reports":ireports+lreports+nreports,"confirmation_reports":creports,
        "n_tests":len(rows),"n_confirmation_variants":len(needed),"n_retained_draws":dict(Counter(r["retained"]["n_valid_pairs"] for r in rows)),
        "historical_native_positive_p1_range":old_range,"summaries":summaries,
        "limitations":["Original historical bank has already been inspected; retained 48 draws are retrospective, not untouched confirmation.",
            "Shared seeds across inputs violate a general guarantee for BH; BY and Holm sensitivity remain available.",
            "Low-effect proxy rules do not establish equivalence or individual causal absence.",
            "Near-OOD exclusion removes every one of the original Black179 base inputs. No Native American source text/token exact-duplicate check is performed here; seed overlap is established from metadata.",
            "Native American historical q values use 1133 tests; original Black-only audit uses its original 338-test family. Do not conflate the two families."]}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,data in (("name_1133_reanalysis.json",rows),("name_variant_provenance.json",list(provenance.values())),("nativeamerican_summary.json",result)):
        (args.out_dir/name).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:{"n":v["n"],"points":v["positives_point_stability"],"tests":{t:{"pos_bh":z["positive_bh"]["n"],"pos_by":z["positive_by"]["n"],"pos_holm":z["positive_holm"]["n"],"small_effect":z["small_effect_fixed_historical_p1_range"]["n"],"stable_pos":z["old_positive_still_positive_bh"],"stable_neg":z["old_negative_still_small_effect"],"matches":z["gender_p1_0.10_matching"]["n_pairs"],"stable_original_matches":z["original_classes_only_stable_matching"]["n_pairs"]} for t,z in v["banks"]["retained"]["tests"].items()}} for k,v in summaries.items()},indent=2))


if __name__=="__main__":
    main()
