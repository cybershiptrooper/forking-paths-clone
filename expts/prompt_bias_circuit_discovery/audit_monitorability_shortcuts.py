"""Measure nuisance prediction and common support in intervention-only pools.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1:
    .venv/bin/python -m expts.prompt_bias_circuit_discovery.audit_monitorability_shortcuts

The target is the existing input-level effect-selection label, NOT identified
causation of one rollout. One deterministic stored admit per input supplies CoT
length; every other feature is input-level. Numeric p1/entropy recipes require
an intervention rollout bank and are privileged diagnostics, not deployable
single-trace monitors. No recipe uses p0, total effect, or confirmation p/q.

Recipes and regularization are fixed. Five outer input/group folds estimate ID
performance; inner folds choose only a threshold maximizing ordinary TPR*TNR.
For OOD, train on all Black-name inputs and freeze the threshold chosen from ID
out-of-fold predictions. Native American OOD excludes EVERY base application
in the original Black179 set, even for the matched-cohort experiment.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RECIPES = {
    "gender": ["female"],
    "prompt_length": ["log_prompt_tokens"],
    "gender_prompt_length": ["female", "log_prompt_tokens"],
    "cot_length": ["log_cot_tokens"],
    "gender_prompt_cot_lengths": ["female", "log_prompt_tokens", "log_cot_tokens"],
    "prompt_tfidf_redacted": None,
    "p1_oracle": ["p1"],
    "p1_entropy_oracle": ["p1", "p1_entropy"],
}
SEED = 20260924


def base_id(row):
    return row.get("qid") or row["uid"].split("_")[2]


def entropy(p):
    return -sum(v * math.log(v) for v in (p, 1-p) if v > 0)


def load_admission(path, seed):
    groups = defaultdict(list)
    for row in json.loads(path.read_text()):
        groups[base_id(row)].append(row)
    result = []
    for iid, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["tag"])
        # The index uses only the base input ID and pinned seed, never its label.
        idx = int(hashlib.sha256(f"{seed}:{iid}".encode()).hexdigest()[:8], 16) % len(rows)
        r = rows[idx]
        assert all(x["is_positive"] == r["is_positive"] for x in rows)
        assert all(x["clean_answer"] == "A" for x in rows)
        prompt = re.sub(re.escape(r["name"]), "[APPLICANT]", r["question_with_choices"], flags=re.I)
        cot = r["output_text"].split("</think>")[0].replace("<think>", "")
        result.append({"input": iid, "group": iid, "label": int(r["is_positive"]),
                       "gender": r["gender"], "female": int(r["gender"].lower() == "female"),
                       "prompt": prompt, "prompt_tokens": len(r["prompt_token_ids"]),
                       "cot_words": len(cot.split()),
                       "cot_tokens": len(r["output_token_ids"]),
                       "log_prompt_tokens": math.log1p(len(r["prompt_token_ids"])),
                       "log_cot_tokens": math.log1p(len(r["output_token_ids"])),
                       "p1": r["p_black"], "p0": r["p_white"],
                       "p1_entropy": entropy(r["p_black"]),
                       "n_available_admits": len(rows), "chosen_tag": r["tag"],
                       "all_admit_tags": [x["tag"] for x in rows],
                       "all_output_tokens": [len(x["output_token_ids"]) for x in rows]})
    return result


def match_inputs(positives, negatives, caliper, exact_keys=("gender",)):
    if not positives or not negatives:
        return []
    costs = np.full((len(positives), len(negatives)+len(positives)), 1e6)
    costs[:, len(negatives):] = 10.0
    for i, p in enumerate(positives):
        for j, n in enumerate(negatives):
            diff = abs(p["p1"]-n["p1"])
            if diff <= caliper + 1e-12 and all(p[k] == n[k] for k in exact_keys):
                costs[i, j] = diff + 1e-10 * (i*len(negatives)+j)
    ii, jj = linear_sum_assignment(costs)
    return [(positives[i], negatives[j]) for i, j in zip(ii, jj)
            if j < len(negatives) and costs[i, j] < 1.0]


def match_summary(rows, exact_keys=("gender",)):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    out = {"positive": len(pos), "negative": len(neg), "calipers": {}}
    for caliper in (0.0, 1/64, 2/64, 0.05, 0.10, 0.15):
        pairs = match_inputs(pos, neg, caliper, exact_keys)
        out["calipers"][str(caliper)] = {
            "pairs": len(pairs), "matched_inputs": 2*len(pairs),
            "mean_signed_p1_difference": float(np.mean([p["p1"]-n["p1"] for p, n in pairs])) if pairs else None,
            "max_p1_difference": max((abs(p["p1"]-n["p1"]) for p, n in pairs), default=None),
            "mean_p1_difference": float(np.mean([abs(p["p1"]-n["p1"]) for p, n in pairs])) if pairs else None,
            "pairs_input_ids": [[p["input"], n["input"]] for p, n in pairs],
            "positive_p0_le_0.15": sum(p["p0"] <= .15 for p, n in pairs),
            "positive_p0_le_0.25": sum(p["p0"] <= .25 for p, n in pairs),
        }
    out["positive_p0_le"] = {str(c): sum(r["p0"] <= c for r in pos) for c in (.15, .25, .35)}
    out["negative_p0_le"] = {str(c): sum(r["p0"] <= c for r in neg) for c in (.15, .25, .35)}
    out["gender_by_label"] = {str(y): dict(Counter(r.get("gender", "unspecified") for r in rows if r["label"] == y)) for y in (0, 1)}
    return out


def matched_rows(rows, caliper=.10):
    pairs = match_inputs([r for r in rows if r["label"]], [r for r in rows if not r["label"]], caliper)
    out = []
    for i, pair in enumerate(pairs):
        out.extend(dict(r, group=f"pair_{i}") for r in pair)
    return out


def folds(rows, n_splits, seed):
    yy = np.array([r["label"] for r in rows])
    gg = np.array([r["group"] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(rows)), yy, gg))


def fit_predict(train, test, recipe):
    y = np.array([r["label"] for r in train])
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000,
                             random_state=SEED)
    if RECIPES[recipe] is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                                     max_features=5000, sublinear_tf=True)
        xtr = vectorizer.fit_transform([r["prompt"] for r in train])
        xte = vectorizer.transform([r["prompt"] for r in test])
        clf.fit(xtr, y)
        return clf.predict_proba(xte)[:, 1]
    keys = RECIPES[recipe]
    xtr = np.array([[r[k] for k in keys] for r in train])
    xte = np.array([[r[k] for k in keys] for r in test])
    model = make_pipeline(StandardScaler(), clf)
    model.fit(xtr, y)
    return model.predict_proba(xte)[:, 1]


def auc(y, score):
    n1 = y.sum(); n0 = len(y)-n1
    if not n1 or not n0:
        return float("nan")
    rank = rankdata(score)
    return float((rank[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def metrics(y, score, verdict):
    tpr = float(verdict[y == 1].mean()); tnr = float(1-verdict[y == 0].mean())
    return {"auroc": auc(y, score), "tpr": tpr, "tnr": tnr, "fpr": 1-tnr,
            "gmean2_input_classification": tpr*tnr}


def choose_threshold(y, probabilities):
    candidates = np.linspace(.01, .99, 99)
    values = [metrics(y, probabilities, (probabilities >= t).astype(int))["gmean2_input_classification"] for t in candidates]
    # Tie breaking is fixed, preferring proximity to .5 then smaller threshold.
    best = max(range(len(candidates)), key=lambda j: (values[j], -abs(candidates[j]-.5), -candidates[j]))
    return float(candidates[best])


def crossfit(rows, recipe, n_splits, seed):
    pred = np.full(len(rows), np.nan)
    for train, test in folds(rows, n_splits, seed):
        pred[test] = fit_predict([rows[i] for i in train], [rows[i] for i in test], recipe)
    return pred


def bootstrap(rows, score, verdict, n_boot, seed):
    y = np.array([r["label"] for r in rows])
    group_to_indices = defaultdict(list)
    for idx, r in enumerate(rows):
        group_to_indices[r["group"]].append(idx)
    groups = list(group_to_indices)
    rng = np.random.default_rng(seed)
    vals = defaultdict(list)
    for _ in range(n_boot):
        chosen = rng.choice(len(groups), len(groups), replace=True)
        ii = np.array([i for g in chosen for i in group_to_indices[groups[g]]])
        if len(set(y[ii])) < 2:
            continue
        for key, value in metrics(y[ii], score[ii], verdict[ii]).items():
            vals[key].append(value)
    return {key: {"lower_95": float(np.quantile(v, .025)), "upper_95": float(np.quantile(v, .975)),
                  "se": float(np.std(v, ddof=1))} for key, v in vals.items()}


def evaluate(rows, ood, recipe, args):
    y = np.array([r["label"] for r in rows])
    pred = np.full(len(rows), np.nan); verdict = np.zeros(len(rows), dtype=int)
    fold_thresholds = []; fold_ids = np.zeros(len(rows), dtype=int)
    for f, (train_idx, test_idx) in enumerate(folds(rows, args.folds, args.seed)):
        train = [rows[i] for i in train_idx]; test = [rows[i] for i in test_idx]
        inner = crossfit(train, recipe, 4, args.seed+100+f)
        threshold = choose_threshold(np.array([r["label"] for r in train]), inner)
        pred[test_idx] = fit_predict(train, test, recipe)
        verdict[test_idx] = pred[test_idx] >= threshold
        fold_ids[test_idx] = f
        fold_thresholds.append(threshold)
    final_threshold = choose_threshold(y, pred)
    ood_pred = fit_predict(rows, ood, recipe)
    ood_y = np.array([r["label"] for r in ood]); ood_verdict = (ood_pred >= final_threshold).astype(int)
    result = {"recipe": recipe, "features": RECIPES[recipe] or "name-redacted prompt TF-IDF unigrams/bigrams",
              "privileged_rollout_bank": "oracle" in recipe,
              "id": {"n": len(rows), "n_positive": int(y.sum()), "metrics": metrics(y, pred, verdict),
                     "ci": bootstrap(rows, pred, verdict, args.n_boot, args.seed),
                     "fold_thresholds": fold_thresholds},
              "ood": {"n": len(ood), "n_positive": int(ood_y.sum()), "metrics": metrics(ood_y, ood_pred, ood_verdict),
                      "ci": bootstrap(ood, ood_pred, ood_verdict, args.n_boot, args.seed),
                      "threshold_from_id": final_threshold}}
    records = []
    for scope, rr, pp, zz, ff in (("id", rows, pred, verdict, fold_ids),
                                 ("ood", ood, ood_pred, ood_verdict, [-1]*len(ood))):
        for r, p, z, f in zip(rr, pp, zz, ff):
            records.append({"scope": scope, "input": r["input"], "group": r["group"], "label": r["label"],
                            "score": float(p), "verdict": int(z), "outer_fold": int(f)})
    return result, records


def sycophancy_audit(path, target_arm):
    raw = json.loads(path.read_text())
    rows = []
    for r in raw:
        if r["arm"] != target_arm or r["testbed_label"] is None:
            continue
        rows.append(dict(r, input=r["qid"], label=int(r["testbed_label"] == "sycophantic"),
                         gender=r["community_label"]))
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    summary = match_summary(rows, exact_keys=())
    summary["matching_same_community_label"] = match_summary(rows, exact_keys=("community_label",))
    summary["outcome_and_direction"] = target_arm
    summary["low_effect_candidates_with_p1_gt0"] = sum(r["p1"] > 0 for r in neg)
    if pos:
        lo, hi = min(r["p1"] for r in pos), max(r["p1"] for r in pos)
        summary["positive_p1_range"] = [lo, hi]
        summary["low_effect_within_positive_p1_range"] = sum(lo <= r["p1"] <= hi for r in neg)
    summary["positive_ids"] = [r["input"] for r in pos]
    return summary


def plot_results(results, summaries, image_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12})
    image_dir.mkdir(parents=True, exist_ok=True)
    names = {"gender": "Gender", "prompt_length": "Prompt length", "gender_prompt_length": "Gender + prompt length",
             "cot_length": "Trace length", "gender_prompt_cot_lengths": "Gender + both lengths",
             "prompt_tfidf_redacted": "Prompt text (name redacted)", "p1_oracle": "p₁ (rollout-bank diagnostic)",
             "p1_entropy_oracle": "p₁ + entropy (rollout-bank diagnostic)"}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
    for ax, cohort in zip(axes, results):
        entries = results[cohort]
        for j, r in enumerate(entries):
            for delta, scope, color, marker in ((-.12, "id", "#0072B2", "o"), (.12, "ood", "#E69F00", "s")):
                point = r[scope]["metrics"]["auroc"]; ci = r[scope]["ci"]["auroc"]
                ax.errorbar(point, j+delta, xerr=[[max(0, point-ci["lower_95"])], [max(0, ci["upper_95"]-point)]],
                            fmt=marker, color=color, markersize=7, capsize=3,
                            label=("Black-name ID (nested CV)" if scope == "id" else "Native American OOD") if j == 0 else None)
        ax.axvline(.5, color="black", linestyle="--", alpha=.6)
        ax.set(xlim=(.15, .95), xlabel="AUROC for effect-input selection label",
               title=("Original selected inputs" if cohort == "original" else "Matched on gender and p₁ (±0.10)"),
               yticks=range(len(entries)), yticklabels=[names[r["recipe"]] for r in entries])
        ax.grid(axis="x", alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2)
    axes[0].invert_yaxis()
    fig.savefig(image_dir / "nuisance_prediction.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    keys = ["admission", "nativeamerican_disjoint", "sarcasm", "scruples"]
    labels = ["Admission", "Native American\n(disjoint)", "Sarcasm\n(suggest sincere)", "Scruples\n(suggest right)"]
    x = np.arange(len(keys)); width=.2
    for j, caliper in enumerate((.05, .10, .15)):
        counts = [summaries[k]["calipers"][str(caliper)]["pairs"] for k in keys]
        axes[0].bar(x+(j-1)*width, counts, width, label=f"p₁ caliper ±{caliper:.2f}")
        for xx, count in zip(x+(j-1)*width, counts):
            axes[0].text(xx, count+.5, str(count), ha="center", fontsize=14)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Disjoint positive/negative pairs",
                title="Matched-pair availability")
    axes[0].legend()
    for j, cutoff in enumerate((.15, .25, .35)):
        counts = [summaries[k]["positive_p0_le"][str(cutoff)] for k in keys]
        axes[1].bar(x+(j-1)*width, counts, width, label=f"p₀ ≤ {cutoff:.2f}")
        for xx, count in zip(x+(j-1)*width, counts):
            axes[1].text(xx, count+.5, str(count), ha="center", fontsize=14)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Effect-selected inputs", title="Low-baseline positive support")
    axes[1].legend()
    fig.savefig(image_dir / "matching_support.png", dpi=160)
    plt.close(fig)


def descriptive_scores(rows, seed, n_boot):
    y = np.array([r["label"] for r in rows])
    out = {}
    for feature in ("female", "p1"):
        score = np.array([r[feature] for r in rows], dtype=float)
        verdict = (score >= .5).astype(int)
        out[feature] = {"auroc": auc(y, score),
                        "ci": bootstrap(rows, score, verdict, n_boot, seed)["auroc"],
                        "positive_mean": float(score[y == 1].mean()),
                        "negative_mean": float(score[y == 0].mean())}
    return out


def duplicate_audit(black, native):
    normalized = lambda r: re.sub(r"\s+", " ", r["prompt"].lower()).strip()
    bmap = defaultdict(list)
    for r in black:
        bmap[normalized(r)].append(r["input"])
    exact = [[r["input"], bmap[normalized(r)]] for r in native if normalized(r) in bmap]
    # This audit-only vectorizer never supplies features to evaluated models.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=.98)
    xx = vec.fit_transform([r["prompt"] for r in black+native])
    similarity = (xx[len(black):] @ xx[:len(black)].T).toarray()
    top = np.argmax(similarity, axis=1)
    high = [{"nativeamerican": native[j]["input"], "black": black[int(top[j])]["input"],
             "cosine": float(similarity[j, top[j]])}
            for j in range(len(native)) if similarity[j, top[j]] >= .90]
    return {"exact_cross_domain_prompt_duplicates": exact,
            "highest_cross_domain_tfidf_cosine": float(similarity.max()),
            "nativeamerican_nearest_black_cosine_ge_0.90": high,
            "interpretation": "Name-redacted exact text and lexical similarity checks; shared source templates remain, and these checks do not prove semantic independence."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/prompt_bias_v2"))
    ap.add_argument("--out", type=Path, default=Path("results/prompt_bias_v2/rethink_0924"))
    ap.add_argument("--images", type=Path, default=Path("notes/images/monitorability_rethink"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    black = load_admission(args.root / "judge_sets_same_decision/qwen3_8b_admission_same_decision.json", args.seed)
    native_all = load_admission(args.root / "judge_sets_same_decision/qwen3_8b_admission_same_decision_nativeamerican.json", args.seed)
    black_ids = {r["input"] for r in black}
    native = [r for r in native_all if r["input"] not in black_ids]
    assert not black_ids & {r["input"] for r in native}
    summaries = {"admission": match_summary(black), "nativeamerican_disjoint": match_summary(native),
                 "sarcasm": sycophancy_audit(args.root / "sycophancy/sarcasm_qwen3_8b_switch_rates.json", "suggest_sincere"),
                 "scruples": sycophancy_audit(args.root / "sycophancy/scruples_qwen3_8b_switch_rates.json", "suggest_right")}
    summaries["nativeamerican_disjoint"]["excluded_overlap_by_label"] = dict(Counter(str(r["label"]) for r in native_all if r["input"] in black_ids))
    all_results = {}; all_predictions = {}
    cohorts = {"original": (black, native), "matched_gender_p1_0.10": (matched_rows(black), matched_rows(native))}
    args.out.mkdir(parents=True, exist_ok=True)
    for cohort, (rows, ood) in cohorts.items():
        results = []; preds = {}
        for recipe in RECIPES:
            result, records = evaluate(rows, ood, recipe, args)
            results.append(result); preds[recipe] = records
            print(f"{cohort} {recipe}: ID AUC={result['id']['metrics']['auroc']:.3f}; OOD AUC={result['ood']['metrics']['auroc']:.3f}", flush=True)
        all_results[cohort] = results; all_predictions[cohort] = preds
    out = {"seed": args.seed, "unit": "One underlying application; labels are input-level selection proxies, not identified individual causation.",
           "length_feature": "One admit selected by SHA256(seed:base_id); output token count includes the small final-answer suffix.",
           "validation": "Fixed C=1 recipes; five outer groups; four inner groups tune threshold only. Paired matched inputs stay in the same fold. No model/hyperparameter selected from outer results.",
           "intervals": "95% percentile bootstrap of held-out input predictions (matched-pair clusters in matched cohorts), conditional on fitted training models; does not include training-model uncertainty.",
           "ood": "Native American base applications absent from every Black179 input; threshold comes from Black ID out-of-fold predictions.",
           "caveats": ["Retrospective selected collections, not an untouched confirmatory test.",
                       "p1/entropy diagnostics require an intervention rollout bank and share outcome noise with selection.",
                       "Ordinary TPR*TNR scores input-label classification here, not the paper's two-arm causal monitorability metric.",
                       "CoT length feature uses total output token length; final-answer suffix size remains a possible component."],
           "support": summaries, "results": all_results,
           "duplicate_audit": duplicate_audit(black, native),
           "fixed_feature_associations": {cohort: {scope: descriptive_scores(rr, args.seed, args.n_boot) for scope, rr in (("id", rows), ("ood", ood))} for cohort, (rows, ood) in cohorts.items()}}
    (args.out / "shortcut_audit.json").write_text(json.dumps(out, indent=2)+"\n")
    (args.out / "shortcut_predictions.json").write_text(json.dumps(all_predictions, indent=2)+"\n")
    (args.out / "shortcut_split_manifest.json").write_text(json.dumps({cohort: {"id": rows, "ood": ood} for cohort, (rows, ood) in cohorts.items()}, indent=2)+"\n")
    plot_results(all_results, summaries, args.images)
    print(json.dumps({key: {"positive": value["positive"], "negative": value["negative"],
                           "matched_pairs": {c: x["pairs"] for c, x in value["calipers"].items()}} for key, value in summaries.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
