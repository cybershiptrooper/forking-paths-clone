"""Linear probes on cached activations for the *same-decision* contrast:
admit traces under the Black name from inputs the name moved (positives,
from the confirmed intervention set) against admit traces under the Black
name from inputs the name did not move (negatives, the same-decision judge
set). Both classes have the same prompt template, the same name group and
the same answer, so the probe cannot read the arm or the decision; it has
to find whatever distinguishes a caused admit from an uncaused one.

Grouped k-fold over inputs; per (position, layer, C) the out-of-fold
probabilities give AUROC and, with a threshold tuned in-fold, g-mean2 with
the minimal-criterion TPR (positives' Reffect from the instance table;
negatives are null inputs). ``--train_sizes`` limits the positives per
input (nested subsets); every negative trace is used.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.probe_same_decision --out results/prompt_bias_v2/probes/same_decision
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from expts.prompt_bias_circuit_discovery.cache_activations import cache_key, load_cached, POSITIONS
from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc


def gmean2(samples, probs, thr, reffect):
    """TPRmc x TNR over instances: positives grouped by input (TPR_i / Reffect_i, capped), negatives grouped by input."""
    by = collections.defaultdict(list)
    for s, p in zip(samples, probs):
        by[(s["_inst"], s["_y"])].append(int(p >= thr))
    tprmc = [min(1.0, np.mean(v) / reffect[i]) for (i, y), v in by.items() if y == 1 and reffect.get(i)]
    tnr = [1 - np.mean(v) for (i, y), v in by.items() if y == 0]
    return float(np.mean(tprmc) * np.mean(tnr)) if tprmc and tnr else 0.0, float(np.mean(tprmc) if tprmc else 0), float(np.mean(tnr) if tnr else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="results/prompt_bias_v2/intervention_eval/qwen3_8b_admission_black_vs_white_confirmed_k32")
    ap.add_argument("--negatives", default="results/prompt_bias_v2/judge_sets_same_decision/qwen3_8b_admission_same_decision.json")
    ap.add_argument("--act_dir", default="results/activations/qwen3_8b")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--train_sizes", nargs="+", type=int, default=[4, 16, 32])
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28, 32, 36])
    ap.add_argument("--positions", nargs="+", default=POSITIONS)
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    S = json.load(open(args.set + ".json")); I = json.load(open(args.set + "_instances.json"))["instances"]
    reffect = {i["instance"]: i["reffect"] for i in I}
    pos = [s for s in S if s["arm"] == 1 and s["y"] == 1]
    order = collections.defaultdict(int)
    for s in pos:
        s["_inst"] = s["instance"]; s["_y"] = 1; s["_pos_in"] = order[s["instance"]]; order[s["instance"]] += 1
        s["_key"] = cache_key(args.model_name, s["prompt_token_ids"], s["output_token_ids"], POSITIONS)
    neg = [r for r in json.load(open(args.negatives)) if not r["is_positive"]]
    for r in neg:
        r["_inst"] = "neg_" + r["uid"]; r["_y"] = 0; r["_pos_in"] = r["rollout_index"]
        r["_key"] = cache_key(args.model_name, r["prompt_token_ids"], r["output_token_ids"], POSITIONS)
    insts = sorted({s["_inst"] for s in pos} | {r["_inst"] for r in neg}); rng.shuffle(insts)
    fold_of = {i: k % args.folds for k, i in enumerate(insts)}
    feats = {}

    def X_of(samples, p, l):
        if (p, l) not in feats:
            allk = [s["_key"] for s in pos] + [r["_key"] for r in neg]
            arr = load_cached(args.act_dir, allk, positions=[p], layers=[l])[:, 0, 0, :].astype(np.float32)
            feats[(p, l)] = dict(zip(allk, arr))
        return np.stack([feats[(p, l)][s["_key"]] for s in samples])

    out = {"args": vars(args), "n_pos_inputs": len({s["_inst"] for s in pos}), "n_neg_inputs": len({r["_inst"] for r in neg}), "runs": []}
    t0 = time.time()
    for size in args.train_sizes:
        samples = [s for s in pos if s["_pos_in"] < size] + neg
        y = np.array([s["_y"] for s in samples])
        grid = []
        for p in args.positions:
            for l in args.layers:
                X = X_of(samples, p, l)
                for C in args.Cs:
                    oof = np.zeros(len(samples)); thr_folds = []
                    for f in range(args.folds):
                        tr = np.array([fold_of[s["_inst"]] != f for s in samples]); va = ~tr
                        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                        p_tr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
                        # threshold tuned on the training fold, applied to the validation fold
                        best = max(((gmean2([s for s, m in zip(samples, tr) if m], p_tr, t, reffect)[0], t) for t in np.linspace(0.01, 0.99, 99)))
                        thr_folds.append(best[1]); oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
                    # score out-of-fold predictions with each fold's own threshold
                    Z = np.zeros(len(samples), int)
                    for f in range(args.folds):
                        m = np.array([fold_of[s["_inst"]] == f for s in samples]); Z[m] = (oof[m] >= thr_folds[f]).astype(int)
                    g, tprmc, tnr = gmean2(samples, Z.astype(float), 0.5, reffect)
                    a = auroc(list(oof[y == 1]), list(oof[y == 0]))
                    # input-level AUROC: mean probability per input
                    mp = collections.defaultdict(list)
                    for s, pr in zip(samples, oof):
                        mp[(s["_inst"], s["_y"])].append(pr)
                    a_in = auroc([np.mean(v) for (i, yy), v in mp.items() if yy == 1], [np.mean(v) for (i, yy), v in mp.items() if yy == 0])
                    grid.append(dict(pos=p, layer=l, C=C, gmean2=g, tprmc=tprmc, tnr=tnr, auroc=a, auroc_inputs=a_in))
                print(f"size {size} {p} L{l}: best AUROC {max(r['auroc'] for r in grid if r['pos'] == p and r['layer'] == l):.3f} g-mean2 {max(r['gmean2'] for r in grid if r['pos'] == p and r['layer'] == l):.3f} ({time.time() - t0:.0f}s)", flush=True)
        best = max(grid, key=lambda r: r["auroc"])
        out["runs"].append(dict(train_size_per_input=size, n_pos=int(y.sum()), n_neg=int((y == 0).sum()), best=best, grid=grid))
        print(f"== size {size}: n_pos {int(y.sum())} n_neg {int((y == 0).sum())} | best {best['pos']} L{best['layer']} C {best['C']}: AUROC {best['auroc']:.3f} (inputs {best['auroc_inputs']:.3f}) g-mean2 {best['gmean2']:.3f} TPRmc {best['tprmc']:.2f} TNR {best['tnr']:.2f}", flush=True)
        json.dump(out, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
