"""Linear probes on cached hidden states for the intervention evaluation
(the LessWrong CoT-interp testbed protocol): logistic regression on one
layer's activation at one position, hyperparameters (layer, position, L2
strength) and the decision threshold chosen by grouped cross-validation
over training instances on g-mean2_mc, then frozen and scored on held-out
instances.

Samples: the rollouts of ``--set`` (``build_intervention_eval_set.py``);
label 1 = an intervention-arm rollout with the behaviour (a Black-name admit),
0 = every control-arm rollout and every intervention-arm reject. Training
uses ``--n_train_instances`` instances and, per arm, the first
``--train_sizes`` rollouts in file order (nested subsets); the test set is
every rollout of the remaining instances. ``--degeneracy_set`` (admit traces
of inputs where the name did nothing) is scored as extra negatives.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.train_activation_probes \
    --set results/prompt_bias_v2/intervention_eval/qwen3_8b_admission_black_vs_white_confirmed_k32 --out results/prompt_bias_v2/probes/admission_confirmed
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from expts.prompt_bias_circuit_discovery.cache_activations import cache_key, load_cached, POSITIONS
from expts.prompt_bias_circuit_discovery.intervention_metrics import aggregate, group_samples
from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc


def gmean_at(samples, instances, scores, thr):
    Z = {s["tag"]: int(sc >= thr) for s, sc in zip(samples, scores)}
    return aggregate(instances, group_samples(samples, Z))


def best_threshold(samples, instances, scores):
    best = (-1, 0.5, None)
    for thr in np.linspace(0.01, 0.99, 99):
        a = gmean_at(samples, instances, scores, thr)
        if a["gmean2"] > best[0]:
            best = (a["gmean2"], float(thr), a)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--degeneracy_set", default="results/prompt_bias_v2/judge_sets_same_decision/qwen3_8b_admission_same_decision.json")
    ap.add_argument("--act_dir", default="results/activations/qwen3_8b")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_train_instances", type=int, default=50)
    ap.add_argument("--train_sizes", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28, 32, 36])
    ap.add_argument("--positions", nargs="+", default=POSITIONS)
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    S = json.load(open(args.set + ".json")); I = json.load(open(args.set + "_instances.json"))["instances"]
    inst_by = {i["instance"]: i for i in I}
    ids = sorted(inst_by); rng.shuffle(ids)
    train_ids, test_ids = set(ids[:args.n_train_instances]), set(ids[args.n_train_instances:])
    order = collections.defaultdict(int)
    for s in S:
        s["_pos_in_arm"] = order[(s["instance"], s["arm"])]; order[(s["instance"], s["arm"])] += 1
        s["_y"] = int(s["arm"] == 1 and s["y"] == 1)
        s["_key"] = cache_key(args.model_name, s["prompt_token_ids"], s["output_token_ids"], POSITIONS)
    D = [r for r in json.load(open(args.degeneracy_set)) if not r["is_positive"]] if args.degeneracy_set else []
    for r in D:
        r["_key"] = cache_key(args.model_name, r["prompt_token_ids"], r["output_token_ids"], POSITIONS)
    test = [s for s in S if s["instance"] in test_ids]
    fold_of = {iid: k % args.folds for k, iid in enumerate(sorted(train_ids))}
    results = {"args": vars(args), "train_instances": sorted(train_ids), "test_instances": sorted(test_ids), "runs": []}
    t0 = time.time()
    feats = {}

    def X_of(samples, pos, layer):
        key = (pos, layer)
        if key not in feats:
            allk = [s["_key"] for s in S] + [r["_key"] for r in D]
            arr = load_cached(args.act_dir, allk, positions=[pos], layers=[layer])[:, 0, 0, :].astype(np.float32)
            feats[key] = dict(zip(allk, arr))
        return np.stack([feats[key][s["_key"]] for s in samples])

    for size in args.train_sizes:
        train = [s for s in S if s["instance"] in train_ids and s["_pos_in_arm"] < size]
        y_tr = np.array([s["_y"] for s in train])
        grid = []
        for pos in args.positions:
            for layer in args.layers:
                X = X_of(train, pos, layer)
                for C in args.Cs:
                    oof = np.zeros(len(train))
                    for f in range(args.folds):
                        tr = np.array([fold_of[s["instance"]] != f for s in train]); va = ~tr
                        sc = StandardScaler().fit(X[tr])
                        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y_tr[tr])
                        oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
                    g, thr, a = best_threshold(train, [inst_by[i] for i in train_ids], oof)
                    grid.append(dict(pos=pos, layer=layer, C=C, cv_gmean2=g, thr=thr, cv_auroc=auroc(list(oof[y_tr == 1]), list(oof[y_tr == 0])),
                                     cv_tprmc=a["tprmc"], cv_tnr1=a["tnr1"], cv_tnr0=a["tnr0"]))
                print(f"size {size} pos {pos} layer {layer}: best cv g-mean2 {max(r['cv_gmean2'] for r in grid if r['pos'] == pos and r['layer'] == layer):.3f} ({time.time() - t0:.0f}s)", flush=True)
        best = max(grid, key=lambda r: r["cv_gmean2"])
        X = X_of(train, best["pos"], best["layer"]); sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=best["C"], max_iter=2000).fit(sc.transform(X), y_tr)
        Xt = X_of(test, best["pos"], best["layer"]); p_te = clf.predict_proba(sc.transform(Xt))[:, 1]
        y_te = np.array([s["_y"] for s in test])
        a_te = gmean_at(test, [inst_by[i] for i in test_ids], p_te, best["thr"])
        # oracle threshold on the test set (upper bound, not the protocol number) and the degeneracy negatives
        g_or, thr_or, _ = best_threshold(test, [inst_by[i] for i in test_ids], p_te)
        deg = {}
        if D:
            Xd = X_of(D, best["pos"], best["layer"]); p_d = clf.predict_proba(sc.transform(Xd))[:, 1]
            deg = dict(n=len(D), flagged_fraction=float((p_d >= best["thr"]).mean()), mean_prob=float(p_d.mean()),
                       auroc_vs_test_positives=auroc(list(p_te[y_te == 1]), list(p_d)))
        run = dict(train_size_per_arm=size, n_train_samples=len(train), n_train_pos=int(y_tr.sum()), best=best,
                   test=dict(n=len(test), n_pos=int(y_te.sum()), gmean2=a_te["gmean2"], tpr=a_te["tpr"], tprmc=a_te["tprmc"], tnr1=a_te["tnr1"], tnr0=a_te["tnr0"],
                             auroc=auroc(list(p_te[y_te == 1]), list(p_te[y_te == 0])), oracle_threshold_gmean2=g_or),
                   degeneracy=deg, grid=grid)
        results["runs"].append(run)
        print(f"== size {size}: best {best['pos']} L{best['layer']} C={best['C']} thr={best['thr']:.2f} cv g-mean2 {best['cv_gmean2']:.3f} | test g-mean2 {a_te['gmean2']:.3f} "
              f"(TPR {a_te['tpr']:.2f} TPRmc {a_te['tprmc']:.2f} TNR1 {a_te['tnr1']:.2f} TNR0 {a_te['tnr0']:.2f}) AUROC {run['test']['auroc']:.3f} | degeneracy flagged {deg.get('flagged_fraction', float('nan')):.2f}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out + ".json", "w"), indent=1)
    print("wrote", args.out + ".json")


if __name__ == "__main__":
    main()
