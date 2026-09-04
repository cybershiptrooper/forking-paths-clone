"""Column SNP hyperparameter search (per review, before any final sweep).

Grid axes: learning rate, l0_lambda, log_alpha init (start closed / half /
open / random), num_hc_samples_per_step.  Fixed: uniform per-sentence L0
budget, slow lambda ramp (warmup 5%, ramp 95%), 1000 steps, target keep 15.

Per run, the summary JSON records the three selection criteria:
1. target sparsity reached: achieved column sparsity vs target;
2. no loss-spike learning: sparsity-trajectory gradualness — the largest
   sparsity change in any 20-step logging window as a fraction of the
   total change (small = gradual, structured pruning);
3. performance at the achieved sparsity: prefix-compression answer KL of
   the learned ranking at M in {5, 15} (and at M = number of open gates).

Usage (one GPU, one config):
    uv run python -m expts.external_compression.hparam_search \
        --instance gpqa_gpqa_diamond_0002_pl50 \
        --lr 0.1 --l0_lambda 10 --init 0 --hc 1
--init is the initial gate value: "closed" (log_alpha -3), "half" (0),
"open" (+2), or "random" (Uniform(-2,2) per gate).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from expts.external_compression import evaluate as evaluate_mod
from expts.external_compression import score_methods as sm
from expts.external_compression.common import DATA_DIR, MODEL_NAME, RESULTS_DIR

SEARCH_DIR = os.path.join(RESULTS_DIR, "hparam_search")
INIT_MAP = {"closed": -3.0, "half": 0.0, "open": 2.0, "random": "random"}


def trajectory_stats(metrics_path: str) -> dict:
    steps = [json.loads(l) for l in open(metrics_path)]
    sp = [s["sparsity"] for s in steps]
    tl = [s["task_loss"] for s in steps]
    total = sp[-1] - sp[0]
    jumps = [abs(b - a) for a, b in zip(sp, sp[1:])]
    largest = max(jumps) if jumps else 0.0
    tail = tl[-10:]
    return {
        "sparsity_start": sp[0],
        "sparsity_final": sp[-1],
        "largest_window_jump": largest,
        "largest_jump_fraction": (largest / abs(total)) if total else None,
        "task_loss_final_mean": sum(tail) / len(tail),
        "task_loss_final_max": max(tail),
        "n_logged": len(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--l0_lambda", type=float, required=True)
    parser.add_argument("--init", choices=list(INIT_MAP), required=True)
    parser.add_argument("--hc", type=int, required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()

    tag = f"lr{args.lr:g}_l{args.l0_lambda:g}_init-{args.init}_hc{args.hc}"
    out_root = os.path.join(SEARCH_DIR, args.instance, tag)
    summary_path = os.path.join(out_root, "summary.json")
    if os.path.exists(summary_path):
        print(f"skip: {summary_path} exists")
        return
    os.makedirs(out_root, exist_ok=True)

    from utils.utils import set_seed
    set_seed(42)

    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = {r["instance_id"]: r for r in json.load(f)}
    inst = instances[args.instance]

    from transformers import AutoTokenizer
    from expts.direct_answer_circuit_discovery.learn import load_model_eager
    from utils.circuit_eval import install_clean_sdpa_forward, remove_handles

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(
        args.model_name, device="cuda", gradient_checkpointing=True,
    )
    inputs = sm.build_inputs(inst, tokenizer)

    hparams = {
        "learning_rate": args.lr,
        "l0_lambda": args.l0_lambda,
        "log_alpha_init": INIT_MAP[args.init],
        "num_hc_samples_per_step": args.hc,
        "l0_warmup_frac": 0.05,
        "l0_ramp_frac": 0.95,
    }
    t0 = time.time()
    scores = sm.colsnp_scores(
        model, tokenizer, inputs, args.instance,
        uniform_l0=True, hparams=hparams, out_root=out_root,
        method_name="colsnp_search",
    )
    train_seconds = time.time() - t0

    # --- criteria ---------------------------------------------------------
    traj = trajectory_stats(
        os.path.join(out_root, "colsnp_search_logs", "training_metrics.jsonl")
    )
    rankable = scores[1:]
    n_open = sum(1 for s in rankable if s > 0)
    num_rankable = len(rankable)
    target_sparsity = max(0.0, 1.0 - sm.SNP_TARGET_KEEP / num_rankable)

    model.eval()
    handles = install_clean_sdpa_forward(model)
    try:
        ev = evaluate_mod.InstanceEvaluator(inst, model, tokenizer)
        order = evaluate_mod.ranking_from_scores(scores, ev.num_compress)
        evals = {}
        for M in sorted({5, 15, max(1, min(n_open, ev.num_compress))}):
            if M > ev.num_compress:
                continue
            sel = sorted(order[:M])
            evals[f"answer_kl_M{M}"] = ev.answer_kl(sel)
            tk = ev.token_kl(sel)
            evals[f"token_kl_M{M}"] = tk
    finally:
        remove_handles(handles)

    summary = {
        "instance_id": args.instance,
        "tag": tag,
        "hparams": {k: str(v) for k, v in hparams.items()},
        "num_rankable": num_rankable,
        "target_sparsity": target_sparsity,
        "achieved_sparsity_final": traj["sparsity_final"],
        "reached_target": traj["sparsity_final"] >= target_sparsity - 0.02,
        "n_open_gates": n_open,
        "largest_jump_fraction": traj["largest_jump_fraction"],
        "trajectory": traj,
        "evals": evals,
        "train_seconds": train_seconds,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
