"""Enqueue the column SNP hyperparameter grid as scheduler jobs.

Grid (per review): lr x l0_lambda x init x hc, on two filter-passing
instances.  Stage 1 runs the full factorial at hc=1 (cheap); the top
configs are validated at hc=8 afterwards (--hc8 --configs tag1,tag2,...).

Usage:
    uv run python -m expts.external_compression.enqueue_hparam            # stage 1
    uv run python -m expts.external_compression.enqueue_hparam \
        --hc8 --configs lr0.1_l10_init-half,lr0.3_l3_init-random          # stage 2
"""

from __future__ import annotations

import argparse
import json
import os

from expts.external_compression.common import REPO_ROOT
from expts.external_compression.scheduler import LOG_DIR, _qdir, STATES

INSTANCES = [
    "gpqa_gpqa_diamond_0002_pl50",          # lt50 bucket, 45 rankable
    "bigbench_logical_deduction_0000_pl85", # 50-100 bucket, 80 rankable
]
LRS = [0.03, 0.1, 0.3]
LAMBDAS = [3, 10, 30]
INITS = ["closed", "half", "open", "random"]


def enqueue_one(inst: str, lr: float, lam: float, init: str, hc: int) -> bool:
    short = inst.replace("gpqa_gpqa_diamond_", "g").replace(
        "bigbench_logical_deduction_", "bb")
    tag = f"lr{lr:g}_l{lam:g}_init-{init}_hc{hc}"
    job_id = f"a_hp_{short}_{tag}"
    if any(os.path.exists(os.path.join(_qdir(s), f"{job_id}.json"))
           for s in STATES + ["held"]):
        return False
    os.makedirs(LOG_DIR, exist_ok=True)
    job = {
        "job_id": job_id,
        "cmd": [
            "uv", "run", "python", "-m",
            "expts.external_compression.hparam_search",
            "--instance", inst, "--lr", str(lr),
            "--l0_lambda", str(lam), "--init", init, "--hc", str(hc),
        ],
        "log": os.path.join(LOG_DIR, f"{job_id}.log"),
    }
    with open(os.path.join(_qdir("pending"), f"{job_id}.json"), "w") as f:
        json.dump(job, f, indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="",
                        help="comma-separated lrX_lY_init-Z tags (validation mode)")
    parser.add_argument("--instances", default=",".join(INSTANCES))
    parser.add_argument("--hc", type=int, default=1)
    args = parser.parse_args()
    instances = args.instances.split(",")

    n = 0
    if args.configs:
        for cfg in args.configs.split(","):
            lr = float(cfg.split("_")[0][2:])
            lam = float(cfg.split("_")[1][1:])
            init = cfg.split("init-")[1].split("_")[0]
            for inst in instances:
                n += enqueue_one(inst, lr, lam, init, hc=args.hc)
    else:
        for inst in instances:
            for lr in LRS:
                for lam in LAMBDAS:
                    for init in INITS:
                        n += enqueue_one(inst, lr, lam, init, hc=args.hc)
    print(f"enqueued {n} hparam jobs")


if __name__ == "__main__":
    main()
