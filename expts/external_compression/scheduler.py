"""File-queue GPU scheduler for the external-compression runs.

Queue layout (under results/external_compression/queue/):
    pending/   one JSON per job: {"job_id", "cmd": [...], "log": path}
    running/   claimed jobs (atomic rename from pending/)
    done/      completed jobs (exit 0), annotated with timing
    failed/    non-zero exit, annotated with exit code

Job files sort lexicographically; workers always claim the first pending
job, so name jobs with a priority prefix (e.g. ``b0_a_...`` < ``b0_b_...``
< ``b1_a_...`` — buckets in order, cheap stage before SNP within a bucket).

Worker (one per GPU, run inside tmux):
    uv run python -m expts.external_compression.scheduler --worker --gpu 3

Enqueue jobs for buckets:
    uv run python -m expts.external_compression.scheduler --enqueue \
        --buckets lt50,50-100 --stages cheap,snp

Status:
    uv run python -m expts.external_compression.scheduler --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from expts.external_compression.common import (
    BUCKET_NAMES,
    DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
)

QUEUE_DIR = os.path.join(RESULTS_DIR, "queue")
STATES = ["pending", "running", "done", "failed"]
LOG_DIR = os.path.join(REPO_ROOT, "logs", "external_compression", "jobs")


def _qdir(state: str) -> str:
    d = os.path.join(QUEUE_DIR, state)
    os.makedirs(d, exist_ok=True)
    return d


def enqueue(buckets, stages) -> None:
    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = json.load(f)
    os.makedirs(LOG_DIR, exist_ok=True)
    n = 0
    for inst in instances:
        if inst["bucket"] not in buckets:
            continue
        b_idx = BUCKET_NAMES.index(inst["bucket"])
        for stage in stages:
            s_idx = {"cheap": "a", "snp": "b", "snp_uniform_l0": "c"}[stage]
            job_id = f"b{b_idx}_{s_idx}_{stage}_{inst['instance_id']}"
            path = os.path.join(_qdir("pending"), f"{job_id}.json")
            if any(
                os.path.exists(os.path.join(_qdir(s), f"{job_id}.json"))
                for s in STATES
            ):
                continue
            job = {
                "job_id": job_id,
                "cmd": [
                    "uv", "run", "python", "-m",
                    "expts.external_compression.run_instance",
                    "--instance", inst["instance_id"], "--stage", stage,
                ],
                "log": os.path.join(LOG_DIR, f"{job_id}.log"),
            }
            with open(path, "w") as f:
                json.dump(job, f, indent=2)
            n += 1
    print(f"enqueued {n} jobs")


def claim_next() -> dict | None:
    pending = _qdir("pending")
    for name in sorted(os.listdir(pending)):
        src = os.path.join(pending, name)
        dst = os.path.join(_qdir("running"), name)
        try:
            os.rename(src, dst)  # atomic claim
        except OSError:
            continue
        with open(dst) as f:
            job = json.load(f)
        job["_file"] = name
        return job
    return None


def worker(gpu: int, drain: bool) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"worker gpu={gpu} started (drain={drain})")
    while True:
        job = claim_next()
        if job is None:
            if drain:
                print("queue empty — draining, worker exits")
                return
            time.sleep(20)
            continue
        name = job["_file"]
        print(f"[gpu{gpu}] running {job['job_id']}")
        t0 = time.time()
        with open(job["log"], "a") as logf:
            logf.write(f"\n=== start {job['job_id']} gpu={gpu} {time.ctime()} ===\n")
            logf.flush()
            proc = subprocess.run(
                job["cmd"], cwd=REPO_ROOT, env=env,
                stdout=logf, stderr=subprocess.STDOUT,
            )
        job["elapsed_s"] = time.time() - t0
        job["exit_code"] = proc.returncode
        job["gpu"] = gpu
        state = "done" if proc.returncode == 0 else "failed"
        dst = os.path.join(_qdir(state), name)
        with open(dst, "w") as f:
            json.dump({k: v for k, v in job.items() if k != "_file"}, f, indent=2)
        os.remove(os.path.join(_qdir("running"), name))
        print(f"[gpu{gpu}] {state}: {job['job_id']} "
              f"({job['elapsed_s']:.0f}s, exit {proc.returncode})")


def status() -> None:
    for s in STATES:
        d = _qdir(s)
        names = sorted(os.listdir(d))
        print(f"{s:8s} {len(names):3d}" + (": " + ", ".join(names[:6]) +
              (" ..." if len(names) > 6 else "") if names else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--drain", action="store_true",
                        help="worker exits when the queue is empty")
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--buckets", default=",".join(BUCKET_NAMES))
    parser.add_argument("--stages", default="cheap,snp")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.enqueue:
        enqueue(args.buckets.split(","), args.stages.split(","))
    elif args.worker:
        worker(args.gpu, args.drain)
    elif args.status:
        status()
    else:
        parser.error("pick one of --worker / --enqueue / --status")


if __name__ == "__main__":
    main()
