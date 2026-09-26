"""Train a termination mask while resampling the candidate bank every N steps.

The candidate bank is normally sampled once from the clean model and fixed
for the whole run (off-policy training).  This driver interleaves training
with bank refreshes WITHOUT modifying the trainer classes, using their
existing checkpoint-resume support:

    segment 0: train steps [0, N)          on the original clean-model bank
    resample:  generate a fresh bank under the segment-0 mask (top-k at the
               target sparsity), label under the clean model
    segment 1: resume from the checkpoint, train steps [N, 2N) on the new
               bank
    ... and so on until --total_steps.

Each segment is a separate ``run.py`` subprocess (the model is reloaded per
phase; ~1.5 min overhead per segment).  The sparsity-penalty schedule of
the single-run baseline (weight 0 for the first 25% of steps, linear ramp
to l0_lambda over the next 50%, then held) is reproduced exactly at every
absolute step by passing per-segment values of l0_lambda / l0_warmup_frac /
l0_ramp_frac: for a segment ending at step S of T total,

    warmup_frac = min(1, 0.25 T / S)
    ramp_frac   = max(0, (min(S, 0.75 T) - 0.25 T) / S)
    l0_lambda   = lambda_max * clip((S - 0.25 T) / (0.5 T), 0, 1),

which makes the within-segment linear ramp interpolate the baseline's
lambda(step) at every step <= S.

Usage (one GPU):
    uv run python -m expts.cot_termination_circuit_discovery.train_with_periodic_bank_resampling \
        --config expts/cot_termination_circuit_discovery/configs/sweeps/loss_comparison/lc_aqua_p080_s57_stop_prob_tsp10.yaml \
        --resample_every 50 \
        --output_dir results/snp_sweep/bank_resampling/masks \
        --work_dir results/snp_sweep/bank_resampling/segments
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from utils.expt_config import load_config


def seg_schedule(seg_end, total, lambda_max, warmup_frac=0.25, ramp_frac=0.50):
    warmup_end = warmup_frac * total
    ramp_end = (warmup_frac + ramp_frac) * total
    ramp_len = ramp_end - warmup_end
    w = min(1.0, warmup_end / seg_end)
    r = max(0.0, (min(seg_end, ramp_end) - warmup_end) / seg_end)
    lam = lambda_max * min(1.0, max(0.0, (seg_end - warmup_end) / ramp_len))
    return lam, w, r


def run_cmd(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="A standard training YAML (e.g. a loss_comparison "
                    "config); the bank/boundary/output paths in it are "
                    "overridden per segment.")
    ap.add_argument("--resample_every", type=int, required=True)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Defaults to the config's num_training_steps.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--work_dir", required=True,
                    help="Where segment banks/boundary data/checkpoints go.")
    ap.add_argument("--file_name", default=None,
                    help="Defaults to <config stem>_resample<N>.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = int(args.total_steps or cfg.get("num_training_steps", 1000))
    interval = args.resample_every
    if total % interval != 0:
        raise ValueError(f"total_steps {total} not divisible by "
                         f"resample_every {interval}")
    n_segments = total // interval
    lambda_max = float(cfg.get("l0_lambda", 100.0))
    target_sparsity = float(cfg["target_sparsity"])
    original_bank = cfg["answer_bank_path"]
    original_bd = cfg.get("boundary_data_path")

    stem = os.path.splitext(os.path.basename(args.config))[0]
    run_name = args.file_name or f"{stem}_resample{interval:03d}.json"
    run_stem = run_name.removesuffix(".json")
    seg_dir = os.path.join(args.work_dir, run_stem)
    os.makedirs(seg_dir, exist_ok=True)
    ckpt = os.path.join(seg_dir, "checkpoint.pt")
    state_path = os.path.join(seg_dir, "driver_state.json")
    done_seg = -1
    if os.path.exists(state_path):
        with open(state_path) as f:
            done_seg = json.load(f).get("last_completed_segment", -1)
        print(f"resuming driver: last completed segment {done_seg}")

    bank_k, bd_k = original_bank, original_bd
    for k in range(n_segments):
        seg_end = (k + 1) * interval
        prev_bank, prev_bd = bank_k, bd_k
        if k > 0:
            bank_k = os.path.join(seg_dir, f"bank_seg{k}.json")
            bd_k = (os.path.join(seg_dir, f"boundary_seg{k}.json")
                    if original_bd else None)
        if k <= done_seg:
            continue
        if k > 0 and not os.path.exists(bank_k):
            cmd = [
                "uv", "run", "python", "-m",
                "expts.cot_termination_circuit_discovery.resample_bank",
                "--original_bank", original_bank,
                "--checkpoint", ckpt,
                "--target_sparsity", str(target_sparsity),
                "--seed", str(args.seed),
                "--seed_offset", str(k),
                "--output_bank", bank_k,
                "--fallback_bank", prev_bank,
            ]
            if original_bd:
                cmd += ["--original_boundary_data", original_bd,
                        "--output_boundary_data", bd_k,
                        "--fallback_boundary_data", prev_bd]
            run_cmd(cmd)
        lam, w, r = seg_schedule(seg_end, total, lambda_max)
        cmd = [
            "uv", "run", "python", "-m",
            "expts.cot_termination_circuit_discovery.run",
            "--config", args.config,
            "--answer_bank_path", bank_k,
            "--num_training_steps", str(seg_end),
            "--l0_lambda", str(lam),
            "--l0_warmup_frac", str(w),
            "--l0_ramp_frac", str(r),
            "--checkpoint_path", ckpt,
            "--checkpoint_every", str(interval),
            "--output_dir", args.output_dir,
            "--file_name", run_name,
        ]
        if bd_k:
            cmd += ["--boundary_data_path", bd_k]
        if os.path.exists(ckpt):
            cmd += ["--resume_from_checkpoint"]
        run_cmd(cmd)
        with open(state_path, "w") as f:
            json.dump({"last_completed_segment": k,
                       "seg_end": seg_end,
                       "bank": bank_k}, f)
        print(f"segment {k + 1}/{n_segments} done (step {seg_end}/{total})",
              flush=True)
    print(f"all segments done; final mask: "
          f"{os.path.join(args.output_dir, run_name)}")


if __name__ == "__main__":
    main()
