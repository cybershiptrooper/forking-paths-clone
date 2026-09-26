"""Evaluate one mask file of this experiment with eval_log_alpha.

- learned SNP masks: top-k at the mask's own training target sparsity;
- Thought Anchors masks: top-k at every sparsity in --all_sparsities,
  with ``--learnable_region prompt_to_trace`` and the record's letters.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.run_eval \
        --mask_path <mask.json> --eval_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--all_sparsities", default="0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.95")
    ap.add_argument("--skip_existing", type=int, default=1)
    args = ap.parse_args()

    with open(args.mask_path) as f:
        mask = json.load(f)
    meta = mask["metadata"]
    stem = os.path.splitext(os.path.basename(args.mask_path))[0]
    out = os.path.join(args.eval_dir, f"{stem}.eval.json")
    if args.skip_existing and os.path.exists(out):
        print(f"exists: {out}")
        return
    is_snp = str(meta.get("objective", "")).startswith("answer_probe")
    cmd = [
        sys.executable, "-m", "expts.direct_answer_circuit_discovery.eval_log_alpha",
        "--mask_path", args.mask_path,
        "--model_name", mask.get("model_name") or "Qwen/Qwen3-8B",
        "--data_path", meta["data_path"],
        "--prompt_index", str(meta["prompt_index"]),
        "--analysis_sentence_step", str(meta["analysis_sentence_step"]),
        "--sentences_after_prefix", str(meta.get("sentences_after_prefix", 5) or 0),
        "--sentence_gap", str(meta.get("sentence_gap", 1)),
        "--learnable_region", "prompt_to_trace",
        "--output", out,
    ]
    if is_snp:
        cmd += ["--top_k_sparsities", str(meta["target_sparsity"])]
    else:
        cmd += ["--top_k_sparsities", args.all_sparsities]
        if not meta.get("answer_letters"):
            with open(meta["data_path"]) as f:
                rec = json.load(f)[meta["prompt_index"]]
            cmd += ["--answer_letters", ",".join(" " + l for l in rec["all_letters"])]
    os.makedirs(args.eval_dir, exist_ok=True)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
