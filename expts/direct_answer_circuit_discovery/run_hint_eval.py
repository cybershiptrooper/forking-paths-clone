"""Eval driver for hint-removal masks: builds the right eval_log_alpha
invocation for a mask file and runs it.

- learned SNP masks: matched-target evaluation only (top-k at the mask's
  own training target sparsity).
- random-baseline masks (score_readout raw_score, random_baseline_seed
  in metadata): evaluated at --all_sparsities.
- Thought Anchors masks (no SNP metadata): evaluated at
  --all_sparsities with --force_freeze_prompt and the dataset's letters.

Usage:
    uv run python -m expts.direct_answer_circuit_discovery.run_hint_eval \
        --mask_path <mask.json> --eval_dir <dir> \
        [--data_path <hinted collection>]
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
    ap.add_argument("--data_path", default=None,
                    help="Override the mask's data_path (needed for TA "
                    "masks whose metadata lacks it).")
    ap.add_argument("--all_sparsities", default="0.05,0.2,0.5")
    args = ap.parse_args()

    with open(args.mask_path) as f:
        mask = json.load(f)
    meta = mask.get("metadata", {})
    data_path = args.data_path or meta.get("data_path")
    prompt_index = meta.get("prompt_index")
    step = meta.get("analysis_sentence_step")
    sentences_after_prefix = meta.get("sentences_after_prefix", 5) or 0
    if data_path is None or prompt_index is None or step is None:
        raise ValueError(
            f"mask {args.mask_path} lacks data_path/prompt_index/"
            f"analysis_sentence_step metadata; pass --data_path and use a "
            f"mask saved by this experiment's configs."
        )

    is_random = "random_baseline_seed" in meta
    is_snp = meta.get("objective", "").startswith("answer_probe") and not is_random

    stem = os.path.splitext(os.path.basename(args.mask_path))[0]
    out = os.path.join(args.eval_dir, f"{stem}.eval.json")
    if os.path.exists(out):
        print(f"exists: {out}")
        return

    cmd = [
        sys.executable, "-m",
        "expts.direct_answer_circuit_discovery.eval_log_alpha",
        "--mask_path", args.mask_path,
        "--model_name", meta.get("model_name") or mask.get("model_name")
        or "Qwen/Qwen3-8B",
        "--data_path", data_path,
        "--prompt_index", str(prompt_index),
        "--analysis_sentence_step", str(step),
        "--sentences_after_prefix", str(sentences_after_prefix),
        "--sentence_gap", str(meta.get("sentence_gap", 0) or 0),
        "--output", out,
    ]
    if is_snp:
        cmd += ["--top_k_sparsities", str(meta["target_sparsity"])]
    else:
        cmd += ["--top_k_sparsities", args.all_sparsities]
        if not meta.get("num_frozen_prompt_sentences"):
            cmd += ["--force_freeze_prompt"]
        # letters override for masks without probe metadata (TA)
        if not meta.get("answer_letters"):
            with open(data_path) as f:
                rec = json.load(f)[prompt_index]
            letters = ",".join(rec["all_letters"])
            cmd += ["--answer_letters", letters]

    os.makedirs(args.eval_dir, exist_ok=True)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
