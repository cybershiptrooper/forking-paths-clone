"""Per-instance job wrapper: one model load, all work for one stage.

Stages:
- ``cheap``: suppress + TA scoring, both attention baselines, then evals
  (reference, suppress, ta, attn_last, attn_next, oracle).
- ``snp``: column SNP training (one training, target keep 15), then its eval.

Usage (one GPU):
    uv run python -m expts.external_compression.run_instance \
        --instance <id> --stage cheap
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from expts.external_compression import baselines as baselines_mod
from expts.external_compression import evaluate as evaluate_mod
from expts.external_compression import score_methods as sm
from expts.external_compression.common import DATA_DIR, MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument(
        "--stage", choices=["cheap", "snp", "snp_uniform_l0"], required=True,
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()

    from utils.utils import set_seed
    set_seed(42)

    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = {r["instance_id"]: r for r in json.load(f)}
    inst = instances[args.instance]

    from transformers import AutoTokenizer
    from expts.direct_answer_circuit_discovery.learn import load_model_eager
    from utils.circuit_eval import install_clean_sdpa_forward, remove_handles

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    t_all = time.time()

    if args.stage == "cheap":
        model, _ = load_model_eager(
            args.model_name, device="cuda", gradient_checkpointing=False,
        )
        inputs = sm.build_inputs(inst, tokenizer)
        out_dir = os.path.join(sm.SCORES_DIR, args.instance)
        os.makedirs(out_dir, exist_ok=True)

        for method, fn in [("suppress", sm.suppress_scores), ("ta", sm.ta_scores)]:
            out_path = os.path.join(out_dir, f"{method}.json")
            if os.path.exists(out_path):
                print(f"skip {method}: exists")
                continue
            t0 = time.time()
            scores = fn(model, inputs)
            with open(out_path, "w") as f:
                json.dump({
                    "instance_id": args.instance, "method": method,
                    "scores": scores, "num_rankable": len(inputs["spans"]) - 1,
                    "seconds": time.time() - t0, "model_name": args.model_name,
                }, f, indent=2)
            print(f"wrote {out_path} ({time.time() - t0:.0f}s)")

        # Plain forwards from here on: patch every layer to the SDPA path.
        handles = install_clean_sdpa_forward(model)
        try:
            if not (
                os.path.exists(os.path.join(out_dir, "attn_last.json"))
                and os.path.exists(os.path.join(out_dir, "attn_next.json"))
            ):
                baselines_mod.run(
                    args.instance, model=model, tokenizer=tokenizer,
                    model_name=args.model_name,
                )
            evaluate_mod.run(
                inst, model, tokenizer,
                ["reference", "suppress", "ta", "attn_last", "attn_next", "oracle"],
            )
        finally:
            remove_handles(handles)

    else:  # snp or snp_uniform_l0
        uniform = args.stage == "snp_uniform_l0"
        method_name = "colsnp_uniform_l0" if uniform else "colsnp"
        model, _ = load_model_eager(
            args.model_name, device="cuda", gradient_checkpointing=True,
        )
        inputs = sm.build_inputs(inst, tokenizer)
        out_dir = os.path.join(sm.SCORES_DIR, args.instance)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{method_name}.json")
        if not os.path.exists(out_path):
            t0 = time.time()
            scores = sm.colsnp_scores(
                model, tokenizer, inputs, args.instance, uniform_l0=uniform,
            )
            with open(out_path, "w") as f:
                json.dump({
                    "instance_id": args.instance, "method": method_name,
                    "scores": scores, "num_rankable": len(inputs["spans"]) - 1,
                    "seconds": time.time() - t0, "model_name": args.model_name,
                    "target_keep": sm.SNP_TARGET_KEEP,
                    "uniform_column_l0": uniform,
                }, f, indent=2)
            print(f"wrote {out_path} ({time.time() - t0:.0f}s)")
        model.eval()
        handles = install_clean_sdpa_forward(model)
        try:
            evaluate_mod.run(inst, model, tokenizer, [method_name])
        finally:
            remove_handles(handles)

    print(f"STAGE_DONE instance={args.instance} stage={args.stage} "
          f"total={time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
