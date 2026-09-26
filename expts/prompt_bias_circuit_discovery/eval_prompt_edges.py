"""Direct ablation of prompt-to-trace attention cells, one prompt at a time.

For a prefix (prompt sentences followed by reasoning sentences up to the
analysis point) and the direct-answer probe, this script measures the
answer distribution after removing

1. every prompt-to-trace cell at once (the ceiling of the pool),
2. one prompt column at a time: every reasoning sentence stops reading
   prompt sentence j (``column_ablation``),
3. every prompt column except j (``column_only``): the reasoning keeps
   reading prompt sentence j and nothing else of the prompt,
4. one cell (i, j) at a time (``edge_ablation``): reasoning sentence i
   stops reading prompt sentence j.

Cells outside the prompt-to-trace pool (reasoning-to-reasoning,
prompt-to-prompt, the diagonal) stay at 1.0 throughout, as in training
with ``learnable_region: prompt_to_trace``.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.eval_prompt_edges \
        --data_path data/collection/qwen3_8b/hinted_merged.json --prompt_index 1 \
        --analysis_sentence_step 50 --sentences_after_prefix 5 --sentence_gap 1 \
        --target_letter C --output results/prompt_bias/edges/hint_p01.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from utils.masks import (
    build_gap_filter, build_mode_filter, build_causal_filter,
    build_combined_filter, build_region_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.probe import (
    answer_probs_from_logits, build_answer_probe, DEFAULT_SUFFIX,
    DEFAULT_ANSWER_LETTERS,
)
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _evaluate_mask, _kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--prompt_index", type=int, required=True)
    ap.add_argument("--analysis_sentence_step", type=int, default=None)
    ap.add_argument("--analysis_timestep", type=int, default=None,
                    help="Token offset from the end of the prompt at which the prefix is cut (e.g. the position of "
                    "</think>: the whole reasoning is the prefix and every sentence is masked). Overrides "
                    "--analysis_sentence_step; --sentences_after_prefix is ignored.")
    ap.add_argument("--sentences_after_prefix", type=int, default=5)
    ap.add_argument("--probe_suffix", default=DEFAULT_SUFFIX)
    ap.add_argument("--no_letter_space", action="store_true",
                    help="Do not prepend a space to each answer letter (for suffixes ending in '(').")
    ap.add_argument("--sentence_gap", type=int, default=1)
    ap.add_argument("--answer_letters", default=None,
                    help="Comma-separated, e.g. ' A, B'. Default: record's all_letters.")
    ap.add_argument("--target_letter", default=None,
                    help="Letter whose probability is reported as p_target "
                    "(the answer the model gives without the cue).")
    ap.add_argument("--skip_edges", action="store_true",
                    help="Only run the column ablations.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    if args.analysis_timestep is None and args.analysis_sentence_step is None:
        raise SystemExit("one of --analysis_timestep / --analysis_sentence_step is required")
    if args.analysis_timestep is not None:
        args.analysis_sentence_step = None
        args.sentences_after_prefix = 0

    tok = AutoTokenizer.from_pretrained(args.model_name)
    prefix_ids, sentences, _, correct_answer, _, num_prompt = _build_prefix(
        tokenizer=tok, prompt=None, data_path=args.data_path,
        prompt_index=args.prompt_index, base_answer_type="stored",
        analysis_timestep=args.analysis_timestep, analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        min_sentence_length=10, sentence_chunk=1,
    )
    with open(args.data_path) as f:
        rec = json.load(f)[args.prompt_index]
    sp = "" if args.no_letter_space else " "
    if args.answer_letters:
        letters = [(sp + l.strip()) for l in args.answer_letters.split(",")]
    else:
        letters = [sp + l for l in rec.get("all_letters", ["A", "B", "C", "D"])]
    probe = build_answer_probe(tok, suffix=args.probe_suffix, answer_letters=letters)
    stripped = [l.strip() for l in probe.answer_letters]
    target_letter = args.target_letter or rec.get("target_letter") or correct_answer
    target_id = stripped.index(target_letter.strip()) if target_letter in stripped else None

    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    input_ids = prefix_ids.to(dev)
    full = torch.cat([input_ids, probe.make_continuation(dev)], dim=-1)
    handles = install_clean_sdpa_forward(model)

    gap_f = build_gap_filter(num_sents, args.sentence_gap, device=dev)
    mode_f = build_mode_filter(num_sents, num_sents, "prefix", device=dev)
    causal_f = build_causal_filter(num_sents, device=dev)
    region_f = build_region_filter("prompt_to_trace", num_prompt, num_sents, device=dev)
    combined = build_combined_filter(gap_f, mode_f, causal_f, region_f)
    valid = ~combined
    n_valid = int(valid.sum().item())
    token_to_sent = torch.full((full.shape[-1],), -1, dtype=torch.long, device=dev)
    for idx, s in enumerate(sentences):
        token_to_sent[s.start:s.end + 1] = idx
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads

    def run(binary):
        return _evaluate_mask(model, layers, num_heads, full, prefix_len, probe,
                              binary, token_to_sent, combined, dev, True, "sdpa")

    def summarise(p, clean):
        return {"answer_probs": p.tolist(), "kl": _kl(clean, p),
                "p_target": (float(p[target_id]) if target_id is not None else None)}

    with torch.no_grad():
        clean_logits = model(full).logits
    clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
    del clean_logits
    clear_cuda()
    ones = torch.ones(num_sents, num_sents, device=dev)

    out = {
        "data_path": args.data_path, "prompt_index": args.prompt_index,
        "analysis_sentence_step": args.analysis_sentence_step,
        "analysis_timestep": args.analysis_timestep,
        "sentences_after_prefix": args.sentences_after_prefix,
        "probe_suffix": args.probe_suffix,
        "sentence_gap": args.sentence_gap, "num_prompt_sentences": num_prompt,
        "num_sentences": num_sents, "n_valid_cells": n_valid,
        "answer_letters": probe.answer_letters, "target_letter": target_letter,
        "sentences": [{"idx": i, "start": s.start, "end": s.end,
                       "is_prompt": i < num_prompt,
                       "text": tok.decode(input_ids[0, s.start:s.end + 1])}
                      for i, s in enumerate(sentences)],
        "clean": summarise(clean_p, clean_p),
    }
    # 1. whole pool removed
    m = ones.clone(); m[valid] = 0.0
    out["all_pool_ablated"] = summarise(run(m), clean_p)
    # 2./3. per prompt column
    cols = []
    for j in range(num_prompt):
        m = ones.clone(); m[num_prompt:, j] = 0.0
        abl = summarise(run(m), clean_p)
        m = ones.clone(); m[valid] = 0.0; m[num_prompt:, j] = 1.0
        only = summarise(run(m), clean_p)
        cols.append({"j": j, "column_ablation": abl, "column_only": only})
        print(f"col {j:2d} ablated: p_target={abl['p_target']} kl={abl['kl']:.4f} | "
              f"only: p_target={only['p_target']} kl={only['kl']:.4f}")
    out["columns"] = cols
    # 4. per cell
    if not args.skip_edges:
        edges = []
        for i in range(num_prompt, num_sents):
            for j in range(num_prompt):
                if not bool(valid[i, j]):
                    continue
                m = ones.clone(); m[i, j] = 0.0
                r = summarise(run(m), clean_p)
                edges.append({"i": i, "j": j, "kl": r["kl"], "p_target": r["p_target"],
                              "answer_probs": r["answer_probs"]})
        out["edges"] = edges
    remove_handles(handles)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"clean p_target={out['clean']['p_target']}  all-pool p_target="
          f"{out['all_pool_ablated']['p_target']} kl={out['all_pool_ablated']['kl']:.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
