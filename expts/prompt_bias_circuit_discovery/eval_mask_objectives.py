"""How well does each mask of the mask-assisted judge study do on the
quantity it was optimised for?

For every example of the dataset, at the analysis point used for the masks
(the whole reasoning up to ``</think>``; the probe forces
``</think>\\n\\n**Final answer (`` and reads the letter logits), this
script evaluates the *binarised* prompt-to-trace masks the judge sees
(top 20 percent of the pool kept, exactly as in ``mask_summaries.kept_cells``):

- ``p2t_rg``: subnetwork probing, reward gap towards the trace's answer;
- ``p2t_kl``: subnetwork probing, answer-KL objective;
- ``ta``: Thought Anchors scores, top 20 percent kept;
- ``random``: ``--n_random`` uniformly random masks of the same size;
- ``all_removed``: every prompt-to-trace cell removed;
- ``clean``: no mask.

and records the answer distribution, P(trace's answer) and the KL
divergence from the clean distribution. One process handles the examples
``--task, --task + --n_tasks, ...`` of the dataset (SLURM array).

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.eval_mask_objectives \
        --dataset results/prompt_bias_masks/dataset_qwen3_8b.json --name masks_qwen3_8b \
        --task 0 --n_tasks 12 --out_dir results/prompt_bias_masks/masks_qwen3_8b/objective_eval
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.masks import (
    build_gap_filter, build_mode_filter, build_causal_filter,
    build_combined_filter, build_region_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.probe import answer_probs_from_logits, build_answer_probe
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _evaluate_mask, _kl
from expts.prompt_bias_circuit_discovery.mask_summaries import kept_cells, find_mask
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--sparsity", type=float, default=0.8)
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["p2t_rg", "p2t_kl", "ta"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    D = json.load(open(args.dataset))
    res = f"results/prompt_bias_masks/{args.name}"
    ids = [r["example_id"] for r in D][args.task::args.n_tasks]
    todo = [i for i in ids if not os.path.exists(f"{args.out_dir}/ex{i:03d}.json")]
    print(f"task {args.task}/{args.n_tasks}: {len(ids)} examples, {len(todo)} to do")
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)
    for i in todo:
        rec = D[i]
        prefix_ids, sentences, _, _, _, num_prompt = _build_prefix(
            tokenizer=tok, prompt=None, data_path=args.dataset, prompt_index=i, base_answer_type="stored",
            analysis_timestep=rec["analysis_timestep"], analysis_sentence_step=None, sentences_after_prefix=0,
            min_sentence_length=10, sentence_chunk=1)
        letters = [l.strip() for l in rec["all_letters"]]  # the suffix ends in "(": no leading space
        probe = build_answer_probe(tok, suffix=PROBE_SUFFIX, answer_letters=letters)
        ti = letters.index(rec["mask_target_letter"].strip())
        num_sents = len(sentences)
        prefix_len = prefix_ids.shape[-1]
        full = torch.cat([prefix_ids.to(dev), probe.make_continuation(dev)], dim=-1)
        gap_f = build_gap_filter(num_sents, 1, device=dev)
        mode_f = build_mode_filter(num_sents, num_sents, "prefix", device=dev)
        causal_f = build_causal_filter(num_sents, device=dev)
        region_f = build_region_filter("prompt_to_trace", num_prompt, num_sents, device=dev)
        combined = build_combined_filter(gap_f, mode_f, causal_f, region_f)
        valid = ~combined.bool()
        valid_np = valid.cpu().numpy()
        n_valid = int(valid.sum().item())
        n_keep = max(0, int(round((1.0 - args.sparsity) * n_valid)))
        token_to_sent = torch.full((full.shape[-1],), -1, dtype=torch.long, device=dev)
        for k, s in enumerate(sentences):
            token_to_sent[s.start:s.end + 1] = k

        def run(binary):
            return _evaluate_mask(model, layers, num_heads, full, prefix_len, probe, binary, token_to_sent,
                                  combined, dev, True, "sdpa")

        with torch.no_grad():
            clean_logits = model(full).logits
        clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
        del clean_logits
        clear_cuda()

        def summarise(p):
            return {"answer_probs": p.tolist(), "p_target": float(p[ti]), "kl": _kl(clean_p, p)}

        out = {"example_id": i, "tag": rec["tag"], "case": rec["case"], "family": rec["family"], "subset": rec.get("subset") or "",
               "is_positive": rec["is_positive"], "target_letter": letters[ti], "num_prompt_sentences": num_prompt,
               "num_sentences": num_sents, "n_valid": n_valid, "n_keep": n_keep, "sparsity": args.sparsity,
               "clean": summarise(clean_p), "conditions": {}}
        ones = torch.ones(num_sents, num_sents, device=dev)
        m = ones.clone(); m[valid] = 0.0
        out["conditions"]["all_removed"] = summarise(run(m))
        for meth in args.methods:
            f = find_mask(res, i, meth)
            if f is None:
                print(f"example {i}: no {meth} mask")
                continue
            obj = json.load(open(f))
            scores = np.array(obj["scores"], float)
            assert scores.shape[0] == num_sents, (i, meth, scores.shape, num_sents)
            keep, valid_k = kept_cells(scores, num_prompt, "p2t", args.sparsity)
            assert (valid_k == valid_np).all(), (i, meth, "pool mismatch")
            m = ones.clone()
            m[valid] = torch.as_tensor(keep, device=dev, dtype=torch.float32)[valid]
            r = summarise(run(m))
            r["n_kept"] = int(keep.sum()); r["mask_file"] = f
            out["conditions"][meth] = r
        valid_idx = np.flatnonzero(valid_np.flatten())
        rand = []
        for k in range(args.n_random):
            rng = np.random.default_rng((args.seed, i, int(round(args.sparsity * 1000)), k))
            keep_idx = rng.choice(valid_idx, size=n_keep, replace=False)
            b = torch.zeros(num_sents * num_sents, device=dev)
            b[torch.as_tensor(keep_idx, device=dev)] = 1.0
            b = b.view(num_sents, num_sents)
            m = ones.clone(); m[valid] = b[valid]
            rand.append(summarise(run(m)))
        out["conditions"]["random"] = {"samples": rand, "p_target": float(np.mean([r["p_target"] for r in rand])),
                                       "kl": float(np.mean([r["kl"] for r in rand]))}
        json.dump(out, open(f"{args.out_dir}/ex{i:03d}.json", "w"), indent=1)
        print(f"example {i}: clean p_target={out['clean']['p_target']:.3f} " +
              " ".join(f"{k}={v['p_target']:.3f}/kl{v['kl']:.4f}" for k, v in out["conditions"].items()))
        del full
        clear_cuda()
    remove_handles(handles)


if __name__ == "__main__":
    main()
