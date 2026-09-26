"""Evaluate the "reduce P(trace's answer)" masks of
``build_reduce_mask_experiment.py`` at their own cut and budget.

For every trained mask: the prefix is rebuilt exactly as in training (same
cut, prompt chunks, excluded sink tokens), the frozen filter is rebuilt from
the mask's metadata (region, frozen option chunks, gap, causal), and the
mask is binarised by removing the ``target_sparsity`` fraction of the
learnable pool with the lowest log-alpha. P(trace's answer) at the probe is
recorded with no mask, with the learned binary mask, with ``--n_random``
random removals of the same size from the same pool, and with the whole pool
removed.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.eval_reduce_masks --name reduce_masks_qwen3_8b --task 0 --n_tasks 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from utils.masks import NodeMask, build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter, build_region_filter
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.probe import answer_probs_from_logits, build_answer_probe
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _evaluate_mask, _kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    res = f"results/prompt_bias_v2/reduce_masks/{args.name}"
    root = f"expts/prompt_bias_circuit_discovery/configs/{args.name}"
    out_dir = f"{res}/eval"; os.makedirs(out_dir, exist_ok=True)
    D = json.load(open(f"{res}/dataset.json"))
    cfgs = sorted(glob.glob(f"{root}/*.yaml"))[args.task::args.n_tasks]
    todo = []
    for c in cfgs:
        stem = os.path.basename(c)[:-5]
        masks = sorted(glob.glob(f"{res}/masks/{stem}*.json"))
        if masks and not os.path.exists(f"{out_dir}/{stem}.json"):
            todo.append((c, stem, masks[0]))
    print(f"task {args.task}/{args.n_tasks}: {len(cfgs)} configs, {len(todo)} trained and not yet evaluated", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers)); num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)
    rng = np.random.default_rng(args.seed)
    for c, stem, mpath in todo:
        cfg = yaml.safe_load(open(c)); rec = D[cfg["prompt_index"]]
        nm = NodeMask.from_json(mpath)
        prefix_ids, sentences, _, _, _, n_p = _build_prefix(
            tokenizer=tok, prompt=None, data_path=cfg["data_path"], prompt_index=cfg["prompt_index"], base_answer_type="stored",
            analysis_timestep=cfg.get("analysis_timestep"), analysis_sentence_step=cfg.get("analysis_sentence_step"), sentences_after_prefix=0,
            min_sentence_length=10, sentence_chunk=1)
        S = len(sentences)
        assert S == len(nm.sentences), (stem, S, len(nm.sentences))
        letters = [l.strip() for l in cfg["answer_letters"]]
        probe = build_answer_probe(tok, suffix=cfg["probe_suffix"], answer_letters=letters)
        ti = letters.index(rec["mask_target_letter"])
        prefix_len = prefix_ids.shape[-1]
        full = torch.cat([prefix_ids.to(dev), probe.make_continuation(dev)], dim=-1)
        combined = build_combined_filter(build_gap_filter(S, cfg["sentence_gap"], device=dev), build_mode_filter(S, S, cfg["mask_mode"], device=dev),
                                         build_causal_filter(S, device=dev), None)
        region = build_region_filter(cfg["learnable_region"], n_p, S, device=dev, frozen_key_sentences=cfg.get("frozen_key_sentences"))
        combined = combined | region
        pool = ~combined
        n_pool = int(pool.sum().item()); n_remove = int(round(cfg["target_sparsity"] * n_pool))
        scores = torch.tensor(np.array(nm.scores, dtype=np.float32), device=dev)
        if scores.dim() == 3:
            scores = scores.mean(0)
        assert scores.shape == (S, S), (stem, scores.shape)
        token_to_sent = torch.full((full.shape[-1],), -1, dtype=torch.long, device=dev)
        for si, s in enumerate(sentences):
            token_to_sent[s.start:s.end + 1] = si

        def run(m):
            return _evaluate_mask(model, layers, num_heads, full, prefix_len, probe, m, token_to_sent, combined, dev, True, "sdpa")

        with torch.no_grad():
            clean_p = answer_probs_from_logits(model(full).logits, probe, prefix_len).cpu()
        # learned: remove the n_remove lowest log-alpha cells of the pool
        flat = torch.where(pool, scores, torch.full_like(scores, float("inf"))).flatten()
        idx = torch.argsort(flat)[:n_remove]
        m_learned = torch.ones(S * S, device=dev); m_learned[idx] = 0.0; m_learned = m_learned.view(S, S)
        p_learned = run(m_learned)
        m_all = torch.ones(S, S, device=dev); m_all[pool] = 0.0
        p_all = run(m_all)
        pool_idx = torch.nonzero(pool.flatten()).flatten().cpu().numpy()
        p_rand = []
        for _ in range(args.n_random):
            pick = rng.choice(pool_idx, n_remove, replace=False)
            m = torch.ones(S * S, device=dev); m[torch.tensor(pick, device=dev)] = 0.0
            p_rand.append(run(m.view(S, S)))
        removed = idx.cpu().numpy()
        rows_removed = (removed // S); cols_removed = (removed % S)
        out = dict(stem=stem, example_id=cfg["prompt_index"], tag=rec["tag"], is_positive=rec["is_positive"], trace_answer=rec["mask_target_letter"],
                   pool=cfg["learnable_region"], cut=("think" if cfg.get("analysis_timestep") is not None else "half"), target_sparsity=cfg["target_sparsity"],
                   n_sentences=S, n_prompt=n_p, n_pool=n_pool, n_removed=n_remove, prefix_tokens=int(prefix_len),
                   clean=dict(p_target=float(clean_p[ti]), probs=clean_p.tolist()),
                   learned=dict(p_target=float(p_learned[ti]), probs=p_learned.tolist(), kl=_kl(clean_p, p_learned)),
                   all_removed=dict(p_target=float(p_all[ti]), probs=p_all.tolist(), kl=_kl(clean_p, p_all)),
                   random=[dict(p_target=float(p[ti]), kl=_kl(clean_p, p)) for p in p_rand],
                   removed_cells_by_key_kind=({str(k): int(v) for k, v in zip(*np.unique([rec["prompt_chunk_kinds"][j] if j < n_p else "reasoning" for j in cols_removed], return_counts=True))} if n_remove else {}),
                   removed_name_cells=int(sum(1 for j in cols_removed if j < n_p and rec["prompt_chunk_kinds"][j] == "question" and nm.sentences[j]["text"].strip() in {rec["name"], rec["name"].split()[0], rec["name"].split()[-1]})),
                   removed_rows_position=(sorted(int(x) for x in (rows_removed - n_p)) if n_remove else []))
        json.dump(out, open(f"{out_dir}/{stem}.json", "w"), default=float)
        print(f"{stem}: pool {n_pool} removed {n_remove} | P(trace) clean {out['clean']['p_target']:.3f} learned {out['learned']['p_target']:.3f} "
              f"random {np.mean([r['p_target'] for r in out['random']]):.3f} all-removed {out['all_removed']['p_target']:.3f}", flush=True)
        del full; clear_cuda()
    remove_handles(handles)


if __name__ == "__main__":
    main()
