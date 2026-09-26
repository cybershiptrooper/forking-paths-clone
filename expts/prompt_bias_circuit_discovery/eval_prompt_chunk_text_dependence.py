"""How much does the reasoning *text* of a single rollout depend on each
prompt chunk, and on the applicant's name in particular?

For every example of the same-decision sweep dataset, the prompt is chunked
into sentences with the applicant's name forced to be its own chunk, the
stored reasoning (up to ``</think>``) is teacher-forced, and for every prompt
chunk j one forward pass removes every read from a reasoning sentence to
chunk j (renormalised attention, as in the sweep). The readout is the
next-token distribution at every reasoning position: per reasoning sentence
the mean KL(clean || ablated) per token and the mean change in the log
probability of the token the model actually wrote; plus P(admit) at the
forced-answer probe appended after the reasoning.

The name is forced to be its own chunk at *every* mention in the
application (full name, first name or last name), and the ``name_all``
condition removes the reasoning's reads of all of them; ``name_all_unread``
also removes every prompt token's reads of them (the name then influences
nothing but its own positions), which is the closest attention-level
counterpart of deleting the name.

At a set of cuts (prompt only, 0.25 / 0.5 / 0.75 of the reasoning, the last
sentence before the decision sentence, ``</think>``) both arms also record
P(admit) at the probe with no mask, and under ``name_all`` and
``name_all_unread``: the difference between the arms is the name-swap
effect on the forced answer given this trace's prefix.

Two arms per example: ``black`` (the prompt and trace as generated) and
``white`` (the same trace, with the name swapped, teacher-forced under the
White-name prompt of the same input). The difference between the arms in
the name chunk's effect is a within-trace control: reads of the name that
only copy or restate it cost the same under either name, reads that changed
the reasoning because of what the name suggests do not.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.eval_prompt_chunk_text_dependence \
        --dataset results/prompt_bias_v2/same_decision_sweep/dataset_qwen3_8b_black.json --task 0 --n_tasks 8 \
        --out_dir results/prompt_bias_v2/same_decision_sweep/text_dependence_qwen3_8b_black
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.masks import build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from utils.utils import get_attention_module
from expts.direct_answer_circuit_discovery.probe import answer_probs_from_logits, build_answer_probe
from expts.direct_answer_circuit_discovery.learn import split_with_prompt_chunks, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _binary_to_per_layer_masks
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX
from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt


def swap_name(text, a, b):
    fa, la = a.split()[0], a.split()[-1]
    fb, lb = b.split()[0], b.split()[-1]
    text = text.replace(a, b)
    text = re.sub(rf"\b{re.escape(fa)}\b", fb, text)
    return re.sub(rf"\b{re.escape(la)}\b", lb, text)


def span(rec):
    s = rec["cue_char_span"]
    return json.loads(s) if isinstance(s, str) else s


def masked_logits(model, layers, num_heads, full, binary, token_to_sent, combined):
    masks = _binary_to_per_layer_masks(binary, layers, num_heads)
    for l in layers:
        attn = get_attention_module(model, l)
        attn._circuit_mask = masks[l]; attn._token_to_sent = token_to_sent; attn._gap_filter = combined; attn._renormalize_masked_attn = True
    try:
        with torch.no_grad():
            return model(full).logits
    finally:
        for l in layers:
            attn = get_attention_module(model, l)
            for a in ("_circuit_mask", "_token_to_sent", "_gap_filter"):
                if hasattr(attn, a):
                    delattr(attn, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rollouts", default="results/prompt_bias_v2/rollouts_confirm_admission_full/qwen3_8b_admission_full_confirm_shard*of8.json")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--only_ids", nargs="*", type=int, default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    D = json.load(open(args.dataset))
    ids = [r["example_id"] for r in D][args.task::args.n_tasks]
    if args.only_ids:
        ids = [i for i in ids if i in set(args.only_ids)]
    todo = [i for i in ids if not os.path.exists(f"{args.out_dir}/ex{i:03d}.json")]
    print(f"task {args.task}/{args.n_tasks}: {len(ids)} examples, {len(todo)} to do", flush=True)
    if not todo:
        return
    W = {}
    need = {D[i]["base_uid"] for i in todo}
    for f in glob.glob(args.rollouts):
        for r in json.load(open(f)):
            if r["uid"] in need:
                W[r["uid"]] = {k: v for k, v in r.items() if k != "rollouts"}
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)
    for i in todo:
        t0 = time.time()
        rec = D[i]
        letters = [l.strip() for l in rec["all_letters"]]
        probe = build_answer_probe(tok, suffix=PROBE_SUFFIX, answer_letters=letters)
        ti = letters.index(rec["mask_target_letter"].strip())
        reasoning_text = tok.decode(rec["output_token_ids"][:rec["analysis_timestep"]])
        white = W[rec["base_uid"]]
        out = {"example_id": i, "tag": rec["tag"], "is_positive": rec["is_positive"], "name": rec["name"], "white_name": white["name"], "arms": {}}
        for arm, prec, text in [("black", rec, reasoning_text), ("white", white, swap_name(reasoning_text, rec["name"], white["name"]))]:
            fa_, la_ = prec["name"].split()[0], prec["name"].split()[-1]
            forced = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(prec['name'])}\b|\b{re.escape(fa_)}\b|\b{re.escape(la_)}\b", prec["question"])]
            chunks, _ = chunk_prompt(tok, prec, "sentence", forced_spans=forced)
            spans = [(c["start"], c["end"]) for c in chunks]
            name_set = {prec["name"], fa_, la_}
            name_idxs = [k for k, c in enumerate(chunks) if c["text"].strip() in name_set]
            assert name_idxs and chunks[name_idxs[0]]["text"].strip() == prec["name"], (i, arm, [c["text"] for c in chunks])
            name_idx = name_idxs[0]
            prompt_ids = torch.tensor(prec["prompt_token_ids"])
            reason_ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
            if arm == "black":
                out["retokenised_equal"] = reason_ids.tolist() == rec["output_token_ids"][:rec["analysis_timestep"]]
            prefix = torch.cat([prompt_ids, reason_ids]).unsqueeze(0)
            prompt_len = prompt_ids.shape[-1]
            prefix_len = prefix.shape[-1]
            sentences = split_with_prompt_chunks(prefix[0], tok, prompt_len, spans, 10)
            n_p = sum(1 for s in sentences if s.start < prompt_len)
            assert n_p == len(chunks), (i, arm, n_p, len(chunks))
            S = len(sentences)
            full = torch.cat([prefix.to(dev), probe.make_continuation(dev)], dim=-1)
            combined = build_combined_filter(build_gap_filter(S, 1, device=dev), build_mode_filter(S, S, "prefix", device=dev),
                                             build_causal_filter(S, device=dev), None)
            token_to_sent = torch.full((full.shape[-1],), -1, dtype=torch.long, device=dev)
            for si, s in enumerate(sentences):
                token_to_sent[s.start:s.end + 1] = si
            # reasoning positions: predictions made at positions prompt_len-1 .. prefix_len-2 for tokens prompt_len .. prefix_len-1
            tgt = full[0, prompt_len:prefix_len]
            sent_of_tgt = token_to_sent[prompt_len:prefix_len] - n_p  # reasoning sentence index per target token (-1 - n_p if none)
            fa, la = prec["name"].split()[0], prec["name"].split()[-1]
            sent_texts = [tok.decode(full[0, s.start:s.end + 1].tolist()) for s in sentences[n_p:]]
            has_name = [bool(re.search(rf"\b{re.escape(fa)}\b|\b{re.escape(la)}\b", t)) for t in sent_texts]
            with torch.no_grad():
                logits = model(full).logits
            clean_lp = F.log_softmax(logits[0, prompt_len - 1:prefix_len - 1].float(), dim=-1)
            clean_tok_lp = clean_lp.gather(1, tgt[:, None])[:, 0]
            clean_p = clean_lp.exp()
            p_admit_clean = float(answer_probs_from_logits(logits, probe, prefix_len).cpu()[ti])
            del logits
            arm_out = {"n_prompt_chunks": n_p, "chunk_texts": [c["text"] for c in chunks], "chunk_kinds": [c["kind"] for c in chunks], "name_chunk": name_idx, "name_chunks": name_idxs,
                       "n_reasoning_sentences": S - n_p, "sentence_has_name": has_name, "sentence_n_tokens": [int((sent_of_tgt == k).sum()) for k in range(S - n_p)],
                       "clean_sentence_logp": [float(clean_tok_lp[sent_of_tgt == k].mean()) if (sent_of_tgt == k).any() else None for k in range(S - n_p)],
                       "p_admit_clean": p_admit_clean, "chunks": {}}
            conds = [(str(j), [j], False) for j in range(n_p)] + [("name_all", name_idxs, False), ("name_all_unread", name_idxs, True), ("all", list(range(n_p)), False)]
            for name, cols, unread in conds:
                m = torch.ones(S, S, device=dev)
                for j in cols:
                    m[(0 if unread else n_p):, j] = 0.0
                logits = masked_logits(model, layers, num_heads, full, m, token_to_sent, combined)
                lp = F.log_softmax(logits[0, prompt_len - 1:prefix_len - 1].float(), dim=-1)
                kl = (clean_p * (clean_lp - lp)).sum(-1)
                dlp = lp.gather(1, tgt[:, None])[:, 0] - clean_tok_lp
                p_admit = float(answer_probs_from_logits(logits, probe, prefix_len).cpu()[ti])
                del logits, lp
                arm_out["chunks"][name] = {
                    "sentence_kl": [float(kl[sent_of_tgt == k].mean()) if (sent_of_tgt == k).any() else None for k in range(S - n_p)],
                    "sentence_dlogp": [float(dlp[sent_of_tgt == k].mean()) if (sent_of_tgt == k).any() else None for k in range(S - n_p)],
                    "total_kl": float(kl.sum()), "mean_kl": float(kl.mean()), "p_admit": p_admit}
            del clean_lp, clean_p, full
            clear_cuda()
            # P(admit) at a set of cuts, with no mask and with the name chunks unread
            n_r = S - n_p
            cut_list = [("prompt", 0), ("0.25", max(1, round(0.25 * n_r))), ("0.5", max(1, round(0.5 * n_r))), ("0.75", max(1, round(0.75 * n_r))),
                        ("before_decision", max(1, n_r - 1)), ("think", n_r)]
            arm_out["cuts"] = {}
            for cname, k in cut_list:
                cut_len = prompt_len if k == 0 else sentences[n_p + k - 1].end + 1
                pre = prefix[:, :cut_len]
                sents_k = split_with_prompt_chunks(pre[0], tok, prompt_len, spans, 10)
                Sk = len(sents_k)
                assert sum(1 for s_ in sents_k if s_.start < prompt_len) == n_p
                fullk = torch.cat([pre.to(dev), probe.make_continuation(dev)], dim=-1)
                comb_k = build_combined_filter(build_gap_filter(Sk, 1, device=dev), build_mode_filter(Sk, Sk, "prefix", device=dev),
                                               build_causal_filter(Sk, device=dev), None)
                t2s_k = torch.full((fullk.shape[-1],), -1, dtype=torch.long, device=dev)
                for si, s_ in enumerate(sents_k):
                    t2s_k[s_.start:s_.end + 1] = si
                with torch.no_grad():
                    lg = model(fullk).logits
                row = {"k": k, "n_tokens": int(cut_len), "clean": float(answer_probs_from_logits(lg, probe, cut_len).cpu()[ti])}
                del lg
                for name, unread in [("name_all", False), ("name_all_unread", True)]:
                    if k == 0 and not unread:
                        continue
                    m = torch.ones(Sk, Sk, device=dev)
                    for j in name_idxs:
                        m[(0 if unread else n_p):, j] = 0.0
                    lg = masked_logits(model, layers, num_heads, fullk, m, t2s_k, comb_k)
                    row[name] = float(answer_probs_from_logits(lg, probe, cut_len).cpu()[ti])
                    del lg
                arm_out["cuts"][cname] = row
                del fullk
                clear_cuda()
            out["arms"][arm] = arm_out
        json.dump(out, open(f"{args.out_dir}/ex{i:03d}.json", "w"))
        b = out["arms"]["black"]; w = out["arms"]["white"]
        print(f"example {i} ({b['n_reasoning_sentences']} sentences, {len(b['name_chunks'])} name mentions, {time.time() - t0:.0f}s, retok {out['retokenised_equal']}): "
              f"name_all mean KL black {b['chunks']['name_all']['mean_kl']:.4f} white {w['chunks']['name_all']['mean_kl']:.4f}; unread black {b['chunks']['name_all_unread']['mean_kl']:.4f}; "
              f"all-chunks black {b['chunks']['all']['mean_kl']:.4f}; P(admit) at 0.25: black {b['cuts']['0.25']['clean']:.3f} white {w['cuts']['0.25']['clean']:.3f} "
              f"name unread {b['cuts']['0.25']['name_all_unread']:.3f}; prompt only: black {b['cuts']['prompt']['clean']:.3f} white {w['cuts']['prompt']['clean']:.3f}", flush=True)
    remove_handles(handles)


if __name__ == "__main__":
    main()
