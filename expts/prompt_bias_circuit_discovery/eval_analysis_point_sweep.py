"""Where along the reasoning does the forced answer depend on the prompt, and
where on the earlier reasoning?

For every example of the mask-assisted judge dataset and every cut of its
reasoning (after reasoning sentence k = 0, 1, 2, ... up to ``</think>``),
the prefix is the prompt plus the first k reasoning sentences, the probe
forces ``</think>\\n\\n**Final answer (`` and reads the letter logits, and
four forward passes are made:

- ``clean``: no mask;
- ``no_prompt``: every read from a reasoning sentence to a question segment
  removed (the prompt-to-trace pool the masks of the study are trained on);
- ``no_trace``: every read from a reasoning sentence to an earlier reasoning
  sentence removed (each reasoning sentence still reads the prompt and
  itself);
- ``no_both``: both pools removed.

With ``--probe_conditions`` three more forward passes per cut treat the probe
tokens (``</think>\\n\\n**Final answer (``) as one extra sentence so that
their own reads of the prompt can be removed as well:

- ``probe_no_prompt``: only the probe tokens' reads of the prompt removed
  (the reasoning keeps every read);
- ``no_prompt_probe``: ``no_prompt`` plus the probe tokens' reads of the
  prompt removed (the answer token can see the prompt only through the
  reasoning, and the reasoning cannot see it either);
- ``no_both_probe``: ``no_both`` plus the probe tokens' reads of the prompt
  removed.

Each cut row also stores the text of the last reasoning sentence in the
prefix, and the ``</think>`` row stores every reasoning sentence's text, so
that cuts can be placed relative to the sentence that states the decision.

Cells outside the pools (prompt-to-prompt, the diagonal, the probe tokens,
which are not part of any sentence) keep full attention, as in training.
The trajectory of P(trace's answer) over k locates the point at which the
model commits to its answer; the four conditions at every k say what the
answer depends on before and after that point.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.eval_analysis_point_sweep \
        --dataset results/prompt_bias_masks/dataset_qwen3_8b.json --task 0 --n_tasks 2 \
        --out_dir results/prompt_bias_masks/masks_qwen3_8b/analysis_point_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.masks import build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda
from utils.cot_analysis import split_tokens_into_sentences
from expts.direct_answer_circuit_discovery.probe import answer_probs_from_logits, build_answer_probe
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _evaluate_mask, _kl
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--max_cuts", type=int, default=60, help="above this many reasoning sentences, every second cut is evaluated")
    ap.add_argument("--only_ids", nargs="*", type=int, default=None)
    ap.add_argument("--probe_conditions", action="store_true", help="also remove the probe tokens' own reads of the prompt (three more conditions)")
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
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)
    for i in todo:
        t0 = time.time()
        rec = D[i]
        n_p = rec["n_prompt_chunks"]
        out_ids = rec["output_token_ids"][:rec["analysis_timestep"]]
        n_r = len(split_tokens_into_sentences(torch.tensor(out_ids), tok, 10))  # the split the masks use
        letters = [l.strip() for l in rec["all_letters"]]
        probe = build_answer_probe(tok, suffix=PROBE_SUFFIX, answer_letters=letters)
        ti = letters.index(rec["mask_target_letter"].strip())
        step = 1 if n_r <= args.max_cuts else 2
        # sentence cuts inside the think block, plus the exact </think> position the masks were trained at
        # (the sentence split of the whole trace can end the last sentence a few tokens before or after </think>)
        think_pos = len(rec["prompt_token_ids"]) + rec["analysis_timestep"]
        cuts = sorted(set([0] + list(range(1, n_r + 1, step)))) + ["think"]
        rows = []
        for k in cuts:
            if k == "think":
                prefix_ids, sentences, _, _, _, num_prompt = _build_prefix(
                    tokenizer=tok, prompt=None, data_path=args.dataset, prompt_index=i, base_answer_type="stored",
                    analysis_timestep=rec["analysis_timestep"], analysis_sentence_step=None, sentences_after_prefix=0,
                    min_sentence_length=10, sentence_chunk=1)
            else:
                try:
                    prefix_ids, sentences, _, _, _, num_prompt = _build_prefix(
                        tokenizer=tok, prompt=None, data_path=args.dataset, prompt_index=i, base_answer_type="stored",
                        analysis_timestep=None, analysis_sentence_step=n_p + k, sentences_after_prefix=0,
                        min_sentence_length=10, sentence_chunk=1)
                except ValueError as ex:  # the whole-trace split has fewer sentences (the tail merged); the "think" cut covers the end
                    print(f"example {i}: cut k={k} skipped ({ex})", flush=True)
                    continue
                if prefix_ids.shape[-1] >= think_pos:
                    continue  # this sentence runs into or past </think>; the "think" cut covers it
            assert num_prompt == n_p, (i, num_prompt, n_p)
            S = len(sentences)
            prefix_len = prefix_ids.shape[-1]
            full = torch.cat([prefix_ids.to(dev), probe.make_continuation(dev)], dim=-1)
            gap_f = build_gap_filter(S, 1, device=dev)
            mode_f = build_mode_filter(S, S, "prefix", device=dev)
            causal_f = build_causal_filter(S, device=dev)
            combined = build_combined_filter(gap_f, mode_f, causal_f, None)  # frozen: diagonal, non-causal
            idx = torch.arange(S, device=dev)
            rows_r = idx[:, None] >= n_p
            p2t = rows_r & (idx[None, :] < n_p) & ~combined
            t2t = rows_r & (idx[None, :] >= n_p) & ~combined
            token_to_sent = torch.full((full.shape[-1],), -1, dtype=torch.long, device=dev)
            for si, s in enumerate(sentences):
                token_to_sent[s.start:s.end + 1] = si

            def run(binary):
                return _evaluate_mask(model, layers, num_heads, full, prefix_len, probe, binary, token_to_sent,
                                      combined, dev, True, "sdpa")

            if args.probe_conditions:
                # the probe tokens form sentence S; mode "both" leaves its reads of the S prefix sentences maskable
                combined_p = build_combined_filter(build_gap_filter(S + 1, 1, device=dev), build_mode_filter(S, S + 1, "both", device=dev),
                                                   build_causal_filter(S + 1, device=dev), None)
                token_to_sent_p = token_to_sent.clone()
                token_to_sent_p[prefix_len:] = S
                probe_prompt = torch.zeros(S + 1, S + 1, dtype=torch.bool, device=dev)
                probe_prompt[S, :n_p] = True

                def run_p(binary):
                    return _evaluate_mask(model, layers, num_heads, full, prefix_len, probe, binary, token_to_sent_p,
                                          combined_p, dev, True, "sdpa")

            last_sent = sentences[-1]
            last_text = tok.decode(full[0, last_sent.start:last_sent.end + 1].tolist()) if S > n_p else ""

            with torch.no_grad():
                logits = model(full).logits
            clean_p = answer_probs_from_logits(logits, probe, prefix_len).cpu()
            del logits
            ones = torch.ones(S, S, device=dev)
            row = {"k": (n_r if k == "think" else k), "at_think": k == "think", "n_tokens": int(prefix_len), "n_p2t": int(p2t.sum().item()), "n_t2t": int(t2t.sum().item()),
                   "n_reasoning_sentences_in_prefix": int(S - n_p), "last_sentence_text": last_text,
                   "clean": {"answer_probs": clean_p.tolist(), "p_target": float(clean_p[ti])}}
            if k == "think":
                row["reasoning_sentence_texts"] = [tok.decode(full[0, s_.start:s_.end + 1].tolist()) for s_ in sentences[n_p:]]
            for name, pools in [("no_prompt", [p2t]), ("no_trace", [t2t]), ("no_both", [p2t, t2t])]:
                if k == 0 and name != "no_prompt":
                    continue  # no reasoning sentences: only the (empty) prompt pool exists
                m = ones.clone()
                for pool in pools:
                    m[pool] = 0.0
                p = run(m)
                row[name] = {"answer_probs": p.tolist(), "p_target": float(p[ti]), "kl": _kl(clean_p, p)}
            if args.probe_conditions and k != 0:
                ones_p = torch.ones(S + 1, S + 1, device=dev)
                p2t_p = torch.zeros_like(probe_prompt); p2t_p[:S, :S] = p2t
                t2t_p = torch.zeros_like(probe_prompt); t2t_p[:S, :S] = t2t
                for name, pools in [("probe_no_prompt", [probe_prompt]), ("no_prompt_probe", [p2t_p, probe_prompt]),
                                    ("no_both_probe", [p2t_p, t2t_p, probe_prompt])]:
                    m = ones_p.clone()
                    for pool in pools:
                        m[pool] = 0.0
                    p = run_p(m)
                    row[name] = {"answer_probs": p.tolist(), "p_target": float(p[ti]), "kl": _kl(clean_p, p)}
            rows.append(row)
            del full
            clear_cuda()
        out = {"example_id": i, "tag": rec["tag"], "case": rec["case"], "family": rec["family"], "subset": rec.get("subset") or "",
               "is_positive": rec["is_positive"], "target_letter": letters[ti], "answer_letters": letters,
               "n_prompt_sentences": n_p, "n_reasoning_sentences": n_r, "cuts": rows}
        json.dump(out, open(f"{args.out_dir}/ex{i:03d}.json", "w"), indent=1)
        traj = " ".join(f"{r['clean']['p_target']:.2f}" for r in rows if not r["at_think"])
        last = rows[-1]
        extra = f" probe_no_prompt {last['probe_no_prompt']['p_target']:.3f} no_prompt_probe {last['no_prompt_probe']['p_target']:.3f}" if "probe_no_prompt" in last else ""
        print(f"example {i} ({n_r} sentences, {len(cuts)} cuts, {time.time() - t0:.0f}s): P(target) clean along the trace: {traj} | at </think>: "
              f"no_prompt {last['no_prompt']['p_target']:.3f} no_trace {last['no_trace']['p_target']:.3f} no_both {last['no_both']['p_target']:.3f}{extra}", flush=True)
    remove_handles(handles)


if __name__ == "__main__":
    main()
