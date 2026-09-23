"""Answer-logit KL of a mask on clean rollout set B only (stage 2b of
``eval_onpolicy_kl.run_cell``), for masks whose on-policy evaluation file was
written before that stage existed (Thought Anchors and the stored-suffix masks).

Set B is the bank the rollout-bank masks were trained on, so this gives the
comparison arms' score on the same 32 continuations. No text is sampled: one
teacher-forced forward per continuation with the binary top-k mask installed.

Writes ``<output>``: a small JSON with ``clean_rollout_kl_B`` (per rollout) and
``clean_rollout_kl_B_median``, next to the existing ``*_onpolicy.json``.
"""
from __future__ import annotations

import json
import os
from argparse import Namespace

import numpy as np
import torch

from utils.masks import NodeMask
from expts.direct_answer_circuit_discovery.probe import build_answer_probe, DEFAULT_SUFFIX
from expts.direct_answer_circuit_discovery.eval_log_alpha import _scores_to_log_alpha, _build_binary_mask, _binary_to_per_layer_masks, _kl
from expts.direct_answer_circuit_discovery.eval_masked_rollouts import _install_mask_on_model, _clear_mask_from_model
from expts.direct_answer_circuit_discovery.eval_onpolicy_kl import (
    _get_model, _prefix_and_sentences, _probe_probs, _filters, _token_to_sent, _write,
)


def run_setB(args: Namespace, ctx: dict):
    """Set B of the mask's own clean-rollout file (the training bank)."""
    return run_rollout_set(args, ctx, set_name="B", start=0, end=None, label="B")


def run_setA(args: Namespace, ctx: dict):
    """Set A of the clean-rollout file (the published held-out set), teacher-forced only:
    the answer-logit KL half of ``run_cell`` without sampling masked continuations."""
    return run_rollout_set(args, ctx, set_name="A", start=0, end=None, label="A")


def run_setC(args: Namespace, ctx: dict):
    """Set C: rollouts 32 to 63 of set A of the 128-rollout file (``args.clean_path``
    must be that file). Never used in training by any arm and disjoint from the
    published held-out set A, which is the first 32 of the same set."""
    return run_rollout_set(args, ctx, set_name="A", start=32, end=64, label="C")


def run_rollout_set(args: Namespace, ctx: dict, set_name: str, start: int, end, label: str):
    model, tokenizer = _get_model(args.model_name, ctx)
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    with open(args.clean_path) as f:
        clean = json.load(f)
    nm = NodeMask.from_json(args.mask_path)
    score_readout = nm.score_readout
    prefix_ids, sentences, _, n_prompt = _prefix_and_sentences(
        tokenizer, args.data_path, args.prompt_index, args.analysis_sentence_step, 0)
    assert int(prefix_ids.shape[-1]) == clean["prefix_len"], "prefix differs from the clean-rollout file"
    num_sents = len(sentences)
    letters = clean["answer_letters"]
    mask_letters = nm.metadata.get("answer_letters")
    if mask_letters:
        assert list(mask_letters) == letters, (mask_letters, letters)
    probe = build_answer_probe(tokenizer, suffix=nm.metadata.get("probe_suffix", DEFAULT_SUFFIX), answer_letters=letters)
    assert probe.suffix == clean["probe_suffix"]
    combined, num_frozen = _filters(nm.metadata, num_sents, args.sentence_gap, args.mask_mode,
                                    args.force_freeze_prompt, n_prompt, device)
    assert num_frozen == clean["num_frozen_prompt_sentences"], (num_frozen, clean["num_frozen_prompt_sentences"])
    valid = ~combined.bool()
    log_alpha = _scores_to_log_alpha(nm.scores, score_readout).to(device)
    binary = _build_binary_mask(log_alpha, "top_k", args.target_sparsity, valid)
    masks = _binary_to_per_layer_masks(binary, layers, num_heads)
    n_removed = int(((binary == 0) & valid).sum().item())
    length = prefix_ids.shape[-1] + clean["max_new_tokens"] + probe.suffix_len + 5
    t2s = _token_to_sent(sentences, length, device)
    _install_mask_on_model(model, layers, masks, t2s, combined, True)
    rows = []
    for r in clean["sets"][set_name]["rollouts"][start:end]:
        pm = _probe_probs(model, probe, prefix_ids, r["tokens"], device)
        pc = torch.tensor(r["clean_answer_probs"])
        rows.append({"kl": _kl(pc, pm), "kl_max": r["kl_max"], "masked_answer_probs": pm.tolist()})
    _clear_mask_from_model(model, layers)
    out = {"mask_path": args.mask_path, "clean_path": args.clean_path, "target_sparsity": args.target_sparsity,
           "n_removed": n_removed, "n_valid": int(valid.sum().item()), "set": label,
           "source_set": set_name, "rollout_slice": [start, end if end is not None else len(clean["sets"][set_name]["rollouts"])],
           f"clean_rollout_kl_{label}": rows, f"clean_rollout_kl_{label}_median": float(np.median([r["kl"] for r in rows])),
           f"clean_rollout_kl_{label}_mean": float(np.mean([r["kl"] for r in rows]))}
    _write(args.output, out)
    print(f"  tsp={args.target_sparsity} set-{label} KL median {out[f'clean_rollout_kl_{label}_median']:.5f}")
