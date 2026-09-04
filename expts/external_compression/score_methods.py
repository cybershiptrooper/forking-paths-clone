"""Per-instance sentence scoring: Suppress, TA, and column SNP (matched).

Runs on one instance = (question, prefix length N).  The input is the
forced sequence ``prompt + ' '.join(first N sentences) + anchor + tail``
tokenized as one string; sentence spans come from
``common.build_masking_input`` (their sentence splits, token spans via
offset mapping).  The sentence list is [prompt block] + the N-5
compress-region sentences; the protected 5-sentence tail and the probe are
unmapped (always readable, query rows never masked) — same layout as the
base experiment.

Suppress / TA are adapted copies of the repo kernels
(`expts/direct_answer_circuit_discovery/suppress.py`,
`expts/thought_anchor_analysis.py`) with two changes for 32B memory:
masks are head-collapsed to shape (1, S, S) (columns are head-uniform, and
the token-level expansion at 64 heads x 9k^2 would need ~10 GB), and
suppress reads only the last-position logits.

Column SNP: one training per instance (per review — every M reuses the
ranking), target sparsity = keep 15 of the rankable sentences, canonical
hyperparameters from the colsnp_v3 sweep.

Usage (GPU):
    uv run python -m expts.external_compression.score_methods \
        --instance gpqa_gpqa_diamond_0001_pl60 --methods suppress,ta,colsnp
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from typing import List

import torch
import torch.nn.functional as F

from expts.external_compression.common import (
    DATA_DIR,
    K_KEEP,
    MODEL_NAME,
    RESULTS_DIR,
    build_masking_input,
    load_rollout,
)

SCORES_DIR = os.path.join(RESULTS_DIR, "scores")
SNP_TARGET_KEEP = 15
# Overridable only for smoke tests; production always uses 1000.
SNP_STEPS = int(os.environ.get("EXTCOMP_SNP_STEPS", "1000"))


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def load_instance(instance_id: str) -> dict:
    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = json.load(f)
    by_id = {r["instance_id"]: r for r in instances}
    return by_id[instance_id]


def build_inputs(instance: dict, tokenizer):
    with open(os.path.join(DATA_DIR, "prompts_rendered.json")) as f:
        rendered = json.load(f)
    qid = instance["question_id"]
    N = instance["prefix_length"]
    roll = load_rollout(qid)
    sents = roll["sentences"][:N]
    r = rendered[qid]
    ids, prefix_len, spans = build_masking_input(
        tokenizer, r["prompt_str"], sents, num_mapped=N - K_KEEP,
    )
    return {
        "ids": ids,
        "prefix_len": prefix_len,
        "spans": spans,               # [prompt] + (N-5) compress sentences
        "answer_ids": r["letter_token_ids"],
        "letters": r["letters"],
        "sentences_text": sents,
        "next_sentence_text": roll["sentences"][N] if N < len(roll["sentences"]) else None,
    }


def last_pos_letter_logprobs(model, input_t, answer_ids):
    with torch.no_grad():
        try:
            logits = model(input_t, logits_to_keep=1).logits
        except TypeError:
            logits = model(input_t, num_logits_to_keep=1).logits
    row = logits[0, -1].float()
    ans = row[torch.tensor(answer_ids, device=row.device)]
    return torch.log_softmax(ans, dim=-1)


# ---------------------------------------------------------------------------
# Suppress (leave-one-out on the answer probe) — head-collapsed masks
# ---------------------------------------------------------------------------

def suppress_scores(model, inputs) -> List[float]:
    from utils.circuit_eval import (
        build_token_to_sent_map,
        install_sdpa_mask_hooks,
        install_clean_sdpa_forward,
        remove_handles,
    )
    from utils.utils import get_attention_module

    ids, spans = inputs["ids"], inputs["spans"]
    S = len(spans)
    device = next(model.parameters()).device
    input_t = torch.tensor([ids], device=device)
    all_layers = list(range(model.config.num_hidden_layers))
    gap_filter = torch.zeros(S, S, dtype=torch.bool, device=device)
    token_to_sent = build_token_to_sent_map(spans, len(ids), device)

    sdpa_clean = install_clean_sdpa_forward(model)
    model.eval()
    clean_lp = last_pos_letter_logprobs(model, input_t, inputs["answer_ids"])
    p_clean = clean_lp.exp()

    ones = torch.ones(1, S, S, device=device)
    handles = install_sdpa_mask_hooks(
        model, all_layers, {l: ones for l in all_layers},
        token_to_sent, gap_filter, renormalize=True,
    )

    scores = [0.0] * S
    t0 = time.time()
    for j in range(1, S):  # skip index 0 (frozen prompt block)
        mask = torch.ones(1, S, S, device=device)
        mask[:, :, j] = 0.0
        for layer_idx in all_layers:
            get_attention_module(model, layer_idx)._circuit_mask = mask
        lp = last_pos_letter_logprobs(model, input_t, inputs["answer_ids"])
        scores[j] = float((p_clean * (clean_lp - lp)).sum().item())
        if j % 25 == 0 or j == S - 1:
            print(f"  suppress [{j}/{S - 1}] kl={scores[j]:.4f} "
                  f"({time.time() - t0:.0f}s)")

    remove_handles(handles)
    remove_handles(sdpa_clean)
    return scores


# ---------------------------------------------------------------------------
# TA (leave-one-out, per-token KL at downstream mapped sentences)
# ---------------------------------------------------------------------------

def ta_scores(model, inputs, sentence_gap: int = 1) -> List[float]:
    from utils.circuit_eval import (
        build_token_to_sent_map,
        install_sdpa_mask_hooks,
        install_clean_sdpa_forward,
        remove_handles,
    )
    from utils.utils import get_attention_module

    spans = inputs["spans"]
    prefix_ids = inputs["ids"][: inputs["prefix_len"]]
    S = len(spans)
    device = next(model.parameters()).device
    input_t = torch.tensor([prefix_ids], device=device)
    all_layers = list(range(model.config.num_hidden_layers))
    gap_filter = torch.zeros(S, S, dtype=torch.bool, device=device)
    token_to_sent = build_token_to_sent_map(spans, len(prefix_ids), device)

    sdpa_clean = install_clean_sdpa_forward(model)
    model.eval()
    with torch.no_grad():
        clean_logits = model(input_t).logits[0].to(torch.bfloat16)  # (T, V) on GPU

    def kl_at_positions(masked_logits, positions):
        """Mean over *positions* of full-vocab KL(clean || masked)."""
        out = []
        chunk = 512
        pos_t = torch.tensor(positions, device=device)
        for c0 in range(0, len(positions), chunk):
            idx = pos_t[c0 : c0 + chunk]
            lc = torch.log_softmax(clean_logits[idx].float(), dim=-1)
            lm = torch.log_softmax(masked_logits[idx].float(), dim=-1)
            out.append((lc.exp() * (lc - lm)).sum(-1))
        return torch.cat(out)

    ones = torch.ones(1, S, S, device=device)
    handles = install_sdpa_mask_hooks(
        model, all_layers, {l: ones for l in all_layers},
        token_to_sent, gap_filter, renormalize=True,
    )

    scores = [0.0] * S
    t0 = time.time()
    for j in range(1, S):
        mask = torch.ones(1, S, S, device=device)
        mask[:, :, j] = 0.0
        for layer_idx in all_layers:
            get_attention_module(model, layer_idx)._circuit_mask = mask
        with torch.no_grad():
            masked_logits = model(input_t).logits[0].to(torch.bfloat16)
        # Downstream mapped sentences i >= j + sentence_gap
        per_sent = []
        for i in range(j + sentence_gap, S):
            positions = list(range(spans[i].start, spans[i].end + 1))
            kls = kl_at_positions(masked_logits, positions)
            per_sent.append(float(kls.mean().item()))
        scores[j] = float(sum(per_sent) / len(per_sent)) if per_sent else 0.0
        del masked_logits
        if j % 25 == 0 or j == S - 1:
            print(f"  ta [{j}/{S - 1}] score={scores[j]:.5f} "
                  f"({time.time() - t0:.0f}s)")

    remove_handles(handles)
    remove_handles(sdpa_clean)
    del clean_logits
    torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
# Column SNP (matched), one training per instance
# ---------------------------------------------------------------------------

def colsnp_scores(
    model, tokenizer, inputs, instance_id: str, uniform_l0: bool = False,
    hparams: dict = None, out_root: str = None, method_name: str = None,
) -> List[float]:
    """Train column SNP and return per-sentence scores.

    ``hparams`` may override any of: learning_rate, l0_lambda,
    log_alpha_init (float or "random"), num_hc_samples_per_step,
    l0_warmup_frac, l0_ramp_frac, num_training_steps.
    ``out_root`` overrides the output directory (default: SCORES_DIR/<inst>).
    """
    from utils.circuit_discovery.factory import create_circuit_discovery
    from utils.objectives import answer_probe_kl_loss

    if method_name is None:
        method_name = "colsnp_uniform_l0" if uniform_l0 else "colsnp"
    hparams = hparams or {}
    if out_root is None:
        out_root = os.path.join(SCORES_DIR, instance_id)

    spans = inputs["spans"]
    prefix_len = inputs["prefix_len"]
    ids = inputs["ids"]
    device = next(model.parameters()).device
    num_rankable = len(spans) - 1
    target_sparsity = max(0.0, 1.0 - SNP_TARGET_KEEP / num_rankable)

    input_ids = torch.tensor([ids[:prefix_len]], device=device)
    # Continuation = probe tokens + placeholder (first letter); logits at the
    # last probe token predict the letter.
    probe_ids = ids[prefix_len:]
    placeholder = inputs["answer_ids"][0]
    continuation = torch.tensor([probe_ids + [placeholder]], device=device)
    full_len = prefix_len + continuation.shape[-1]
    position_mask = torch.zeros(1, full_len, device=device)
    position_mask[0, len(ids) - 1] = 1.0  # logits here predict the letter

    objective_fn = partial(
        answer_probe_kl_loss,
        answer_token_ids=torch.tensor(inputs["answer_ids"]),
    )
    objective_fn.__name__ = "answer_probe_kl_loss"

    log_dir = os.path.join(out_root, f"{method_name}_logs")
    discovery_kwargs = dict(
        uniform_column_l0=uniform_l0,
        model=model,
        tokenizer=tokenizer,
        layers=list(range(model.config.num_hidden_layers)),
        objective_fn=objective_fn,
        sentence_gap=1,
        ablate_non_target_layers=False,
        renormalize_masked_attention=True,
        negate_scores=True,
        pair_aggregation="mean",
        mask_granularity="column",
        training_gap_mode="matched",
        sparsity_loss_mode="target_size_relu",
        target_sparsity=target_sparsity,
        optimizer="hybrid",
        save_log_alpha=True,
        l0_lambda=100.0,
        learning_rate=0.1,
        log_alpha_init=2.0,
        num_training_steps=SNP_STEPS,
        log_every=20,
        num_hc_samples_per_step=8,
        log_dir=log_dir,
    )
    discovery_kwargs.update(hparams)
    discoverer = create_circuit_discovery(
        "column_subnetwork_probing", **discovery_kwargs,
    )
    node_mask = discoverer.discover(
        input_ids=input_ids,
        sentences=inputs["spans"],
        continuations=[continuation],
        mask_mode="prefix",
        num_prefix_sentences=len(spans),
        branch_rewards=None,
        position_mask_overrides=[position_mask],
        num_frozen_prompt_sentences=1,
    )
    node_mask.metadata.update({
        "instance_id": instance_id,
        "target_keep": SNP_TARGET_KEEP,
        "target_sparsity": target_sparsity,
        "probe": "their_anchor_think_close",
        "uniform_column_l0": uniform_l0,
        "hparam_overrides": {k: str(v) for k, v in hparams.items()},
    })
    os.makedirs(out_root, exist_ok=True)
    node_mask.to_json(os.path.join(out_root, f"{method_name}_mask.json"))
    scores = node_mask.scores
    assert isinstance(scores, list) and len(scores) == len(spans), (
        f"unexpected colsnp scores shape: {type(scores)}"
    )
    return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--methods", default="suppress,ta")
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()
    methods = args.methods.split(",")

    from utils.utils import set_seed
    set_seed(42)

    instance = load_instance(args.instance)
    out_dir = os.path.join(SCORES_DIR, args.instance)
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = build_inputs(instance, tokenizer)
    S = len(inputs["spans"])
    print(f"instance={args.instance} rankable={S - 1} "
          f"prefix_tokens={inputs['prefix_len']} total_tokens={len(inputs['ids'])}")

    need_training = "colsnp" in methods
    from expts.direct_answer_circuit_discovery.learn import load_model_eager
    model, _ = load_model_eager(
        args.model_name, device="cuda", gradient_checkpointing=need_training,
    )

    for method in methods:
        out_path = os.path.join(out_dir, f"{method}.json")
        if os.path.exists(out_path):
            print(f"skip {method}: {out_path} exists")
            continue
        t0 = time.time()
        if method == "suppress":
            scores = suppress_scores(model, inputs)
        elif method == "ta":
            scores = ta_scores(model, inputs)
        elif method == "colsnp":
            scores = colsnp_scores(model, tokenizer, inputs, args.instance)
        else:
            raise ValueError(method)
        rec = {
            "instance_id": args.instance,
            "method": method,
            "scores": scores,          # index 0 = prompt block (not rankable)
            "num_rankable": S - 1,
            "seconds": time.time() - t0,
            "model_name": args.model_name,
        }
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote {out_path} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
