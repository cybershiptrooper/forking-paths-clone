"""On-policy score-function (REINFORCE) training of a termination mask.

The reference point for the gradient-check comparison: instead of any
teacher-forced surrogate, each step samples binary edge masks from the
current gate probabilities, generates continuations from the analysis
point *with the mask installed*, and rewards masks whose generations
terminate within the horizon AND conclude (via the forced answer probe)
with the trace's own answer — the true objective of Experiment 1 in
notes/termination_and_hint_masks.md, optimized without importance
sampling or teacher forcing.

Parameters: one logit per learnable sentence-pair edge; edge keep
probability p = sigmoid(logit).  Gradient of E[reward] via REINFORCE
with a leave-one-out baseline over the K mask samples per step:

    g = (1/K) sum_k (r_k - mean_{j != k} r_j) * grad log P(z_k)

The sparsity penalty is applied analytically (no score function needed):
lambda_sp * ReLU(mean(p) - (1 - target_sparsity)), matching the
trainer's target-size penalty on gate openness.

This is expensive (each step generates K * n_gen continuations of up to
--horizon tokens) — intended as a 1-prompt pilot, not a sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from utils.masks import (
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda

from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.cot_termination_circuit_discovery.eval_utils import (
    binary_to_per_layer_masks,
)
from expts.cot_termination_circuit_discovery.eval_termination_rollouts import (
    _install, _clear, _generate_and_grade, THINK_END_ID, PROBE_SUFFIX_TEXT,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_path", required=True,
                    help="Bank JSON — used only for prompt/step/horizon "
                    "metadata and the trace answer, not for candidates.")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--num_training_steps", type=int, default=100)
    ap.add_argument("--num_mask_samples", type=int, default=6,
                    help="K: independent mask samples per step.")
    ap.add_argument("--n_gen_per_mask", type=int, default=2,
                    help="Generations per mask sample; reward is their mean.")
    ap.add_argument("--horizon", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--init_logit", type=float, default=2.0,
                    help="Initial edge-keep logit; 2.0 -> p ~ 0.88, near "
                    "the SNP log_alpha_init convention (start close to the "
                    "unmasked model).")
    ap.add_argument("--target_sparsity", type=float, default=0.2)
    ap.add_argument("--sparsity_lambda", type=float, default=5.0)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--sentence_gap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.bank_path) as f:
        bank = json.load(f)
    data_path = bank["data_path"]
    prompt_index = int(bank["prompt_index"])
    step_idx = int(bank["analysis_sentence_step"])
    trace_answer = (bank["trace_answer"] or "").strip()
    all_letters = bank.get("all_letters") or ["A", "B", "C", "D"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]

    prefix_ids, sentences, _, _, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step_idx,
        sentences_after_prefix=0, min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    model, _ = load_model_eager(args.model_name, device="cuda")
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode,
                                    device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    prompt_filter = (
        build_prompt_filter(num_prompt_sentences, num_sents, device=device)
        if num_prompt_sentences else None
    )
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter)
    valid = ~combined_filter.bool()          # learnable edges
    n_valid = int(valid.sum().item())
    print(f"{num_sents} sentences, {n_valid} learnable edges, "
          f"prefix {prefix_len} tokens")

    max_len = prefix_len + args.horizon + 8
    token_to_sent = torch.full((max_len,), -1, dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        token_to_sent[s.start:s.end + 1] = idx

    logits_param = torch.full(
        (num_sents, num_sents), args.init_logit,
        dtype=torch.float32, device=device, requires_grad=True,
    )
    optim = torch.optim.Adam([logits_param], lr=args.learning_rate)
    gen_kwargs = dict(suffix_ids=suffix_ids, letter_ids=letter_ids,
                      trace_answer=trace_answer,
                      batch_size=args.n_gen_per_mask)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    prev_grad = None
    K = args.num_mask_samples
    for step in range(args.num_training_steps):
        t0 = time.time()
        p = torch.sigmoid(logits_param)
        z_samples, rewards, term_rates = [], [], []
        for k in range(K):
            with torch.no_grad():
                z = (torch.rand_like(p) < p).float()
                z_full = torch.ones_like(z)
                z_full[valid] = z[valid]
            binary = z_full
            _install(model, layers, binary, num_heads, token_to_sent,
                     combined_filter)
            rolls = _generate_and_grade(
                model, tokenizer, prefix_ids, args.n_gen_per_mask,
                args.horizon, args.temperature, device, **gen_kwargs)
            _clear(model, layers)
            r = sum(
                1.0 for x in rolls
                if x["terminated"] and x.get("probe_label") == trace_answer
            ) / len(rolls)
            term = sum(1.0 for x in rolls if x["terminated"]) / len(rolls)
            z_samples.append(z_full.detach())
            rewards.append(r)
            term_rates.append(term)
            del rolls
            clear_cuda()

        rewards_t = torch.tensor(rewards, device=device)
        # Leave-one-out baseline per sample.
        if K > 1:
            baselines = (rewards_t.sum() - rewards_t) / (K - 1)
        else:
            baselines = torch.zeros_like(rewards_t)
        advantages = rewards_t - baselines

        optim.zero_grad(set_to_none=True)
        # REINFORCE surrogate: -(1/K) sum_k adv_k * log P(z_k); its gradient
        # is the negated score-function estimate of grad E[reward] (Adam
        # minimizes, reward is maximized).
        logp_terms = []
        for z_full, adv in zip(z_samples, advantages):
            zb = z_full[valid]
            pv = torch.sigmoid(logits_param)[valid]
            log_pz = (
                zb * torch.log(pv + 1e-8)
                + (1.0 - zb) * torch.log(1.0 - pv + 1e-8)
            ).sum()
            logp_terms.append(adv.detach() * log_pz)
        surrogate = -torch.stack(logp_terms).mean()
        # Analytic sparsity penalty on expected gate openness.
        keep_frac = torch.sigmoid(logits_param)[valid].mean()
        sparsity_pen = args.sparsity_lambda * torch.relu(
            keep_frac - (1.0 - args.target_sparsity))
        (surrogate + sparsity_pen).backward()

        with torch.no_grad():
            g = logits_param.grad[valid].flatten()
            grad_norm = float(g.norm().item())
            if prev_grad is not None and grad_norm > 0 \
                    and float(prev_grad.norm().item()) > 0:
                cosine = float(
                    (prev_grad @ g).item()
                    / (prev_grad.norm().item() * g.norm().item())
                )
            else:
                cosine = None
            prev_grad = g.detach().clone()
        optim.step()

        rec = {
            "step": step,
            "mean_reward": float(rewards_t.mean().item()),
            "reward_std": float(rewards_t.std().item()) if K > 1 else 0.0,
            "mean_termination_rate": sum(term_rates) / len(term_rates),
            "advantage_abs_mean": float(advantages.abs().mean().item()),
            "task_grad_norm": grad_norm,
            "task_grad_cosine": cosine,
            "keep_frac": float(keep_frac.detach().item()),
            "sparsity_penalty": float(sparsity_pen.detach().item()),
            "seconds_per_step": time.time() - t0,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{step}] reward={rec['mean_reward']:.3f} "
              f"term={rec['mean_termination_rate']:.3f} "
              f"|g|={grad_norm:.3e} cos={cosine} "
              f"keep={rec['keep_frac']:.3f} "
              f"({rec['seconds_per_step']:.0f}s)", flush=True)

    remove_handles(handles)
    out = {
        "bank_path": args.bank_path,
        "algorithm": "reinforce_termination",
        "num_training_steps": args.num_training_steps,
        "num_mask_samples": K,
        "n_gen_per_mask": args.n_gen_per_mask,
        "horizon": args.horizon,
        "learning_rate": args.learning_rate,
        "init_logit": args.init_logit,
        "target_sparsity": args.target_sparsity,
        "sparsity_lambda": args.sparsity_lambda,
        "seed": args.seed,
        "edge_keep_probs": torch.sigmoid(logits_param).detach().cpu().tolist(),
    }
    with open(os.path.join(args.output_dir, "final_gate_probs.json"), "w") as f:
        json.dump(out, f)
    print(f"Saved {args.output_dir}/final_gate_probs.json")


if __name__ == "__main__":
    main()
