"""Two on-policy evaluations of a learned mask for the answer-distribution (KL) task.

Both start from the same prefix as the rollout evaluation in
``eval_masked_rollouts.py`` (the stored trace cut at the analysis point,
no continuation sentences) and use the same probe suffix.

Stage A (``build_clean_rollouts``, once per prompt): sample two independent
sets of ``K`` continuations from the *unmasked* model (temperature 0.7,
at most ``max_new_tokens`` tokens), append the probe suffix, and record
for every continuation the unmasked answer distribution and the answer
distribution with every eligible edge removed (the ``kl_max`` reference of
the fixed-continuation evaluation). Tokens are stored so that Stage B can
re-run the exact same continuations under a mask.

Stage B (``run_cell``, once per (mask, target sparsity)):

1. **Outcome-distribution KL.** Sample ``K`` continuations with the mask
   installed, read the answer letter each one lands on (argmax of the
   probe distribution), and compare the histogram of those letters with
   the histogram of the clean set A, as KL(p_clean || p_mask) with a
   pseudo-count of ``alpha`` per letter (Figure 2 of
   ``notes/reports_cot_termination/new_fixes_and_variants.md``). The
   clean floor is the same quantity between the two clean sets A and B.
   The KL between the two mean answer distributions and the total
   variation distance of the histograms are recorded alongside.

2. **Answer-logit KL on the clean continuations.** For every continuation
   of clean set A, run one forward pass with the mask installed and
   compare the masked answer distribution with the unmasked one on the
   same text: KL(P_clean(. | y) || P_mask(. | y)). No new text is sampled
   from the masked model, so this is the training objective evaluated on
   the unmasked model's own samples instead of on the stored suffix.
"""

from __future__ import annotations

import json
import os
from argparse import Namespace

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward
from utils.utils import set_seed, clear_cuda
from expts.direct_answer_circuit_discovery.probe import (
    answer_probs_from_logits,
    build_answer_probe,
    DEFAULT_SUFFIX,
    DEFAULT_ANSWER_LETTERS,
)
from expts.direct_answer_circuit_discovery.learn import (
    _build_prefix,
    load_model_eager,
)
from expts.direct_answer_circuit_discovery.eval_log_alpha import (
    _scores_to_log_alpha,
    _build_binary_mask,
    _binary_to_per_layer_masks,
    _all_zero_mask,
    _kl,
)
from expts.direct_answer_circuit_discovery.eval_masked_rollouts import (
    _install_mask_on_model,
    _clear_mask_from_model,
)


# --------------------------------------------------------------------------- helpers
def _get_model(model_name, ctx):
    cached = ctx.setdefault(model_name, {})
    if "model" not in cached:
        cached["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        model, _ = load_model_eager(model_name, device="cuda")
        cached["sdpa_handles"] = install_clean_sdpa_forward(model)
        cached["model"] = model
    return cached["model"], cached["tokenizer"]


def _prefix_and_sentences(tokenizer, data_path, prompt_index, step, k):
    prefix_ids, sentences, _, correct_answer, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=k, min_sentence_length=10, sentence_chunk=1,
    )
    return prefix_ids, sentences, correct_answer, num_prompt_sentences


def _eos_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tokenizer.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    return set(int(i) for i in ids)


def _sample_continuations(model, tokenizer, prefix_ids, n, max_new_tokens,
                          temperature, seed, gen_batch):
    """Sample ``n`` continuations in batches of ``gen_batch``, each batch with
    its own seed. Returns a list of token-id lists, cut after the first
    end-of-sequence token so padded tails are dropped."""
    device = next(model.parameters()).device
    eos = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prefix_len = prefix_ids.shape[-1]
    out = []
    for b, lo in enumerate(range(0, n, gen_batch)):
        bs = min(gen_batch, n - lo)
        set_seed(seed + b)
        inp = prefix_ids.to(device).expand(bs, -1)
        gen = model.generate(
            input_ids=inp, attention_mask=torch.ones_like(inp),
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, use_cache=True, pad_token_id=pad_id,
        )
        for i in range(bs):
            toks = gen[i, prefix_len:].tolist()
            cut = len(toks)
            for j, t in enumerate(toks):
                if t in eos:
                    cut = j + 1
                    break
            out.append(toks[:cut])
        del gen
        clear_cuda()
    return out


def _probe_probs(model, probe, prefix_ids, cont_tokens, device):
    """Answer distribution after ``prefix + continuation + probe suffix``."""
    cont = torch.tensor(cont_tokens, dtype=torch.long, device=device).unsqueeze(0)
    full = torch.cat([prefix_ids.to(device), cont, probe.make_continuation(device)], dim=-1)
    with torch.no_grad():
        logits = model(full).logits
    p = answer_probs_from_logits(logits, probe, prefix_ids.shape[-1] + cont.shape[-1]).cpu()
    del logits
    return p


def _filters(nm_meta, num_sents, sentence_gap, mask_mode, force_freeze_prompt,
             num_prompt_sentences, device):
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen = int((nm_meta or {}).get("num_frozen_prompt_sentences", 0) or 0)
    if force_freeze_prompt and not num_frozen:
        num_frozen = num_prompt_sentences
    prompt_filter = build_prompt_filter(num_frozen, num_sents, device=device) if num_frozen else None
    combined = build_combined_filter(gap_filter, mode_filter, causal_filter, prompt_filter)
    return combined, num_frozen


def _token_to_sent(sentences, length, device):
    t2s = torch.full((length,), -1, dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        t2s[s.start: s.end + 1] = idx
    return t2s


def outcome_hist(probs_list, n_letters):
    """Fraction of continuations whose argmax answer is each letter."""
    h = np.zeros(n_letters)
    for p in probs_list:
        h[int(np.argmax(p))] += 1
    return (h / max(len(probs_list), 1)).tolist()


def kl_outcome(h_clean, h_mask, n_clean, n_mask, alpha=0.5):
    """KL(clean || mask) between two outcome histograms (fractions over the
    same fixed set of letters), each smoothed with ``alpha`` pseudo-counts."""
    K = len(h_clean)
    p = (np.asarray(h_clean) * n_clean + alpha) / (n_clean + K * alpha)
    q = (np.asarray(h_mask) * n_mask + alpha) / (n_mask + K * alpha)
    return float(np.sum(p * np.log(p / q)))


def tv(h1, h2):
    return float(0.5 * np.abs(np.asarray(h1) - np.asarray(h2)).sum())


def kl_of_means(probs_a, probs_b):
    """KL between the mean answer distributions of two rollout sets."""
    a = torch.tensor(np.mean(np.asarray(probs_a), axis=0))
    b = torch.tensor(np.mean(np.asarray(probs_b), axis=0))
    return _kl(a, b)


# --------------------------------------------------------------------------- stage A
def build_clean_rollouts(args: Namespace, ctx: dict):
    """Sample two clean sets of K continuations for one prompt and record,
    per continuation, the unmasked answer distribution and the all-edges-
    removed answer distribution. Writes ``args.output``."""
    model, tokenizer = _get_model(args.model_name, ctx)
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads

    prefix_ids, sentences, correct_answer, n_prompt = _prefix_and_sentences(
        tokenizer, args.data_path, args.prompt_index, args.analysis_sentence_step, 0)
    letters = [s for s in args.answer_letters.split(",")] if args.answer_letters else list(DEFAULT_ANSWER_LETTERS)
    probe = build_answer_probe(tokenizer, suffix=args.probe_suffix or DEFAULT_SUFFIX, answer_letters=letters)
    num_sents = len(sentences)
    # The all-zero mask uses the same eligible-edge pool as every mask of
    # this prompt: gap/mode/causal filters plus the frozen prompt sentences.
    combined, num_frozen = _filters({"num_frozen_prompt_sentences": n_prompt}, num_sents,
                                    args.sentence_gap, args.mask_mode, True, n_prompt, device)
    length = prefix_ids.shape[-1] + args.max_new_tokens + probe.suffix_len + 5
    t2s = _token_to_sent(sentences, length, device)
    zero_masks = _binary_to_per_layer_masks(_all_zero_mask(num_sents, len(layers), num_heads, device), layers, num_heads)

    sets = {}
    for name, seed in (("A", args.seed), ("B", args.seed + 1000)):
        toks = _sample_continuations(model, tokenizer, prefix_ids, args.n_rollouts,
                                     args.max_new_tokens, args.temperature, seed, args.gen_batch)
        clean_p = [_probe_probs(model, probe, prefix_ids, t, device).tolist() for t in toks]
        _install_mask_on_model(model, layers, zero_masks, t2s, combined, True)
        zero_p = [_probe_probs(model, probe, prefix_ids, t, device).tolist() for t in toks]
        _clear_mask_from_model(model, layers)
        clear_cuda()
        sets[name] = {
            "seed": seed,
            "rollouts": [{"tokens": t, "n_tokens": len(t), "text": tokenizer.decode(t, skip_special_tokens=False),
                          "clean_answer_probs": cp, "all_zero_answer_probs": zp,
                          "kl_max": _kl(torch.tensor(cp), torch.tensor(zp))}
                         for t, cp, zp in zip(toks, clean_p, zero_p)],
        }
    n = len(letters)
    hA = outcome_hist([r["clean_answer_probs"] for r in sets["A"]["rollouts"]], n)
    hB = outcome_hist([r["clean_answer_probs"] for r in sets["B"]["rollouts"]], n)
    out = {
        "model_name": args.model_name, "data_path": args.data_path, "prompt_index": args.prompt_index,
        "analysis_sentence_step": args.analysis_sentence_step, "sentence_gap": args.sentence_gap,
        "mask_mode": args.mask_mode, "num_frozen_prompt_sentences": num_frozen, "num_sentences": num_sents,
        "prefix_len": int(prefix_ids.shape[-1]), "answer_letters": letters, "probe_suffix": probe.suffix,
        "correct_answer": correct_answer, "n_rollouts": args.n_rollouts, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "gen_batch": args.gen_batch, "alpha": args.alpha,
        "outcome_hist_A": hA, "outcome_hist_B": hB,
        "kl_outcome_A_vs_B": kl_outcome(hA, hB, args.n_rollouts, args.n_rollouts, args.alpha),
        "tv_A_vs_B": tv(hA, hB),
        "kl_means_A_vs_B": kl_of_means([r["clean_answer_probs"] for r in sets["A"]["rollouts"]],
                                       [r["clean_answer_probs"] for r in sets["B"]["rollouts"]]),
        "sets": sets,
    }
    _write(args.output, out)
    print(f"  clean sets written: hist A {hA} hist B {hB} floor KL {out['kl_outcome_A_vs_B']:.4f}")


# --------------------------------------------------------------------------- stage B
def run_cell(args: Namespace, ctx: dict):
    """One (mask, target sparsity) cell. Needs the prompt's clean-rollout file
    (``args.clean_path``) from stage A. Writes ``args.output``."""
    model, tokenizer = _get_model(args.model_name, ctx)
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads

    with open(args.clean_path) as f:
        clean = json.load(f)
    nm = NodeMask.from_json(args.mask_path)
    score_readout = nm.score_readout
    granularity = nm.metadata.get("mask_granularity") or nm.granularity or "pair"
    assert granularity == "pair", f"only pair masks are supported here, got {granularity}"

    prefix_ids, sentences, _, n_prompt = _prefix_and_sentences(
        tokenizer, args.data_path, args.prompt_index, args.analysis_sentence_step, 0)
    prefix_k, _, _, _ = _prefix_and_sentences(
        tokenizer, args.data_path, args.prompt_index, args.analysis_sentence_step, args.sentences_after_prefix)
    assert int(prefix_ids.shape[-1]) == clean["prefix_len"], "prefix differs from the clean-rollout file"
    num_sents = len(sentences)
    letters = clean["answer_letters"]
    mask_letters = nm.metadata.get("answer_letters")
    if args.answer_letters:
        assert [s for s in args.answer_letters.split(",")] == letters, (args.answer_letters, letters)
    elif mask_letters:
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
    n_valid = int(valid.sum().item())

    length = prefix_ids.shape[-1] + args.max_new_tokens + probe.suffix_len + 5
    t2s = _token_to_sent(sentences, max(length, prefix_k.shape[-1] + probe.suffix_len + 5), device)

    # -- fixed-continuation check (must reproduce the published eval file)
    fixed_cont = prefix_k[0, prefix_ids.shape[-1]:].tolist()
    clean_fixed = _probe_probs(model, probe, prefix_ids, fixed_cont, device)
    _install_mask_on_model(model, layers, masks, t2s, combined, True)
    masked_fixed = _probe_probs(model, probe, prefix_ids, fixed_cont, device)
    kl_fixed = _kl(clean_fixed, masked_fixed)
    _clear_mask_from_model(model, layers)
    zero_masks = _binary_to_per_layer_masks(_all_zero_mask(num_sents, len(layers), num_heads, device), layers, num_heads)
    _install_mask_on_model(model, layers, zero_masks, t2s, combined, True)
    kl_max_fixed = _kl(clean_fixed, _probe_probs(model, probe, prefix_ids, fixed_cont, device))
    _clear_mask_from_model(model, layers)
    _install_mask_on_model(model, layers, masks, t2s, combined, True)

    # -- 2. answer-logit KL on clean set A (mask on, text fixed)
    A = clean["sets"]["A"]["rollouts"]
    logit_rows = []
    for r in A:
        pm = _probe_probs(model, probe, prefix_ids, r["tokens"], device)
        pc = torch.tensor(r["clean_answer_probs"])
        kl = _kl(pc, pm)
        logit_rows.append({"kl": kl, "kl_max": r["kl_max"],
                           "kl_normalized": (kl / r["kl_max"] if r["kl_max"] > 0 else None),
                           "masked_answer_probs": pm.tolist()})

    # -- 2b. the same on clean set B (the training set of masks trained on a rollout bank)
    logit_rows_B = []
    for r in clean["sets"]["B"]["rollouts"]:
        pm = _probe_probs(model, probe, prefix_ids, r["tokens"], device)
        pc = torch.tensor(r["clean_answer_probs"])
        logit_rows_B.append({"kl": _kl(pc, pm), "kl_max": r["kl_max"]})

    # -- 1. outcome distribution of K masked rollouts vs clean set A
    toks = _sample_continuations(model, tokenizer, prefix_ids, args.n_rollouts, args.max_new_tokens,
                                 args.temperature, args.seed, args.gen_batch)
    masked_p = [_probe_probs(model, probe, prefix_ids, t, device).tolist() for t in toks]
    _clear_mask_from_model(model, layers)
    clear_cuda()

    n = len(letters)
    hM = outcome_hist(masked_p, n)
    hA = clean["outcome_hist_A"]
    nA = clean["n_rollouts"]
    cleanA_p = [r["clean_answer_probs"] for r in A]
    out = {
        "mask_path": args.mask_path, "model_name": args.model_name, "data_path": args.data_path,
        "prompt_index": args.prompt_index, "analysis_sentence_step": args.analysis_sentence_step,
        "target_sparsity": args.target_sparsity, "n_removed": n_removed, "n_valid": n_valid,
        "clean_path": args.clean_path, "answer_letters": letters, "n_rollouts": args.n_rollouts,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "alpha": args.alpha,
        "kl_masked_fixed": kl_fixed, "kl_max_fixed": kl_max_fixed, "clean_fixed_answer_probs": clean_fixed.tolist(),
        "masked_fixed_answer_probs": masked_fixed.tolist(),
        # 1. outcome distribution
        "outcome_hist_masked": hM, "outcome_hist_clean": hA, "outcome_hist_clean_B": clean["outcome_hist_B"],
        "kl_outcome": kl_outcome(hA, hM, nA, args.n_rollouts, args.alpha),
        "kl_outcome_floor": clean["kl_outcome_A_vs_B"],
        "tv_outcome": tv(hA, hM), "tv_outcome_floor": clean["tv_A_vs_B"],
        "kl_means": kl_of_means(cleanA_p, masked_p), "kl_means_floor": clean["kl_means_A_vs_B"],
        "masked_rollouts": [{"n_tokens": len(t), "text": tokenizer.decode(t, skip_special_tokens=False),
                             "answer_probs": p} for t, p in zip(toks, masked_p)],
        # 2. answer-logit KL on the clean continuations
        "clean_rollout_kl": logit_rows,
        "clean_rollout_kl_median": float(np.median([r["kl"] for r in logit_rows])),
        "clean_rollout_kl_mean": float(np.mean([r["kl"] for r in logit_rows])),
        "clean_rollout_kl_B": logit_rows_B,
        "clean_rollout_kl_B_median": float(np.median([r["kl"] for r in logit_rows_B])),
    }
    _write(args.output, out)
    print(f"  tsp={args.target_sparsity} fixed KL {kl_fixed:.5f} | outcome KL {out['kl_outcome']:.4f} "
          f"(floor {out['kl_outcome_floor']:.4f}) hist {hM} | clean-rollout KL median {out['clean_rollout_kl_median']:.5f}")


def _write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
