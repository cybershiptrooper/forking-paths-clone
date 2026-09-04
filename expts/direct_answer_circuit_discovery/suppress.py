"""Attention-suppression discovery measuring on the answer token.

Like :mod:`expts.thought_anchor_analysis` but instead of measuring KL at
each downstream prefix sentence, we append the deterministic answer
probe and measure KL between the masked and clean answer-token
distributions at one position.

For each prefix sentence *i*, we zero all attention TO sentence *i*
across every layer/head, run forward on ``prefix + suffix +
placeholder``, extract the answer-token logits, and compute KL against
the clean baseline.  Output is a length-S vector of importance scores
(one per prefix sentence), packaged as a pair-granularity ``NodeMask``
with column duplicated across rows for compatibility with the
sentence-pair convention used by ``evaluate_mask.py``.
"""

from __future__ import annotations

import os
from typing import Optional, List

import torch
import torch.nn.functional as F

from utils.utils import set_seed, clear_cuda, get_attention_module
from utils.cot_analysis import (
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)
from utils.masks import NodeMask
from utils.circuit_eval import (
    build_token_to_sent_map,
    install_mask_hooks,
    install_sdpa_mask_hooks,
    install_clean_sdpa_forward,
    remove_handles,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from expts.direct_answer_circuit_discovery.probe import (
    DEFAULT_ANSWER_LETTERS,
    DEFAULT_SUFFIX,
    AnswerProbe,
    answer_logprobs_from_logits,
    answer_probs_from_logits,
    build_answer_probe,
)
from expts.direct_answer_circuit_discovery.learn import _build_prefix


def load_model_eager(model_name: str, device: str = "cuda"):
    print(f"Loading {model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def compute_suppression_scores_on_answer(
    *,
    model,
    full_input: torch.Tensor,
    sentences,
    probe: AnswerProbe,
    prefix_len: int,
    sentence_gap: int,
    renormalize: bool = True,
    backend: str = "sdpa",
) -> List[float]:
    """Per-prefix-sentence suppression score, measured on the answer probe.

    Returns a length-S float list where ``score[i]`` is the KL between
    clean and masked answer distributions when attention to sentence
    *i* is zeroed across all layers / heads.  Higher = sentence *i* is
    more important for the model's answer prediction.
    """
    num_sents = len(sentences)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    all_layers = list(range(num_layers))
    device = next(model.parameters()).device

    # Sentence-level gap filter (we suppress freely, so no gap)
    gap_filter = torch.zeros(
        num_sents, num_sents, dtype=torch.bool, device=device,
    )

    # token_to_sent must cover full_input (prefix + suffix + placeholder).
    # The suffix tokens are not part of any prefix sentence — they get
    # token_to_sent == -1, which the masking machinery treats as a sentinel.
    token_to_sent = build_token_to_sent_map(
        sentences, full_input.shape[-1], device,
    )

    # Backend-aware setup: with SDPA, patch every layer's forward to SDPA
    # before the clean forward so clean and masked logits both use SDPA.
    sdpa_clean_handles = None
    if backend == "sdpa":
        sdpa_clean_handles = install_clean_sdpa_forward(model)

    # Clean answer-token logprobs
    model.eval()
    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_lp = answer_logprobs_from_logits(
        clean_logits, probe, prefix_len, renormalize=renormalize,
    ).detach()
    del clean_logits
    torch.cuda.empty_cache()

    # Install hooks once with all-ones masks; swap mask in-place per iter.
    ones_mask = torch.ones(num_heads, num_sents, num_sents, device=device)
    binary_masks = {layer: ones_mask for layer in all_layers}
    if backend == "sdpa":
        handles = install_sdpa_mask_hooks(
            model, all_layers, binary_masks, token_to_sent, gap_filter,
            renormalize=True,
        )
    else:
        handles = install_mask_hooks(
            model, all_layers, binary_masks, token_to_sent, gap_filter,
            renormalize=True,
        )

    scores: List[float] = [0.0] * num_sents

    for s_suppress in range(num_sents):
        # Zero column s_suppress (attention TO this sentence), respecting
        # the optional gap (don't suppress if it's the very last sentence
        # and would interfere with the immediate next-token prediction).
        mask = torch.ones(num_heads, num_sents, num_sents, device=device)
        mask[:, :, s_suppress] = 0.0
        for layer_idx in all_layers:
            attn_module = get_attention_module(model, layer_idx)
            attn_module._circuit_mask = mask

        with torch.no_grad():
            masked_logits = model(full_input).logits
        masked_lp = answer_logprobs_from_logits(
            masked_logits, probe, prefix_len, renormalize=renormalize,
        )
        # KL(P_clean || P_masked)
        p_clean = clean_lp.exp()
        kl = (p_clean * (clean_lp - masked_lp)).sum().item()
        scores[s_suppress] = kl
        del masked_logits, masked_lp
        if s_suppress % 5 == 0 or s_suppress == num_sents - 1:
            print(f"  [{s_suppress + 1}/{num_sents}] KL={kl:.4f}")
        torch.cuda.empty_cache()

    remove_handles(handles)
    if sdpa_clean_handles is not None:
        remove_handles(sdpa_clean_handles)
    # sentence_gap currently used only for downstream sparsity book-keeping;
    # the suppression itself does not consult it.
    _ = sentence_gap
    return scores


def main(
    *,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    model_to_analyse: Optional[str] = None,
    prompt: Optional[str] = None,
    data_path: Optional[str] = None,
    prompt_index: Optional[int] = None,
    correct_answer: Optional[str] = None,
    base_answer_type: str = "stored",
    analysis_timestep: Optional[int] = None,
    analysis_sentence_step: Optional[int] = None,
    sentences_after_prefix: int = 0,
    probe_suffix: str = DEFAULT_SUFFIX,
    answer_letters: Optional[List[str]] = None,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    mask_mode: str = "prefix",
    renormalize_masked_attention: bool = True,
    freeze_prompt_sentences: bool = False,
    min_sentence_length: int = 10,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/direct_answer_circuit_discovery",
    file_name: Optional[str] = None,
    backend: str = "sdpa",
):
    if model_to_analyse is None:
        model_to_analyse = model_name
    if answer_letters is None:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Step 1: Build prefix and split into sentences")
    print("=" * 80)
    tokenizer_for_split = AutoTokenizer.from_pretrained(model_name)
    prefix_ids, sentences_to_be_masked, prompt, correct_from_record, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer_for_split,
        prompt=prompt,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type=base_answer_type,
        analysis_timestep=analysis_timestep,
        analysis_sentence_step=analysis_sentence_step,
        sentences_after_prefix=sentences_after_prefix,
        min_sentence_length=min_sentence_length,
        sentence_chunk=sentence_chunk,
    )
    sentences = sentences_to_be_masked
    if correct_answer is None:
        correct_answer = correct_from_record
    prefix_len = prefix_ids.shape[-1]
    print(f"  Prefix length: {prefix_len} tokens")
    print(f"  Sentences (masked): {len(sentences)}")

    print("=" * 80)
    print("Step 2: Build answer probe")
    print("=" * 80)
    probe = build_answer_probe(
        tokenizer_for_split, suffix=probe_suffix, answer_letters=answer_letters,
    )
    print(f"  Suffix: {probe_suffix!r} → {probe.suffix_len} tokens")
    print(f"  Answer letters: {probe.answer_letters}")

    print("=" * 80)
    print(f"Step 3: Loading model with eager attention ({model_to_analyse})")
    print("=" * 80)
    model, tokenizer = load_model_eager(model_to_analyse, device=device)
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)
    continuation = probe.make_continuation(target_device)
    full_input = torch.cat([input_ids, continuation], dim=-1)

    print("=" * 80)
    print("Step 4: Clean answer distribution")
    print("=" * 80)
    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
    print(f"  Clean P(answer): {dict(zip(probe.answer_letters, clean_p.tolist()))}")
    del clean_logits
    clear_cuda()

    print("=" * 80)
    print("Step 5: Suppression scores (per prefix sentence)")
    print("=" * 80)
    score_vec = compute_suppression_scores_on_answer(
        model=model,
        full_input=full_input,
        sentences=sentences,
        probe=probe,
        prefix_len=prefix_len,
        sentence_gap=sentence_gap,
        renormalize=True,
        backend=backend,
    )

    # Package as a (S, S) pair-granularity NodeMask: scores[src][tgt] is
    # the importance of suppressing tgt for src's downstream prediction.
    # We have one scalar per prefix sentence (the answer position), so we
    # broadcast the vector across rows.  Eval code reads these as per-pair
    # importances; the broadcast row layout makes that consistent.
    num_sents = len(sentences)
    score_matrix = [[score_vec[j] for j in range(num_sents)] for _ in range(num_sents)]

    sentence_dicts = [
        {
            "start": s.start,
            "end": s.end,
            "text": tokenizer.decode(input_ids[0, s.start : s.end + 1]),
        }
        for s in sentences
    ]

    node_mask = NodeMask(
        model_name=model_to_analyse,
        algorithm="attention_suppression_on_answer",
        layers=list(range(model.config.num_hidden_layers)),
        sentences=sentence_dicts,
        objective_name="answer_probe_kl",
        metadata={
            "mask_granularity": "pair",
            "sentence_gap": sentence_gap,
            "num_heads": model.config.num_attention_heads,
            "mask_mode": mask_mode,
            "num_prefix_sentences": num_sents,
            "negate_scores": False,
            "renormalize_masked_attention": renormalize_masked_attention,
            "objective": "answer_probe_kl",
            "seed": seed,
            "mode": "direct_answer_suppress",
            "probe_suffix": probe_suffix,
            "probe_suffix_token_ids": probe.suffix_ids.tolist(),
            "answer_letters": probe.answer_letters,
            "answer_token_ids": probe.answer_token_ids.tolist(),
            "answer_logit_position": probe.answer_logit_position(prefix_len),
            "prefix_len": prefix_len,
            "answer_probs_clean": clean_p.tolist(),
            "per_sentence_kl": score_vec,
            "backend": backend,
            "freeze_prompt_sentences": freeze_prompt_sentences,
            "num_frozen_prompt_sentences": (
                num_prompt_sentences if freeze_prompt_sentences else 0
            ),
        },
        scores=score_matrix,
    )
    if correct_answer is not None:
        node_mask.metadata["correct_answer"] = correct_answer
    if data_path is not None and prompt_index is not None:
        node_mask.metadata["data_path"] = data_path
        node_mask.metadata["prompt_index"] = prompt_index
        node_mask.metadata["base_answer_type"] = base_answer_type
    if analysis_sentence_step is not None:
        node_mask.metadata["analysis_sentence_step"] = analysis_sentence_step
    if analysis_timestep is not None:
        node_mask.metadata["analysis_timestep"] = analysis_timestep
    if sentences_after_prefix:
        node_mask.metadata["sentences_after_prefix"] = sentences_after_prefix

    if file_name is not None:
        base = file_name.removesuffix(".json")
        output_file = os.path.join(
            output_dir, f"{base}_attention_suppression_on_answer.json",
        )
    else:
        output_file = os.path.join(
            output_dir, "attention_suppression_on_answer.json",
        )
    node_mask.to_json(output_file)
    print(f"Saved NodeMask to {output_file}")

    del model
    clear_cuda()
    print("\nDone!")
    return output_file
