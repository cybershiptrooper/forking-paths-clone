"""Thought-anchors-style suppression (prefix-KL) without vLLM branch gen.

Uses the same ``compute_suppression_scores`` function as
:mod:`expts.thought_anchor_analysis` — for each prefix sentence i, zero
all attention to i and measure per-prefix-sentence KL.  Saves a
pair-granularity NodeMask compatible with the comparison script.

Pipeline:
1. Build prefix (data_path + prompt_index, using stored base path) —
   reuses :func:`expts.direct_answer_circuit_discovery.learn._build_prefix`.
2. Load eager-attention model.
3. Run ``compute_suppression_scores`` (from thought_anchor_analysis.py).
4. Save NodeMask with ``algorithm="attention_suppression"`` (the original
   thought-anchors name) so existing dashboards / evaluators recognise it.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda
from utils.masks import NodeMask
from utils.circuit_eval import build_token_to_sent_map
from utils.expt_config import load_config

from expts.thought_anchor_analysis import compute_suppression_scores
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


def main(
    *,
    model_name: str,
    data_path: Optional[str] = None,
    prompt_index: Optional[int] = None,
    prompt: Optional[str] = None,
    base_answer_type: str = "stored",
    analysis_timestep: Optional[int] = None,
    analysis_sentence_step: Optional[int] = None,
    sentences_after_prefix: int = 0,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    mask_mode: str = "prefix",
    freeze_prompt_sentences: bool = False,
    min_sentence_length: int = 10,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/direct_answer_circuit_discovery",
    file_name: Optional[str] = None,
    backend: str = "sdpa",
    **_ignored,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Step 1: Build prefix")
    print("=" * 80)
    tokenizer_for_split = AutoTokenizer.from_pretrained(model_name)
    # sentences_to_be_masked covers the first analysis_sentence_step
    # sentences only; any extra k context sentences are in prefix_ids
    # (the model sees them) but excluded from masking.
    prefix_ids, sentences_to_be_masked, prompt, correct_answer, _, num_prompt_sentences = _build_prefix(
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
    prefix_len = prefix_ids.shape[-1]
    print(f"  Prefix length: {prefix_len} tokens")
    print(f"  Sentences (masked): {len(sentences)}")

    print("=" * 80)
    print(f"Step 2: Loading model")
    print("=" * 80)
    model, tokenizer = load_model_eager(model_name, device=device)
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)

    token_to_sent = build_token_to_sent_map(
        sentences, input_ids.shape[-1], target_device,
    )

    print("=" * 80)
    print("Step 3: Compute suppression scores (prefix-KL — thought anchors)")
    print("=" * 80)
    scores = compute_suppression_scores(
        model=model,
        input_ids=input_ids,
        sentences=sentences,
        token_to_sent=token_to_sent,
        sentence_gap=sentence_gap,
        backend=backend,
    )

    sentence_dicts = [
        {
            "start": s.start,
            "end": s.end,
            "text": tokenizer.decode(input_ids[0, s.start : s.end + 1]),
        }
        for s in sentences
    ]
    node_mask = NodeMask(
        model_name=model_name,
        algorithm="attention_suppression",
        layers=list(range(model.config.num_hidden_layers)),
        sentences=sentence_dicts,
        objective_name="kl_divergence",
        metadata={
            "mask_granularity": "pair",
            "sentence_gap": sentence_gap,
            "num_heads": model.config.num_attention_heads,
            "mask_mode": mask_mode,
            "num_prefix_sentences": len(sentences),
            "negate_scores": False,
            "objective": "kl_divergence",
            "seed": seed,
            "mode": "thought_anchors_prefix_kl",
            "prefix_len": prefix_len,
            "backend": backend,
            "freeze_prompt_sentences": freeze_prompt_sentences,
            "num_frozen_prompt_sentences": (
                num_prompt_sentences if freeze_prompt_sentences else 0
            ),
        },
        scores=scores,
    )
    if correct_answer is not None:
        node_mask.metadata["correct_answer"] = correct_answer
    if data_path is not None and prompt_index is not None:
        node_mask.metadata["data_path"] = data_path
        node_mask.metadata["prompt_index"] = prompt_index
        node_mask.metadata["base_answer_type"] = base_answer_type
    if analysis_sentence_step is not None:
        node_mask.metadata["analysis_sentence_step"] = analysis_sentence_step
    if sentences_after_prefix:
        node_mask.metadata["sentences_after_prefix"] = sentences_after_prefix

    if file_name is not None:
        base = file_name.removesuffix(".json")
        out_file = os.path.join(output_dir, f"{base}_thought_anchors.json")
    else:
        out_file = os.path.join(output_dir, "thought_anchors.json")
    node_mask.to_json(out_file)
    print(f"Saved NodeMask to {out_file}")

    del model
    clear_cuda()
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)
    cfg.pop("config", None)
    main(**cfg)
