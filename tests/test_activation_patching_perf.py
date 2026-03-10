"""Benchmark sequential vs. batched continuation evaluation in NodewiseActivationPatching.

Run with:
    python -m tests.test_activation_patching_perf

What this tests
---------------
For each activation-patching probe the model is called once per continuation
(sequential) or once for *all* continuations together (batched).  Both should
produce the same KL values; the batched path should be ~N_CONTS× faster.

The test uses a tiny Llama model built from scratch so it runs on CPU in
seconds — no checkpoint download needed.
"""

import time
import types
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from utils.utils import Sentence
from utils.masks import build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask
from utils.circuit_discovery.nodewise_activation_patching import NodewiseActivationPatching
from utils.objectives import get_objective

# ---------------------------------------------------------------------------
# Tiny model config
# ---------------------------------------------------------------------------
VOCAB = 512
SEQ_LEN = 40        # prefix length
CONT_LEN = 12       # continuation length (each branch)
N_CONTS = 8         # number of branches
N_HEADS = 4
N_LAYERS = 4
HIDDEN = 64
INTERMED = 128
DEVICE = "cpu"


def make_tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMED,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def make_inputs(model):
    """Return input_ids, continuations, sentences, clean_logits_list."""
    torch.manual_seed(0)
    prefix_len = SEQ_LEN

    input_ids = torch.randint(1, VOCAB, (1, prefix_len))

    # Variable-length continuations so padding is exercised
    continuations = [
        torch.randint(1, VOCAB, (1, CONT_LEN - (i % 3)))
        for i in range(N_CONTS)
    ]

    # Three sentences spanning the prefix
    s0 = Sentence(start=0, end=12)
    s1 = Sentence(start=13, end=25)
    s2 = Sentence(start=26, end=prefix_len - 1)
    sentences = [s0, s1, s2]

    # Pre-compute clean logits (no patching)
    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float32):
        clean_logits_list = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            out = model(full_input)
            clean_logits_list.append(out.logits.float().detach())

    return input_ids, continuations, sentences, clean_logits_list


# ---------------------------------------------------------------------------
# Sequential reference (old approach, extracted from original _compute_mean_kl)
# ---------------------------------------------------------------------------
def sequential_mean_kl(
    model,
    objective_fn,
    input_ids,
    continuations,
    clean_logits_list,
    prefix_len,
    device,
):
    def _build_pos_mask(full_len, prefix_len, device):
        mask = torch.zeros(1, full_len, device=device)
        mask[0, prefix_len - 1 : full_len - 1] = 1.0
        return mask

    obj_sum = 0.0
    for cont_idx, cont in enumerate(continuations):
        full_input = torch.cat([input_ids, cont], dim=-1)
        full_len = full_input.shape[-1]
        position_mask = _build_pos_mask(full_len, prefix_len, device)
        clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)
        with torch.amp.autocast("cpu", dtype=torch.float32):
            logits = model(full_input).logits
        obj = objective_fn(clean_logits, logits.float(), position_mask, token_ids=full_input)
        obj_sum += obj.item()
    return obj_sum / len(continuations)


# ---------------------------------------------------------------------------
# Batched (new approach — mirrors NodewiseActivationPatching._compute_mean_kl)
# ---------------------------------------------------------------------------
def batched_mean_kl(
    model,
    objective_fn,
    input_ids,
    continuations,
    clean_logits_list,
    prefix_len,
    device,
):
    def _build_pos_mask(full_len, prefix_len, device):
        mask = torch.zeros(1, full_len, device=device)
        mask[0, prefix_len - 1 : full_len - 1] = 1.0
        return mask

    cont_lens = [c.shape[-1] for c in continuations]
    max_cont_len = max(cont_lens)
    batch_size = len(continuations)
    full_len = prefix_len + max_cont_len

    batch_input = torch.zeros(batch_size, full_len, dtype=input_ids.dtype, device=device)
    attn_mask = torch.zeros(batch_size, full_len, device=device)
    for i, (cont, clen) in enumerate(zip(continuations, cont_lens)):
        actual = prefix_len + clen
        batch_input[i, :prefix_len] = input_ids[0]
        batch_input[i, prefix_len:actual] = cont[0]
        attn_mask[i, :actual] = 1.0

    with torch.amp.autocast("cpu", dtype=torch.float32):
        logits_batch = model(batch_input, attention_mask=attn_mask).logits

    obj_sum = 0.0
    for cont_idx, (clen, clean_logits) in enumerate(zip(cont_lens, clean_logits_list)):
        actual = prefix_len + clen
        logits_i = logits_batch[cont_idx : cont_idx + 1, :actual]
        clean_logits_i = clean_logits[:, :actual].to(device)
        position_mask = _build_pos_mask(actual, prefix_len, device)
        obj = objective_fn(
            clean_logits_i, logits_i.float(), position_mask,
            token_ids=batch_input[cont_idx : cont_idx + 1, :actual],
        )
        obj_sum += obj.item()
    return obj_sum / batch_size


# ---------------------------------------------------------------------------
# End-to-end NodewiseActivationPatching discover() benchmark
# ---------------------------------------------------------------------------
def bench_discoverer(model, input_ids, continuations, sentences, objective_fn, n_probes_limit=20):
    """Run discover() on a small problem and time it."""
    discoverer = NodewiseActivationPatching(
        model=model,
        tokenizer=None,
        layers=[0, 1],
        objective_fn=objective_fn,
        sentence_gap=1,
        ablate_non_target_layers=False,
        renormalize_masked_attention=True,
        mask_granularity="pair",
    )
    t0 = time.perf_counter()
    mask = discoverer.discover(
        input_ids=input_ids,
        sentences=sentences,
        continuations=continuations,
        mask_mode="prefix",
        num_prefix_sentences=len(sentences),
    )
    elapsed = time.perf_counter() - t0
    return elapsed, mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building tiny Llama model...")
    model = make_tiny_llama()
    input_ids, continuations, sentences, clean_logits_list = make_inputs(model)
    prefix_len = input_ids.shape[-1]
    objective_fn = get_objective("kl_divergence")
    N_BENCH = 10  # probes to benchmark for the per-probe comparison

    print(f"\nConfig: prefix={prefix_len} tokens, {N_CONTS} continuations, "
          f"{len(sentences)} sentences, device={DEVICE}")

    # ------------------------------------------------------------------
    # 1. Correctness check: sequential vs batched give the same KL
    # ------------------------------------------------------------------
    print("\n--- Correctness check ---")
    with torch.no_grad():
        kl_seq = sequential_mean_kl(
            model, objective_fn, input_ids, continuations, clean_logits_list, prefix_len, DEVICE
        )
        kl_bat = batched_mean_kl(
            model, objective_fn, input_ids, continuations, clean_logits_list, prefix_len, DEVICE
        )
    print(f"  Sequential KL : {kl_seq:.6f}")
    print(f"  Batched    KL : {kl_bat:.6f}")
    diff = abs(kl_seq - kl_bat)
    print(f"  Absolute diff : {diff:.2e}")
    assert diff < 1e-4, f"KL values differ too much: {diff}"
    print("  PASS — values agree within 1e-4")

    # ------------------------------------------------------------------
    # 2. Per-probe timing: N_BENCH probes, sequential vs batched
    # ------------------------------------------------------------------
    print(f"\n--- Per-probe timing ({N_BENCH} probes) ---")

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            sequential_mean_kl(
                model, objective_fn, input_ids, continuations, clean_logits_list, prefix_len, DEVICE
            )
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            batched_mean_kl(
                model, objective_fn, input_ids, continuations, clean_logits_list, prefix_len, DEVICE
            )
        t_bat = time.perf_counter() - t0

    print(f"  Sequential : {t_seq:.3f}s total  ({t_seq/N_BENCH*1000:.1f} ms/probe)")
    print(f"  Batched    : {t_bat:.3f}s total  ({t_bat/N_BENCH*1000:.1f} ms/probe)")
    speedup = t_seq / t_bat
    print(f"  Speedup    : {speedup:.2f}x")
    # Batching should at minimum not be slower
    assert speedup > 0.5, f"Batched is unexpectedly much slower: {speedup:.2f}x"
    print("  PASS")

    # ------------------------------------------------------------------
    # 3. Full discover() end-to-end timing (uses batched internally)
    # ------------------------------------------------------------------
    print("\n--- Full discover() timing ---")
    with torch.no_grad():
        elapsed, node_mask = bench_discoverer(
            model, input_ids, continuations, sentences, objective_fn
        )
    n_sents = len(sentences)
    n_pairs = n_sents * (n_sents - 1) // 2  # rough active pairs
    print(f"  discover() completed in {elapsed:.3f}s")
    print(f"  Sentences: {n_sents}, ~{n_pairs} active pairs, layers=[0,1]")
    print(f"  Scores shape (pair): {len(node_mask.scores)} x {len(node_mask.scores[0])}")
    print("  PASS")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
