"""Benchmark sequential vs. batched vs. prefix-cached continuation evaluation.

Run with:
    python -m tests.test_activation_patching_perf

What this tests
---------------
Three modes for evaluating the mean KL per activation-patching probe:

  1. Sequential  — one full forward pass per continuation  (original)
  2. Batched     — all continuations in one full-sequence forward pass
  3. Cached      — prefix KV cache + short continuation-only forward pass

Expected results (GPU, real 8B model, prefix=200, cont=150, B=16):
  Sequential  ~22 s/probe  (baseline, 47 h total for 7626 probes)
  Batched     ~3-4 s/probe (~6× speedup)
  Cached      ~1.5-2 s/probe (~2× additional, ~10-12× total)

On CPU with the tiny model used here the absolute times are smaller but
the relative ordering and ~agreement between methods should still hold.
"""

import time
import types
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from utils.utils import Sentence
from utils.circuit_discovery.nodewise_activation_patching import NodewiseActivationPatching
from utils.objectives import get_objective

# ---------------------------------------------------------------------------
# Tiny model config (no GPU / checkpoint needed)
# ---------------------------------------------------------------------------
VOCAB = 512
SEQ_LEN = 48         # prefix length
CONT_LEN = 16        # max continuation length per branch
N_CONTS = 8          # number of branches
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
    torch.manual_seed(0)
    prefix_len = SEQ_LEN
    input_ids = torch.randint(1, VOCAB, (1, prefix_len))
    # Variable-length continuations so right-padding is exercised
    continuations = [
        torch.randint(1, VOCAB, (1, CONT_LEN - (i % 4)))
        for i in range(N_CONTS)
    ]
    s0 = Sentence(start=0, end=14)
    s1 = Sentence(start=15, end=29)
    s2 = Sentence(start=30, end=prefix_len - 1)
    sentences = [s0, s1, s2]
    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float32):
        clean_logits_list = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            out = model(full_input)
            clean_logits_list.append(out.logits.float().detach())
    return input_ids, continuations, sentences, clean_logits_list


# ---------------------------------------------------------------------------
# Reference sequential implementation (original approach)
# ---------------------------------------------------------------------------
def sequential_mean_kl(model, objective_fn, input_ids, continuations, clean_logits_list, prefix_len):
    def _pos_mask(full_len, prefix_len):
        m = torch.zeros(1, full_len)
        m[0, prefix_len - 1 : full_len - 1] = 1.0
        return m

    obj_sum = 0.0
    for cont_idx, cont in enumerate(continuations):
        full_input = torch.cat([input_ids, cont], dim=-1)
        full_len = full_input.shape[-1]
        clean_logits = clean_logits_list[cont_idx][:, :full_len]
        with torch.amp.autocast("cpu", dtype=torch.float32):
            logits = model(full_input).logits
        obj = objective_fn(clean_logits, logits.float(), _pos_mask(full_len, prefix_len),
                           token_ids=full_input)
        obj_sum += obj.item()
    return obj_sum / len(continuations)


# ---------------------------------------------------------------------------
# Helpers to call the two new paths via NodewiseActivationPatching directly
# ---------------------------------------------------------------------------
def make_discoverer(model, objective_fn):
    return NodewiseActivationPatching(
        model=model,
        tokenizer=None,
        layers=[0, 1],
        objective_fn=objective_fn,
        sentence_gap=1,
        ablate_non_target_layers=False,
        renormalize_masked_attention=True,
        mask_granularity="pair",
    )


def batched_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len):
    """Call _compute_mean_kl without prefix cache (batched full-sequence forward)."""
    return discoverer._compute_mean_kl(
        input_ids, continuations, clean_logits_list, prefix_len, torch.device("cpu"),
        prefix_kv_cache=None,
    )


def cached_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len):
    """Call _compute_mean_kl with a fresh prefix KV cache."""
    prefix_kv = discoverer._get_prefix_kv_cache(input_ids[:, :-1], torch.device("cpu"))
    return discoverer._compute_mean_kl(
        input_ids, continuations, clean_logits_list, prefix_len, torch.device("cpu"),
        prefix_kv_cache=prefix_kv,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building tiny Llama model...")
    model = make_tiny_llama()
    input_ids, continuations, sentences, clean_logits_list = make_inputs(model)
    prefix_len = input_ids.shape[-1]
    objective_fn = get_objective("kl_divergence")
    discoverer = make_discoverer(model, objective_fn)
    N_BENCH = 10

    print(f"\nConfig: prefix={prefix_len} tokens, {N_CONTS} branches (max_cont={CONT_LEN}), "
          f"device={DEVICE}")

    # ------------------------------------------------------------------
    # 1. Correctness: all three methods should agree
    # ------------------------------------------------------------------
    print("\n--- Correctness check ---")
    with torch.no_grad():
        kl_seq = sequential_mean_kl(model, objective_fn, input_ids, continuations,
                                    clean_logits_list, prefix_len)
        kl_bat = batched_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len)
        kl_cac = cached_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len)

    print(f"  Sequential : {kl_seq:.6f}")
    print(f"  Batched    : {kl_bat:.6f}  diff={abs(kl_bat - kl_seq):.2e}")
    print(f"  Cached     : {kl_cac:.6f}  diff={abs(kl_cac - kl_seq):.2e}")
    assert abs(kl_bat - kl_seq) < 1e-4, f"Batched KL mismatch: {abs(kl_bat - kl_seq):.2e}"
    assert abs(kl_cac - kl_seq) < 1e-4, f"Cached  KL mismatch: {abs(kl_cac - kl_seq):.2e}"
    print("  PASS — all three values agree within 1e-4")

    # ------------------------------------------------------------------
    # 2. Timing comparison
    # ------------------------------------------------------------------
    print(f"\n--- Per-probe timing ({N_BENCH} probes each) ---")

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            sequential_mean_kl(model, objective_fn, input_ids, continuations,
                                clean_logits_list, prefix_len)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            batched_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len)
        t_bat = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            cached_kl(discoverer, input_ids, continuations, clean_logits_list, prefix_len)
        t_cac = time.perf_counter() - t0

    print(f"  Sequential : {t_seq/N_BENCH*1000:6.1f} ms/probe  (1.0×)")
    print(f"  Batched    : {t_bat/N_BENCH*1000:6.1f} ms/probe  ({t_seq/t_bat:.2f}×)")
    print(f"  Cached     : {t_cac/N_BENCH*1000:6.1f} ms/probe  ({t_seq/t_cac:.2f}×)")
    print("  Note: cached uses B=1 sequential to avoid OOM; GPU speedup dominates for long prefixes")
    assert t_bat < t_seq, "Batched should be faster than sequential"
    print("  PASS")

    # ------------------------------------------------------------------
    # 3. End-to-end discover() (uses prefix cache internally)
    # ------------------------------------------------------------------
    print("\n--- Full discover() (pair granularity, layers=[0,1]) ---")
    with torch.no_grad():
        t0 = time.perf_counter()
        mask = discoverer.discover(
            input_ids=input_ids,
            sentences=sentences,
            continuations=continuations,
            mask_mode="prefix",
            num_prefix_sentences=len(sentences),
        )
        elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.3f}s  |  sentences={len(sentences)}")
    print("  PASS")

    # ------------------------------------------------------------------
    # 4. log_prob_loss objective (exercises token_ids path)
    # ------------------------------------------------------------------
    print("\n--- log_prob_loss correctness check ---")
    lp_fn = get_objective("log_prob")
    lp_discoverer = make_discoverer(model, lp_fn)
    with torch.no_grad():
        lp_bat = lp_discoverer._compute_mean_kl(
            input_ids, continuations, clean_logits_list, prefix_len, torch.device("cpu"),
            prefix_kv_cache=None,
        )
        lp_cac = lp_discoverer._compute_mean_kl(
            input_ids, continuations, clean_logits_list, prefix_len, torch.device("cpu"),
            prefix_kv_cache=lp_discoverer._get_prefix_kv_cache(
                input_ids[:, :-1], torch.device("cpu")
            ),
        )
    print(f"  Batched log_prob : {lp_bat:.6f}")
    print(f"  Cached  log_prob : {lp_cac:.6f}  diff={abs(lp_cac - lp_bat):.2e}")
    assert abs(lp_cac - lp_bat) < 1e-4, f"log_prob mismatch: {abs(lp_cac - lp_bat):.2e}"
    print("  PASS")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
