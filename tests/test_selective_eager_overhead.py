"""Test peak GPU memory: selective eager attention vs full eager.

Loads the model with SDPA (memory-efficient), then temporarily swaps
batches of k consecutive layers to eager attention for the forward+backward.
Only the eager layers materialise the full Q×K^T matrix; SDPA layers do not.

Modes compared for each seq_len:
  full_eager          : all layers eager (baseline / current behaviour)
  batch_size=k layers : k consecutive layers swapped to eager at a time;
                        worst-case peak across all batches is reported

Usage:
    python tests/test_selective_eager_overhead.py
    python tests/test_selective_eager_overhead.py --model_name Qwen/Qwen3-4B
    python tests/test_selective_eager_overhead.py --batch_sizes 1 4 8 --seq_lens 2000 4000 6000
"""

import argparse
import gc
import types

import torch
from transformers import AutoModelForCausalLM

from utils.circuit_discovery.common import make_attention_forward
from utils.utils import get_attention_module

MODEL_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def bytes_to_gb(b):
    return b / (1024**3)


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def current_allocated_gb():
    torch.cuda.synchronize()
    return bytes_to_gb(torch.cuda.memory_allocated())


def peak_allocated_gb():
    torch.cuda.synchronize()
    return bytes_to_gb(torch.cuda.max_memory_allocated())


# ---------------------------------------------------------------------------
# Eager swap helpers
# ---------------------------------------------------------------------------

def _swap_layers_to_eager(model, layer_indices):
    """Replace SDPA forward with custom eager forward for the given layers.

    Returns a list of (attn_module, original_forward) for restoration.
    """
    model_type = model.config.model_type
    eager_fwd = make_attention_forward(model_type, injection_fn=None)
    saved = []
    for idx in layer_indices:
        attn = get_attention_module(model, idx)
        saved.append((attn, attn.forward))
        attn.forward = types.MethodType(eager_fwd, attn)
    return saved


def _restore_layers(saved):
    for attn, original_fwd in saved:
        attn.forward = original_fwd


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------

def _run_forward_backward(model, input_ids):
    mask_scalar = torch.tensor(1.0, requires_grad=True, device=input_ids.device)
    with torch.amp.autocast("cuda"):
        outputs = model(input_ids)
        logits = outputs.logits
    loss = (logits.float() * mask_scalar).mean()
    loss.backward()
    del outputs, logits, loss, mask_scalar


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_full_eager(model, seq_len, device="cuda"):
    """All layers already loaded as eager — one forward+backward."""
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()

    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    try:
        _run_forward_backward(model, input_ids)
        return peak_allocated_gb(), "OK"
    except torch.OutOfMemoryError:
        return None, "OOM"
    finally:
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()


def test_selective_eager(model, seq_len, batch_size, device="cuda"):
    """Swap batch_size consecutive layers to eager at a time, rest stay SDPA.

    Iterates over all layers in non-overlapping windows of batch_size.
    Returns the worst-case (maximum) peak memory across all windows.
    """
    num_layers = model.config.num_hidden_layers
    batches = [
        list(range(i, min(i + batch_size, num_layers)))
        for i in range(0, num_layers, batch_size)
    ]

    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    peak_max = 0.0
    try:
        for batch in batches:
            gc.collect()
            torch.cuda.empty_cache()
            reset_peak()

            saved = _swap_layers_to_eager(model, batch)
            try:
                _run_forward_backward(model, input_ids)
                peak_max = max(peak_max, peak_allocated_gb())
            except torch.OutOfMemoryError:
                return None, "OOM"
            finally:
                _restore_layers(saved)

        return peak_max, "OK"
    finally:
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_DEFAULT)
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Number of layers swapped to eager at a time. 'all' is always shown.",
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 6000, 7000],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load SDPA model (used for all selective-eager tests)
    print(f"Loading {args.model_name} with SDPA attention...")
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    model_sdpa.eval()

    # Load eager model (used only for the full_eager baseline column)
    print(f"Loading {args.model_name} with eager attention (baseline)...")
    model_eager = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model_eager.eval()

    num_layers = model_sdpa.config.num_hidden_layers
    num_heads  = model_sdpa.config.num_attention_heads
    total_vram = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
    # measure after both models are loaded
    both_models_mem = current_allocated_gb()

    print(f"\nGPU total VRAM     : {total_vram:.1f} GB")
    print(f"Both models loaded : {both_models_mem:.1f} GB  (each ~{both_models_mem/2:.1f} GB)")
    print(f"Total layers       : {num_layers}")
    print(f"Attention heads    : {num_heads}")
    print(f"Batch sizes        : {args.batch_sizes}  (+ full_eager baseline)")
    print()

    # Header
    batch_cols = args.batch_sizes
    col = 11
    header_parts = [f"{'seq_len':>8}", f"{'full_eager':>{col}}  {'':>3}"]
    for k in batch_cols:
        header_parts.append(f"{'k='+str(k):>{col}}  {'':>3}")
    print("  ".join(header_parts))
    print("-" * (10 + (col + 6) * (1 + len(batch_cols))))

    for seq_len in args.seq_lens:
        full_peak, full_st = test_full_eager(model_eager, seq_len, args.device)

        row_parts = [f"{seq_len:>8}"]
        val = f"{full_peak:{col}.1f}" if full_peak is not None else f"{'---':>{col}}"
        row_parts.append(f"{val}  {full_st:>3}")

        for k in batch_cols:
            peak, st = test_selective_eager(model_sdpa, seq_len, k, args.device)
            val = f"{peak:{col}.1f}" if peak is not None else f"{'---':>{col}}"
            row_parts.append(f"{val}  {st:>3}")

        print("  ".join(row_parts))

    del model_sdpa, model_eager
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
