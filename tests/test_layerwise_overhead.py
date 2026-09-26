"""Compare peak GPU memory: all-layers vs layerwise vs layerwise+detach AP-IG.

Three modes:

  all_layers        : clean + corrupted attention maps for ALL N target layers held
                      simultaneously; one forward+backward through the full model.

  layerwise         : 1 layer's captures at a time; one full forward+backward per layer.
                      Same peak as all_layers minus the extra capture tensors.

  layerwise_detach  : same as layerwise, but the hidden states are detached before
                      layer n so that backprop only runs through layers n..num_layers.
                      Saves activation memory proportional to n / num_layers.

The detach works by registering a forward pre-hook on model.model.layers[n] that
calls .detach() on the hidden states. PyTorch will not record a backward graph for
any operation before that point, so layers 0..n-1 store no activations for the
backward pass and their parameter gradients are never computed.

Usage:
    python tests/test_layerwise_overhead.py
    python tests/test_layerwise_overhead.py --model_name Qwen/Qwen3-4B
    python tests/test_layerwise_overhead.py --target_layers 1 8 14 20 28
"""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM

MODEL_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_TARGET_LAYERS = [8, 14, 20, 28]


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


def capture_size_gb(num_heads, seq_len, n_layers):
    """Memory for clean + corrupted fp32 attention maps for n_layers."""
    return 2 * n_layers * num_heads * seq_len * seq_len * 4 / (1024**3)


def _run_forward_backward(model, input_ids):
    mask_scalar = torch.tensor(1.0, requires_grad=True, device=input_ids.device)
    with torch.amp.autocast("cuda"):
        outputs = model(input_ids)
        logits = outputs.logits
    loss = (logits.float() * mask_scalar).mean()
    loss.backward()
    del outputs, logits, loss, mask_scalar


def _alloc_captures(num_heads, seq_len, n_layers, device):
    return [
        torch.zeros(1, num_heads, seq_len, seq_len, dtype=torch.float32, device=device)
        for _ in range(2 * n_layers)
    ]


def _run_one(model, input_ids, num_heads, n_capture_layers, detach_at_layer=None):
    """Run one forward+backward with n_capture_layers worth of dummy captures held.

    If detach_at_layer is set, registers a pre-hook that detaches hidden states
    before that layer — stopping gradient flow into earlier layers.

    Returns (peak_gb, status).
    """
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()

    seq_len = input_ids.shape[-1]
    captured = None
    hook_handle = None
    try:
        captured = _alloc_captures(num_heads, seq_len, n_capture_layers, input_ids.device)

        if detach_at_layer is not None:
            decoder_layer = model.model.layers[detach_at_layer]

            def _detach_hook(_module, args):
                # args[0] is hidden_states; detach cuts the backward graph here
                return (args[0].detach(),) + args[1:]

            hook_handle = decoder_layer.register_forward_pre_hook(_detach_hook)

        _run_forward_backward(model, input_ids)
        return peak_allocated_gb(), "OK"

    except torch.OutOfMemoryError:
        return None, "OOM"

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        try:
            del captured
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def test_all_layers(model, seq_len, target_layers, num_heads, device="cuda"):
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    try:
        return _run_one(model, input_ids, num_heads, n_capture_layers=len(target_layers))
    finally:
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()


def test_layerwise(model, seq_len, target_layers, num_heads, device="cuda"):
    """One forward+backward per layer, 1 layer's captures at a time, no detach."""
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    try:
        peak_max = 0.0
        for layer in target_layers:
            peak, status = _run_one(model, input_ids, num_heads, n_capture_layers=1)
            if status == "OOM":
                return None, "OOM"
            peak_max = max(peak_max, peak)
        return peak_max, "OK"
    finally:
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()


def test_layerwise_detach(model, seq_len, target_layers, num_heads, device="cuda"):
    """One forward+backward per layer with gradient detach before that layer.

    Processes layers from last to first (reverse order), since later layers save
    the most activation memory (fewer layers need backward storage).
    """
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    try:
        peak_max = 0.0
        for layer in sorted(target_layers, reverse=True):
            peak, status = _run_one(
                model, input_ids, num_heads,
                n_capture_layers=1,
                detach_at_layer=layer,
            )
            if status == "OOM":
                return None, "OOM"
            peak_max = max(peak_max, peak)
        return peak_max, "OK"
    finally:
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_DEFAULT)
    parser.add_argument(
        "--target_layers", type=int, nargs="+", default=DEFAULT_TARGET_LAYERS
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 6000, 7000],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    n_target = len(args.target_layers)
    total_vram = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
    model_mem = current_allocated_gb()

    print(f"\nGPU total VRAM  : {total_vram:.1f} GB")
    print(f"Model footprint : {model_mem:.1f} GB")
    print(f"Free after load : {total_vram - model_mem:.1f} GB")
    print(f"Total layers    : {num_layers}")
    print(f"Target layers   : {args.target_layers}  ({n_target} layers)")
    print(f"Attention heads : {num_heads}")
    # Show what fraction of the backward each layer saves
    print(f"\nDetach savings by layer (fraction of backward skipped):")
    for l in sorted(args.target_layers, reverse=True):
        print(f"  layer {l:>2d} : skip layers 0..{l-1}  ({l}/{num_layers} = {l/num_layers:.0%} of backward)")
    print()

    c = 12
    print(
        f"{'seq_len':>8}  "
        f"{'all_layers':>{c}}  {'':>4}  "
        f"{'layerwise':>{c}}  {'':>4}  "
        f"{'lw+detach':>{c}}  {'':>4}"
    )
    print("-" * 70)

    for seq_len in args.seq_lens:
        all_peak,  all_st  = test_all_layers(model, seq_len, args.target_layers, num_heads, args.device)
        lw_peak,   lw_st   = test_layerwise(model, seq_len, args.target_layers, num_heads, args.device)
        det_peak,  det_st  = test_layerwise_detach(model, seq_len, args.target_layers, num_heads, args.device)

        def fmt(v, s): return f"{v:{c}.1f}  {s:>4}" if v is not None else f"{'---':>{c}}  {s:>4}"

        print(
            f"{seq_len:>8}  "
            f"{fmt(all_peak, all_st)}  "
            f"{fmt(lw_peak, lw_st)}  "
            f"{fmt(det_peak, det_st)}"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
