"""Test peak GPU memory for eager attention + backward at various sequence lengths.

Simulates the exact memory pattern of circuit discovery:
- Full 32-layer forward pass with eager attention
- Backward pass through all layers (to propagate grads to a mask at one layer)
- Optionally patches attention to stay in float16 instead of float32
  (simulating removing the .float() cast in common.py apply_sentence_mask)

Usage:
    python expts/test_eager_attn_memory.py                     # float32 attn (current behaviour)
    python expts/test_eager_attn_memory.py --fp16_attn         # float16 attn (hypothetical)
    python expts/test_eager_attn_memory.py --seq_lens 1000 2000 3500 5000 8500
"""

import argparse
import gc
import torch
from transformers import AutoModelForCausalLM

MODEL_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


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


def test_seq_len(model, seq_len: int, fp16_attn: bool = False, device: str = "cuda"):
    """Run one forward+backward and return peak memory in GB.

    fp16_attn=False (default): autocast lets softmax run in float32, matching
                               the .float() cast in common.py apply_sentence_mask.
    fp16_attn=True:            autocast forces float16 throughout, keeping
                               attention weights in float16 — hypothetical optimisation.
    """
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()

    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    mask_scalar = torch.tensor(0.5, requires_grad=True, device=device)

    autocast_dtype = torch.float16 if fp16_attn else torch.bfloat16

    try:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            outputs = model(input_ids)
            logits = outputs.logits

        loss = (logits.float() * mask_scalar).mean()
        loss.backward()

        peak = peak_allocated_gb()
        status = "OK"
    except torch.OutOfMemoryError as e:
        peak = None
        status = f"OOM"
    finally:
        try:
            del input_ids, outputs, logits, loss, mask_scalar
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    return peak, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_DEFAULT)
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 7000, 8500],
    )
    parser.add_argument(
        "--fp16_attn",
        action="store_true",
        help="Use float16 autocast (simulates keeping attn weights in fp16 "
        "instead of the float32 cast in common.py apply_sentence_mask).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    total_vram = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
    model_mem = current_allocated_gb()
    attn_dtype_str = "float16" if args.fp16_attn else "float32"
    print(f"\nGPU total VRAM  : {total_vram:.1f} GB")
    print(f"Model footprint : {model_mem:.1f} GB")
    print(f"Free after load : {total_vram - model_mem:.1f} GB")
    print(f"Attention dtype : {attn_dtype_str}")
    print()
    print(
        f"{'seq_len':>10}  {'peak_mem_GB':>12}  {'headroom_GB':>12}  {'status':<6}  theoretical_attn_GB"
    )
    print("-" * 72)

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    attn_bytes = 2 if args.fp16_attn else 4  # float16=2, float32=4

    for seq_len in args.seq_lens:
        theoretical_attn_gb = (
            num_layers * num_heads * seq_len * seq_len * attn_bytes / (1024**3)
        )
        peak, status = test_seq_len(
            model, seq_len, fp16_attn=args.fp16_attn, device=args.device
        )

        if peak is not None:
            headroom = total_vram - peak
            print(
                f"{seq_len:>10}  {peak:>12.1f}  {headroom:>12.1f}  {status:<6}  {theoretical_attn_gb:.1f}"
            )
        else:
            print(
                f"{seq_len:>10}  {'---':>12}  {'---':>12}  {status:<6}  {theoretical_attn_gb:.1f}"
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
