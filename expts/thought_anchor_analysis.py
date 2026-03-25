"""Figure 6 analysis from Thought Anchors (Bogdan et al., arXiv 2506.19143).

Computes two analyses given a mask JSON with sentence boundaries:

1. **Attention suppression matrix**: For each sentence, suppress all attention
   to it across all layers/heads and measure KL divergence at every other
   sentence. Produces an (S, S) causal impact matrix.

2. **Receiver head attention**: Capture clean attention from all heads, aggregate
   to sentence level, find the top-K heads with highest kurtosis (sharpest
   focus), and compute per-sentence vertical attention (how much attention each
   sentence receives from later sentences via these heads).

Results are saved into the mask JSON under ``metadata``.

Usage:
    python expts/thought_anchor_analysis.py \\
        --mask results/circuit_discovery/test_global_2.json

    python expts/thought_anchor_analysis.py \\
        --mask results/circuit_discovery/test_global_2.json \\
        --top_k 32 --min_gap 4
"""

import argparse
import json
import os
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import Sentence, get_attention_module, set_seed, clear_cuda
from utils.circuit_eval import (
    build_token_to_sent_map,
    install_mask_hooks,
    remove_handles,
)
from utils.circuit_discovery.common import make_attention_forward
from utils.circuit_discovery.base import AblationHandle


# ------------------------------------------------------------------
# Model loading (same as learn_circuit.py)
# ------------------------------------------------------------------

def load_model_eager(model_name: str, device: str = "cuda"):
    """Load model with eager attention for circuit analysis."""
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


# ------------------------------------------------------------------
# Attention capture
# ------------------------------------------------------------------

def aggregate_attn_to_sentences(attn_weights, token_to_sent, num_sents):
    """Aggregate token-level attention to sentence-level using einsum.

    Args:
        attn_weights: (1, H, q_len, k_len) post-softmax attention.
        token_to_sent: (total_seq_len,) int tensor, -1 for unassigned tokens.
        num_sents: number of sentences.

    Returns:
        (H, S, S) mean attention per head per sentence pair, on CPU.
    """
    attn = attn_weights[0].float()  # (H, q_len, k_len)
    H, q_len, k_len = attn.shape
    device = attn.device

    q_sent = token_to_sent[:q_len].to(device)
    k_sent = token_to_sent[:k_len].to(device)

    q_onehot = torch.zeros(q_len, num_sents, device=device)
    q_onehot[q_sent >= 0, q_sent[q_sent >= 0]] = 1.0

    k_onehot = torch.zeros(k_len, num_sents, device=device)
    k_onehot[k_sent >= 0, k_sent[k_sent >= 0]] = 1.0

    sent_attn = torch.einsum("hqk,qs,kt->hst", attn, q_onehot, k_onehot)
    counts = torch.einsum("qs,kt->st", q_onehot, k_onehot).clamp(min=1)

    return (sent_attn / counts.unsqueeze(0)).cpu()


def capture_clean_pass(model, input_ids, token_to_sent, num_sents):
    """Run a clean forward pass, capturing sentence-level attention from every layer.

    Returns:
        clean_logits: (1, seq_len, vocab) on CPU.
        sent_attns: dict[int, Tensor] mapping layer → (H, S, S).
    """
    num_layers = model.config.num_hidden_layers

    def capture_injection(module, attn_weights, q_len, k_len, cache_position):
        module._sent_attn = aggregate_attn_to_sentences(
            attn_weights, token_to_sent, num_sents,
        )
        return attn_weights  # pass through unmodified

    forward_fn = make_attention_forward(model.config.model_type, capture_injection)
    handles = []
    for layer_idx in range(num_layers):
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    clean_logits = outputs.logits.cpu()

    sent_attns = {}
    for layer_idx in range(num_layers):
        attn_module = get_attention_module(model, layer_idx)
        sent_attns[layer_idx] = attn_module._sent_attn
        del attn_module._sent_attn

    remove_handles(handles)
    return clean_logits, sent_attns


# ------------------------------------------------------------------
# Receiver head analysis
# ------------------------------------------------------------------

def compute_receiver_heads(sent_attns, top_k=32, min_gap=4):
    """Find high-kurtosis (receiver) heads and their vertical attention.

    Vertical attention for sentence j = mean attention j receives from
    sentences i where i − j ≥ min_gap, averaged across tokens.

    Returns:
        receiver_heads: list of (layer, head, kurtosis) sorted descending.
        vertical_attention: list[float] of length S.
    """
    num_sents = next(iter(sent_attns.values())).shape[-1]
    all_heads = []  # (layer, head, kurtosis, vert_vector)

    for layer_idx, attn in sent_attns.items():
        H = attn.shape[0]
        for h in range(H):
            head_attn = attn[h]  # (S, S): head_attn[i, j] = attn from sent i → sent j

            # Vertical attention per sentence j
            vert = torch.zeros(num_sents)
            for j in range(num_sents):
                future = head_attn[j + min_gap:, j]  # attention TO j from i >= j+min_gap
                if future.numel() > 0:
                    vert[j] = future.mean()

            # Excess kurtosis
            std = vert.std()
            if std > 1e-8:
                kurt = ((vert - vert.mean()) / std).pow(4).mean().item() - 3.0
            else:
                kurt = 0.0

            all_heads.append((layer_idx, h, kurt, vert))

    all_heads.sort(key=lambda x: x[2], reverse=True)
    top = all_heads[:top_k]

    avg_vertical = torch.stack([h[3] for h in top]).mean(dim=0)
    receiver_heads = [(l, h, k) for l, h, k, _ in top]
    return receiver_heads, avg_vertical.tolist()


# ------------------------------------------------------------------
# Attention suppression matrix
# ------------------------------------------------------------------

def compute_suppression_matrix(model, input_ids, sentences, token_to_sent, clean_logits):
    """Suppress attention to each sentence and measure KL at all others.

    Returns:
        list[list[float]] of shape (S, S) where [suppressed][affected] = KL.
    """
    num_sents = len(sentences)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    all_layers = list(range(num_layers))
    device = next(model.parameters()).device
    seq_len = input_ids.shape[-1]

    # No gap filter — we want to suppress any column freely
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool, device=device)

    # Clean log-probs (on CPU to save GPU memory)
    log_clean = F.log_softmax(clean_logits.float(), dim=-1)

    # Install hooks once with an all-ones mask, then swap the mask each iteration
    ones_mask = torch.ones(num_heads, num_sents, num_sents, device=device)
    binary_masks = {layer: ones_mask for layer in all_layers}
    handles = install_mask_hooks(
        model, all_layers, binary_masks, token_to_sent, gap_filter, renormalize=True,
    )

    suppression_matrix = torch.zeros(num_sents, num_sents)
    model.eval()

    for s_suppress in range(num_sents):
        # Build suppression mask: zero out column s_suppress
        mask = torch.ones(num_heads, num_sents, num_sents, device=device)
        mask[:, :, s_suppress] = 0.0

        # Update mask on every layer's attention module
        for layer_idx in all_layers:
            attn_module = get_attention_module(model, layer_idx)
            attn_module._circuit_mask = mask

        with torch.no_grad():
            logits = model(input_ids).logits

        log_masked = F.log_softmax(logits.float().cpu(), dim=-1)
        kl_tokens = F.kl_div(
            log_masked, log_clean, log_target=True, reduction="none",
        ).sum(dim=-1)[0]  # (seq_len,)

        for s_affected in range(num_sents):
            start = sentences[s_affected].start
            end = min(sentences[s_affected].end, seq_len - 1)
            if start >= seq_len:
                continue
            suppression_matrix[s_suppress, s_affected] = (
                kl_tokens[start : end + 1].mean().item()
            )

        max_kl = suppression_matrix[s_suppress].max().item()
        print(f"  [{s_suppress}/{num_sents - 1}] max KL = {max_kl:.4f}")

        del logits, log_masked, kl_tokens
        torch.cuda.empty_cache()

    remove_handles(handles)
    return suppression_matrix.tolist()


# ------------------------------------------------------------------
# Input reconstruction
# ------------------------------------------------------------------

def reconstruct_input(sentences_raw, tokenizer, device):
    """Reconstruct input_ids from sentence texts stored in the mask JSON.

    Concatenates sentence texts and prepends the BOS token to approximate
    the original tokenized input.
    """
    full_text = "".join(s["text"] for s in sentences_raw)
    token_ids = tokenizer.encode(full_text, add_special_tokens=True)
    input_ids = torch.tensor([token_ids], device=device)

    expected_end = sentences_raw[-1]["end"] + 1
    actual_len = input_ids.shape[-1]
    if abs(actual_len - expected_end) > 3:
        print(
            f"  WARNING: Reconstructed {actual_len} tokens but mask expects "
            f"{expected_end}. Provide --prompt for exact reconstruction."
        )
    # Clip or pad to match expected length
    if actual_len > expected_end:
        input_ids = input_ids[:, :expected_end]
    return input_ids


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thought Anchors Figure 6: suppression matrix + receiver heads.",
    )
    parser.add_argument("--mask", required=True, help="Path to mask JSON file")
    parser.add_argument("--prompt", default=None, help="Original prompt text")
    parser.add_argument("--data_path", default=None, help="Path to data JSON")
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=32, help="Number of receiver heads")
    parser.add_argument("--min_gap", type=int, default=4, help="Min sentence gap for receiver analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load mask JSON
    with open(args.mask) as f:
        data = json.load(f)

    model_name = data["model_name"]
    sentences_raw = data["sentences"]
    sentences = [Sentence(start=s["start"], end=s["end"]) for s in sentences_raw]
    num_sents = len(sentences)

    print(f"Mask: {args.mask}")
    print(f"Model: {model_name}")
    print(f"Sentences: {num_sents}")

    # Load model
    model, tokenizer = load_model_eager(model_name, device=args.device)
    device = next(model.parameters()).device

    # Reconstruct input_ids
    if args.prompt is not None:
        chat = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        # We only have the prompt tokens; the rest were generated.
        # Reconstruct from sentence texts for the generated part.
        prompt_ids = tokenizer.encode(formatted)
        gen_text = "".join(
            s["text"] for s in sentences_raw
            if s["start"] >= len(prompt_ids)
        )
        if gen_text:
            gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            all_ids = prompt_ids + gen_ids
        else:
            all_ids = prompt_ids
        expected_end = sentences_raw[-1]["end"] + 1
        input_ids = torch.tensor([all_ids[:expected_end]], device=device)
    elif args.data_path and args.prompt_index is not None:
        with open(args.data_path) as f:
            records = json.load(f)
        prompt = records[args.prompt_index]["question"]
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(formatted)
        gen_text = "".join(
            s["text"] for s in sentences_raw
            if s["start"] >= len(prompt_ids)
        )
        if gen_text:
            gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            all_ids = prompt_ids + gen_ids
        else:
            all_ids = prompt_ids
        expected_end = sentences_raw[-1]["end"] + 1
        input_ids = torch.tensor([all_ids[:expected_end]], device=device)
    else:
        print("No --prompt provided, reconstructing from sentence texts...")
        input_ids = reconstruct_input(sentences_raw, tokenizer, device)

    print(f"Input: {input_ids.shape[-1]} tokens, {num_sents} sentences")

    # Build token-to-sentence map
    token_to_sent = build_token_to_sent_map(
        sentences, input_ids.shape[-1], device,
    )

    # ------------------------------------------------------------------
    # Step 1: Capture clean attention + logits
    # ------------------------------------------------------------------
    print("\nStep 1: Capturing clean attention from all layers...")
    clean_logits, sent_attns = capture_clean_pass(
        model, input_ids, token_to_sent, num_sents,
    )

    # ------------------------------------------------------------------
    # Step 2: Receiver head analysis
    # ------------------------------------------------------------------
    print(f"\nStep 2: Computing receiver heads (top-{args.top_k}, min_gap={args.min_gap})...")
    receiver_heads, vertical_attention = compute_receiver_heads(
        sent_attns, top_k=args.top_k, min_gap=args.min_gap,
    )
    del sent_attns

    print("Top 5 receiver heads:")
    for l, h, k in receiver_heads[:5]:
        print(f"  Layer {l}, Head {h}: kurtosis = {k:.2f}")

    # ------------------------------------------------------------------
    # Step 3: Suppression matrix
    # ------------------------------------------------------------------
    print(f"\nStep 3: Computing suppression matrix ({num_sents} forward passes)...")
    suppression_matrix = compute_suppression_matrix(
        model, input_ids, sentences, token_to_sent, clean_logits,
    )

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    data["metadata"]["suppression_matrix"] = suppression_matrix
    data["metadata"]["receiver_head_attention"] = vertical_attention
    data["metadata"]["receiver_heads"] = [
        {"layer": l, "head": h, "kurtosis": round(k, 4)}
        for l, h, k in receiver_heads
    ]
    data["metadata"]["figure6_config"] = {
        "top_k": args.top_k,
        "min_gap": args.min_gap,
    }

    with open(args.mask, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Updated {args.mask} with:")
    print(f"  metadata.suppression_matrix: [{num_sents} x {num_sents}]")
    print(f"  metadata.receiver_head_attention: [{num_sents}]")
    print(f"  metadata.receiver_heads: [{len(receiver_heads)} heads]")

    del model
    clear_cuda()


if __name__ == "__main__":
    main()
