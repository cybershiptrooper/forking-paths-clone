"""Test that attention monkey-patching works correctly for Qwen3 models.

Loads a small Qwen3 model, runs "The capital of France is", and verifies
that masking attention to "France" tokens reduces P(Paris).
"""

import types
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import get_attention_module, Sentence
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask
from utils.masks import build_gap_filter


def load_model(model_name: str, device: str = "cuda"):
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


def get_token_probs(model, input_ids):
    """Return probability distribution at the last position."""
    with torch.no_grad():
        logits = model(input_ids).logits
    return torch.softmax(logits[0, -1].float(), dim=-1)


def find_token_span(token_ids_list, tokenizer, target_text):
    """Find the token indices that correspond to *target_text* in the tokenized input."""
    full_text = tokenizer.decode(token_ids_list)
    # Find character-level position
    char_start = full_text.lower().find(target_text.lower())
    if char_start == -1:
        raise ValueError(f"Could not find '{target_text}' in: {full_text!r}")
    char_end = char_start + len(target_text)

    # Map character positions to token indices
    tok_start = None
    tok_end = None
    cursor = 0
    for i, tid in enumerate(token_ids_list):
        tok_text = tokenizer.decode([tid])
        tok_char_start = cursor
        tok_char_end = cursor + len(tok_text)
        if tok_start is None and tok_char_end > char_start:
            tok_start = i
        if tok_char_end >= char_end:
            tok_end = i
            break
        cursor = tok_char_end

    if tok_start is None or tok_end is None:
        raise ValueError(f"Could not map '{target_text}' to token indices")
    return tok_start, tok_end


def run_test(model_name: str, device: str = "cuda"):
    model, tokenizer = load_model(model_name, device=device)
    model_type = model.config.model_type
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    print(f"Model type: {model_type}, heads: {num_heads}, layers: {num_layers}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    token_ids_list = input_ids[0].tolist()

    # Find "Paris" token id — try " Paris" (with space prefix) first since
    # the model predicts the next token including its leading space.
    paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
    if not paris_ids:
        paris_ids = tokenizer.encode("Paris", add_special_tokens=False)
    paris_id = paris_ids[0]  # first subword token
    print(f"Paris token id: {paris_id} ('{tokenizer.decode([paris_id])}')")

    # Print tokenization
    print(f"Tokens: {[tokenizer.decode([t]) for t in token_ids_list]}")

    # ---- Clean forward ----
    clean_probs = get_token_probs(model, input_ids)
    p_paris_clean = clean_probs[paris_id].item()
    top5_clean = torch.topk(clean_probs, 5)
    print(f"\n--- Clean forward ---")
    print(f"P(Paris) = {p_paris_clean:.4f}")
    print("Top-5 predictions:")
    for prob, idx in zip(top5_clean.values, top5_clean.indices):
        print(f"  {tokenizer.decode([idx.item()]):>15s}: {prob.item():.4f}")

    # ---- Identify "France" tokens ----
    france_start, france_end = find_token_span(token_ids_list, tokenizer, "France")
    print(f"\n'France' spans tokens [{france_start}:{france_end}]")
    print(f"  = {tokenizer.decode(token_ids_list[france_start:france_end + 1])!r}")

    # ---- Build mask: 2 sentences ----
    # Sentence 0: everything except France
    # Sentence 1: France tokens
    seq_len = input_ids.shape[-1]
    token_to_sent = torch.full((seq_len,), 0, dtype=torch.long)  # all -> sent 0
    token_to_sent[france_start: france_end + 1] = 1  # France -> sent 1
    token_to_sent = token_to_sent.to(device)

    num_sents = 2
    gap_filter = build_gap_filter(num_sents, sentence_gap=0, device=device)

    # mask: (num_heads, 2, 2) — block all attention TO France (column 1)
    mask = torch.ones(num_heads, num_sents, num_sents, device=device)
    mask[:, :, 1] = 0.0  # no head can attend to sentence 1 (France)

    # ---- Patch all layers ----
    forward_fn = make_attention_forward(model_type, apply_sentence_mask)
    handles = []
    for layer_idx in range(num_layers):
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = mask
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = True
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append((attn_module, original_forward))

    # ---- Masked forward ----
    masked_probs = get_token_probs(model, input_ids)
    p_paris_masked = masked_probs[paris_id].item()
    top5_masked = torch.topk(masked_probs, 5)
    print(f"\n--- Masked forward (France blocked) ---")
    print(f"P(Paris) = {p_paris_masked:.4f}")
    print("Top-5 predictions:")
    for prob, idx in zip(top5_masked.values, top5_masked.indices):
        print(f"  {tokenizer.decode([idx.item()]):>15s}: {prob.item():.4f}")

    # ---- Restore ----
    for attn_module, original_forward in handles:
        attn_module.forward = original_forward
        for attr in ["_circuit_mask", "_token_to_sent", "_gap_filter", "_renormalize_masked_attn"]:
            if hasattr(attn_module, attr):
                delattr(attn_module, attr)

    # ---- Verify restoration ----
    restored_probs = get_token_probs(model, input_ids)
    p_paris_restored = restored_probs[paris_id].item()
    print(f"\n--- Restored forward ---")
    print(f"P(Paris) = {p_paris_restored:.4f}")

    # ---- Results ----
    drop = p_paris_clean - p_paris_masked
    ratio = p_paris_clean / max(p_paris_masked, 1e-10)
    restored_ok = abs(p_paris_clean - p_paris_restored) < 1e-4

    print(f"\n{'=' * 60}")
    print(f"RESULTS for {model_name} (model_type={model_type})")
    print(f"{'=' * 60}")
    print(f"P(Paris) clean:    {p_paris_clean:.6f}")
    print(f"P(Paris) masked:   {p_paris_masked:.6f}")
    print(f"P(Paris) restored: {p_paris_restored:.6f}")
    print(f"Drop:              {drop:.6f} ({ratio:.1f}x)")
    print(f"Restoration match: {'PASS' if restored_ok else 'FAIL'}")
    print(f"Masking effect:    {'PASS' if drop > 0.01 else 'FAIL'} (drop > 0.01)")

    if not restored_ok:
        print("WARNING: Restored P(Paris) does not match clean — patching may be leaking state.")
    if drop <= 0.01:
        print("WARNING: Masking France had negligible effect — patching may not be working.")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "p_paris_clean": p_paris_clean,
        "p_paris_masked": p_paris_masked,
        "p_paris_restored": p_paris_restored,
        "pass": drop > 0.01 and restored_ok,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test attention patching for Qwen3 models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model to test patching on.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_test(args.model_name, device=args.device)
