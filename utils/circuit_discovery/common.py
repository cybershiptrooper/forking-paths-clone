"""Common functions shared across circuit discovery algorithms.

Contains the patched LlamaAttention forward factory and mask expansion
utilities used by all circuit discovery variants.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils.masks import apply_gap_filter


# ---------------------------------------------------------------------------
# Mask expansion utility
# ---------------------------------------------------------------------------

def expand_sentence_mask_to_tokens(
    mask: torch.Tensor,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand (num_heads, num_sents, num_sents) mask to (num_heads, q_len, k_len).

    Uses PyTorch advanced indexing for differentiability.
    Tokens not assigned to any sentence (index -1) get mask value 1.0.
    Gap-filtered entries (|src-tgt| < gap) also get 1.0.

    Args:
        mask: (num_heads, num_sents, num_sents) with requires_grad
        token_to_sent: (total_seq_len,) int tensor mapping token -> sentence idx (-1 if none)
        gap_filter: (num_sents, num_sents) bool, True where |i-j| < gap
        q_len: number of query positions in current forward
        k_len: number of key positions (total including cache)
        cache_position: (q_len,) absolute positions of current queries, or None

    Returns:
        (num_heads, q_len, k_len) differentiable token-level mask
    """
    num_heads, num_sents, _ = mask.shape
    device = mask.device

    # Get sentence indices for query and key positions
    if cache_position is not None:
        q_sent = token_to_sent[cache_position.long()]  # (q_len,)
    else:
        q_sent = token_to_sent[:q_len]  # (q_len,)
    k_sent = token_to_sent[:k_len]  # (k_len,)

    # Pad mask with sentinel row/col of 1s for tokens not in any sentence
    # padded_mask: (num_heads, num_sents+1, num_sents+1)
    padded = torch.ones(
        num_heads, num_sents + 1, num_sents + 1, device=device, dtype=mask.dtype
    )
    # Apply gap filter: gap entries stay 1.0, non-gap entries use mask values
    effective_mask = apply_gap_filter(mask, gap_filter, fill_value=1.0)
    padded[:, :num_sents, :num_sents] = effective_mask

    # Remap -1 -> num_sents (sentinel index)
    q_idx = q_sent.clone().to(device)
    k_idx = k_sent.clone().to(device)
    q_idx[q_sent == -1] = num_sents
    k_idx[k_sent == -1] = num_sents

    # Advanced indexing: token_mask[h, i, j] = padded[h, q_idx[i], k_idx[j]]
    token_mask = padded[:, q_idx][:, :, k_idx]  # (num_heads, q_len, k_len)
    return token_mask


# ---------------------------------------------------------------------------
# Composable injection building blocks
# ---------------------------------------------------------------------------

def apply_sentence_mask(
    module,
    attn_weights: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply differentiable sentence-level mask to post-softmax attention weights.

    Reads ``_circuit_mask``, ``_token_to_sent``, ``_gap_filter``, and
    ``_renormalize_masked_attn`` from the attention *module*.

    This is a composable building block: algorithm-specific injection
    functions can call this, then add their own logic on top.
    """
    mask = getattr(module, "_circuit_mask", None)
    token_to_sent = getattr(module, "_token_to_sent", None)
    gap_filter = getattr(module, "_gap_filter", None)

    if mask is not None and token_to_sent is not None:
        original_dtype = attn_weights.dtype
        token_mask = expand_sentence_mask_to_tokens(
            mask, token_to_sent, gap_filter, q_len, k_len, cache_position
        )
        # token_mask: (num_heads, q_len, k_len)
        # attn_weights: (bsz, num_heads, q_len, k_len)
        # Compute in float32 for numerical stability, then cast back
        attn_weights = attn_weights.float() * token_mask.unsqueeze(0)
        renormalize = getattr(module, "_renormalize_masked_attn", True)
        if renormalize:
            row_sums = attn_weights.sum(dim=-1, keepdim=True) + 1e-12
            attn_weights = (attn_weights / row_sums).to(original_dtype)
        else:
            attn_weights = attn_weights.to(original_dtype)

    return attn_weights


# ---------------------------------------------------------------------------
# Patched attention forward factory
# ---------------------------------------------------------------------------

def make_llama_attention_forward(
    injection_fn: Optional[Callable] = None,
):
    """Factory that builds a patched LlamaAttention forward.

    The returned function replaces the standard LlamaAttention forward
    and injects algorithm-specific behaviour after softmax via
    *injection_fn*.

    Args:
        injection_fn: Optional callback with signature::

                injection_fn(module, attn_weights, q_len, k_len, cache_position)
                    -> attn_weights

            Called right after softmax (before dropout).  Use composable
            building blocks like ``apply_sentence_mask`` to build these.
            If ``None``, the forward performs standard attention with no
            injection.

    Returns:
        A function suitable for monkey-patching via ``types.MethodType``.
    """

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Patched LlamaAttention forward with pluggable post-softmax injection."""
        bsz, q_len, _ = hidden_states.size()

        # --- Standard Q/K/V Projection ---
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.config.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.config.num_attention_heads * self.head_dim)
                // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                nn.functional.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [
                nn.functional.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [
                nn.functional.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # --- RoPE ---
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # --- KV Cache ---
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # --- GQA repeat ---
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # --- Attention Weights ---
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask
            if attention_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # --- Algorithm-specific injection (post-softmax, pre-dropout) ---
        if injection_fn is not None:
            k_len = key_states.shape[-2]
            attn_weights = injection_fn(self, attn_weights, q_len, k_len, cache_position)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.config.num_attention_heads,
            q_len,
            self.head_dim,
        ):
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        else:
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .reshape(bsz, q_len, self.config.hidden_size)
            )

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.config.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.config.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    nn.functional.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, past_key_values

    return _forward
