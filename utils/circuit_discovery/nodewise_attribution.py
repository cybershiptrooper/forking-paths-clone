"""Nodewise Attribution Patching for attention-based circuit discovery.

Learns per-head, per-layer sentence-to-sentence masks by computing gradients of
a KL divergence objective through differentiable attention. All analyzed layers
are patched simultaneously so gradients capture inter-layer effects.
"""

import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils.circuit_discovery.base import CircuitDiscoveryAlgorithm
from utils.masks import NodeMask
from utils.objectives import get_objective
from utils.utils import Sentence, get_attention_module


# =========================================================================
# Differentiable mask utilities
# =========================================================================


def build_token_to_sentence_map(
    seq_len: int,
    sentences: List[Sentence],
) -> torch.LongTensor:
    """Map each token position to its sentence index (-1 if not in any sentence).

    Args:
        seq_len: Total sequence length.
        sentences: Sentence boundaries (start/end inclusive).

    Returns:
        (seq_len,) LongTensor where map[t] = sentence_idx or -1.
    """
    mapping = torch.full((seq_len,), -1, dtype=torch.long)
    for idx, sent in enumerate(sentences):
        mapping[sent.start : sent.end + 1] = idx
    return mapping


def build_maskable_matrix(
    num_sentences: int,
    sentence_gap: int = 1,
) -> torch.BoolTensor:
    """Create a boolean matrix indicating which sentence pairs are maskable.

    Pairs within sentence_gap of each other are NOT maskable (always kept at 1).

    Args:
        num_sentences: Number of sentences.
        sentence_gap: Minimum distance for an edge to be maskable.

    Returns:
        (num_sentences, num_sentences) BoolTensor. True = maskable edge.
    """
    idx = torch.arange(num_sentences)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return dist >= sentence_gap


def expand_sentence_mask_to_tokens(
    mask: torch.Tensor,
    token_to_sent: torch.LongTensor,
    q_len: int,
    k_len: int,
) -> torch.Tensor:
    """Expand a (num_heads, S, S) sentence-level mask to (1, num_heads, q_len, k_len).

    Uses differentiable advanced indexing via a padded sentinel approach:
    tokens not belonging to any sentence map to a padding row/column of 1.0.

    Args:
        mask: (num_heads, S, S) with requires_grad=True.
        token_to_sent: (max_seq_len,) LongTensor mapping tokens to sentence idx (-1 for none).
        q_len: Number of query positions.
        k_len: Number of key positions.

    Returns:
        (1, num_heads, q_len, k_len) differentiable token-level mask.
    """
    num_heads, num_s, _ = mask.shape
    device = mask.device

    # Pad with a sentinel row/column of 1.0 at index num_s
    padded = torch.ones(
        num_heads, num_s + 1, num_s + 1, device=device, dtype=mask.dtype
    )
    padded[:, :num_s, :num_s] = mask

    # Remap -1 -> num_s (the sentinel index)
    q_idx = token_to_sent[:q_len].clone().to(device)
    k_idx = token_to_sent[:k_len].clone().to(device)
    q_idx = q_idx.clamp(min=0).where(token_to_sent[:q_len] >= 0, torch.tensor(num_s, device=device))
    k_idx = k_idx.clamp(min=0).where(token_to_sent[:k_len] >= 0, torch.tensor(num_s, device=device))

    # Advanced indexing: (num_heads, q_len, k_len) — differentiable
    token_mask = padded[:, q_idx, :][:, :, k_idx]

    return token_mask.unsqueeze(0)  # (1, num_heads, q_len, k_len)


# =========================================================================
# Differentiable attention forward
# =========================================================================


def llama_attention_forward_with_differentiable_mask(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Patched LlamaAttention forward with differentiable sentence-level mask.

    Expects the following attributes on `self`:
        self._node_mask: (num_heads, S, S) tensor with requires_grad=True
        self._token_to_sentence: (seq_len,) LongTensor
        self._maskable: (S, S) BoolTensor
    """
    bsz, q_len, _ = hidden_states.size()

    # --- Standard Q, K, V projections ---
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

        query_states = torch.cat(
            [nn.functional.linear(hidden_states, s) for s in query_slices], dim=-1
        )
        key_states = torch.cat(
            [nn.functional.linear(hidden_states, s) for s in key_slices], dim=-1
        )
        value_states = torch.cat(
            [nn.functional.linear(hidden_states, s) for s in value_slices], dim=-1
        )
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- KV cache ---
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # --- GQA repeat ---
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # --- Attention scores ---
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask
        if attention_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # --- Softmax (upcast to fp32) ---
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)

    # =====================================================================
    # DIFFERENTIABLE MASK APPLICATION
    # =====================================================================
    node_mask = getattr(self, "_node_mask", None)
    token_to_sent = getattr(self, "_token_to_sentence", None)
    maskable = getattr(self, "_maskable", None)

    if node_mask is not None and token_to_sent is not None:
        k_len = key_states.shape[-2]

        # Apply gap constraint: force non-maskable edges to 1.0
        effective_mask = node_mask.clone()
        if maskable is not None:
            # maskable: (S, S) bool. Where False, force mask to 1.0
            # Expand to (1, S, S) then broadcast with (H, S, S)
            effective_mask = torch.where(
                maskable.unsqueeze(0), effective_mask, torch.ones_like(effective_mask)
            )

        # Expand sentence-level mask to token-level
        token_mask = expand_sentence_mask_to_tokens(
            effective_mask, token_to_sent, q_len, k_len
        )
        # token_mask: (1, num_heads, q_len, k_len)

        attn_weights = attn_weights * token_mask

        # Renormalize
        row_sums = attn_weights.sum(dim=-1, keepdim=True) + 1e-12
        attn_weights = attn_weights / row_sums
    # =====================================================================

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

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
            nn.functional.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, past_key_values


# =========================================================================
# Handle for cleanup
# =========================================================================


class DifferentiableMaskHandle:
    """Handle to restore original forward and clean up mask attributes."""

    _ATTRS = ("_node_mask", "_token_to_sentence", "_maskable")

    def __init__(self, attn_module, original_forward):
        self.attn_module = attn_module
        self.original_forward = original_forward

    def remove(self):
        self.attn_module.forward = self.original_forward
        for attr in self._ATTRS:
            if hasattr(self.attn_module, attr):
                delattr(self.attn_module, attr)


# =========================================================================
# Nodewise Attribution Discovery
# =========================================================================


class NodewiseAttributionDiscovery(CircuitDiscoveryAlgorithm):
    """Nodewise attribution patching for attention circuits.

    Computes per-head, per-layer sentence-to-sentence attributions via a
    single forward+backward pass with differentiable attention masks applied
    to ALL analyzed layers simultaneously.
    """

    @property
    def name(self) -> str:
        return "nodewise_attribution"

    def discover(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prefix_token_ids: List[int],
        branch_token_ids: List[List[int]],
        sentences: List[Sentence],
        layers: List[int],
        analysis_timestep: int,
        sentence_gap: int = 1,
        sentence_chunk: int = 1,
        objective_name: str = "kl_divergence",
        batch_size: int = 4,
        **kwargs,
    ) -> NodeMask:
        """Run nodewise attribution patching.

        All analyzed layers are patched simultaneously. Branches are processed
        in sub-batches to manage GPU memory, with gradients accumulated across
        batches.

        Args:
            model: HF model loaded with eager attention.
            tokenizer: Corresponding tokenizer.
            prefix_token_ids: Prefix tokens (prompt up to analysis timestep).
            branch_token_ids: List of N branch continuations (token id lists).
            sentences: Sentence boundaries over the prefix.
            layers: Which layers to analyze.
            analysis_timestep: Token position where analysis starts.
            sentence_gap: Min sentence distance for maskable edges.
            sentence_chunk: Group N consecutive sentences into one chunk.
            objective_name: Loss function name.
            batch_size: Number of branches per sub-batch for memory management.

        Returns:
            NodeMask with per-head (num_heads, S, S) attribution per layer.
        """
        device = next(model.parameters()).device
        objective_fn = get_objective(objective_name)

        # --- Chunk sentences ---
        if sentence_chunk > 1:
            sentences = self._chunk_sentences(sentences, sentence_chunk)
        num_sentences = len(sentences)

        # --- Build full sequences ---
        all_sequences = [prefix_token_ids + branch for branch in branch_token_ids]
        max_len = max(len(seq) for seq in all_sequences)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0

        padded_ids = []
        attn_masks = []
        for seq in all_sequences:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_token_id] * pad_len)
            attn_masks.append([1] * len(seq) + [0] * pad_len)

        # --- Shared structures ---
        token_to_sent = build_token_to_sentence_map(max_len, sentences).to(device)
        maskable = build_maskable_matrix(num_sentences, sentence_gap).to(device)

        # Positions for loss computation (after analysis timestep)
        loss_positions = torch.arange(analysis_timestep, max_len, device=device)

        # Determine num_heads from model config
        num_heads = model.config.num_attention_heads

        # --- Create masks for ALL analyzed layers (requires_grad) ---
        masks: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            masks[layer_idx] = torch.ones(
                num_heads,
                num_sentences,
                num_sentences,
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )

        # --- Patch ALL analyzed layers simultaneously ---
        handles = []
        for layer_idx in layers:
            attn_module = get_attention_module(model, layer_idx)
            original_forward = attn_module.forward

            attn_module._node_mask = masks[layer_idx]
            attn_module._token_to_sentence = token_to_sent
            attn_module._maskable = maskable

            attn_module.forward = types.MethodType(
                llama_attention_forward_with_differentiable_mask, attn_module
            )
            handles.append(DifferentiableMaskHandle(attn_module, original_forward))

        try:
            # --- Process branches in sub-batches ---
            num_branches = len(all_sequences)
            total_loss = torch.tensor(0.0, device=device)

            for batch_start in range(0, num_branches, batch_size):
                batch_end = min(batch_start + batch_size, num_branches)
                batch_ids = torch.tensor(
                    padded_ids[batch_start:batch_end], device=device
                )
                batch_attn = torch.tensor(
                    attn_masks[batch_start:batch_end], device=device
                )

                # Clean forward (no mask effect since mask=1, but we need reference)
                with torch.no_grad():
                    clean_out = model(
                        input_ids=batch_ids, attention_mask=batch_attn
                    )
                    clean_logits = clean_out.logits.detach()

                # Masked forward (masks are all 1.0, so output is same as clean,
                # but gradients tell us the sensitivity to each mask element)
                masked_out = model(
                    input_ids=batch_ids, attention_mask=batch_attn
                )
                masked_logits = masked_out.logits

                loss = objective_fn(clean_logits, masked_logits, positions=loss_positions)
                loss.backward()

                total_loss = total_loss + loss.detach()

            # --- Extract attributions ---
            layer_scores: Dict[int, torch.Tensor] = {}
            for layer_idx in layers:
                grad = masks[layer_idx].grad
                if grad is not None:
                    layer_scores[layer_idx] = grad.abs().detach().cpu()
                else:
                    layer_scores[layer_idx] = torch.zeros(
                        num_heads, num_sentences, num_sentences
                    )

        finally:
            # --- Clean up: unpatch all layers ---
            for handle in handles:
                handle.remove()
            model.zero_grad()
            for mask in masks.values():
                if mask.grad is not None:
                    mask.grad = None

        # --- Build sentence texts ---
        sentence_texts = []
        for s in sentences:
            tokens = prefix_token_ids[s.start : s.end + 1]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            sentence_texts.append(text.strip())

        return NodeMask(
            scores=layer_scores,
            sentences=sentences,
            sentence_texts=sentence_texts,
            metadata={
                "algorithm": self.name,
                "objective": objective_name,
                "analysis_timestep": analysis_timestep,
                "sentence_gap": sentence_gap,
                "sentence_chunk": sentence_chunk,
                "layers": layers,
                "num_heads": num_heads,
                "num_branches": len(branch_token_ids),
                "total_loss": total_loss.item(),
            },
        )

    @staticmethod
    def _chunk_sentences(
        sentences: List[Sentence], chunk_size: int
    ) -> List[Sentence]:
        """Group consecutive sentences into chunks."""
        chunked = []
        for i in range(0, len(sentences), chunk_size):
            group = sentences[i : i + chunk_size]
            chunked.append(Sentence(start=group[0].start, end=group[-1].end))
        return chunked
