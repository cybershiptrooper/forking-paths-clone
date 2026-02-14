"""Nodewise attribution via integrated gradients for circuit discovery.

Each attention head is treated as a separate node. Uses integrated gradients
to compute per-(layer, head, source_sentence, target_sentence) attribution
scores measuring how important each attention edge is for the model's output.
"""

import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from tqdm import tqdm

from utils.masks import NodeMask, build_gap_filter, apply_gap_filter
from utils.utils import Sentence
from utils.circuit_discovery.base import CircuitDiscovery


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
        total_len = token_to_sent.shape[0]
        q_sent = token_to_sent[total_len - q_len : total_len]  # (q_len,)
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


def llama_attention_forward_with_differentiable_mask(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Patched LlamaAttention forward with per-head differentiable sentence mask.

    Applies a continuous [0,1] mask per (head, src_sentence, tgt_sentence)
    to the post-softmax attention weights. The mask is differentiable for
    gradient-based circuit discovery.
    """
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

    # =========================================================================
    # [INJECTION] Per-head differentiable sentence mask
    # =========================================================================
    mask = getattr(self, "_circuit_mask", None)
    token_to_sent = getattr(self, "_token_to_sent", None)
    gap_filter = getattr(self, "_gap_filter", None)

    if mask is not None and token_to_sent is not None:
        k_len = key_states.shape[-2]
        original_dtype = attn_weights.dtype
        token_mask = expand_sentence_mask_to_tokens(
            mask, token_to_sent, gap_filter, q_len, k_len, cache_position
        )
        # token_mask: (num_heads, q_len, k_len)
        # attn_weights: (bsz, num_heads, q_len, k_len)
        # Compute in float32 for numerical stability, then cast back
        attn_weights = (attn_weights.float() * token_mask.unsqueeze(0))
        # Renormalize
        row_sums = attn_weights.sum(dim=-1, keepdim=True) + 1e-12
        attn_weights = (attn_weights / row_sums).to(original_dtype)
    # =========================================================================

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


class NodewiseAttribution(CircuitDiscovery):
    """Nodewise attribution using integrated gradients.

    Treats each attention head as a separate node. Computes the attribution
    of each (layer, head, src_sentence, tgt_sentence) edge to the objective
    using integrated gradients along the path from fully ablated (mask=0)
    to fully present (mask=1).

    Sign convention:
    - Raw IG gradients are w.r.t. the KL objective, so positive means
      increasing KL (worse retention).
    - By default we negate scores so positive means *reducing* KL (helpful).
      Set `negate_scores=False` to preserve raw (harmful-positive) scores.
    """

    def __init__(self, num_ig_steps: int = 10, negate_scores: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_ig_steps = num_ig_steps
        self.negate_scores = negate_scores

    def _get_clean_logits(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Pre-compute clean logits (no mask) for each continuation."""
        clean_logits_list = []
        self.model.eval()
        with torch.no_grad():
            for cont in continuations:
                full_input = torch.cat([input_ids, cont], dim=-1)
                outputs = self.model(full_input)
                clean_logits_list.append(outputs.logits.detach().cpu())
        return clean_logits_list

    def _build_position_mask(
        self, full_len: int, prefix_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build mask with 1 for continuation tokens, 0 for prefix."""
        mask = torch.zeros(1, full_len, device=device)
        # We care about logits that predict continuation tokens
        # logits at position i predict token i+1
        # So for continuation starting at prefix_len, we want positions prefix_len-1 onwards
        # But the first useful prediction is at prefix_len-1 (predicting token at prefix_len)
        mask[0, prefix_len - 1 : full_len - 1] = 1.0
        return mask

    def discover(
        self,
        input_ids: torch.Tensor,
        sentences: List[Sentence],
        continuations: List[torch.Tensor],
        **kwargs,
    ) -> NodeMask:
        """Run integrated gradients circuit discovery.

        Args:
            input_ids: (1, prompt_len) tokenized prompt
            sentences: Sentence boundaries in the prompt
            continuations: List of (1, cont_len) continuation token tensors

        Returns:
            NodeMask with attribution scores per (layer, head, src_sent, tgt_sent)
        """
        device = next(self.model.parameters()).device
        num_sents = len(sentences)
        num_heads = self.model.config.num_attention_heads
        prefix_len = input_ids.shape[-1]

        # Build mappings
        # token_to_sent needs to cover the full sequence (prompt + longest continuation)
        max_cont_len = max(c.shape[-1] for c in continuations)
        total_seq_len = prefix_len + max_cont_len
        token_to_sent = self._build_token_to_sentence_map(sentences, total_seq_len)
        token_to_sent = token_to_sent.to(device)
        gap_filter = build_gap_filter(num_sents, self.sentence_gap, device=device)

        # 0. Optionally ablate all non-target layers
        non_target_handles = []
        if self.ablate_non_target_layers:
            print(
                f"Ablating all layers outside {self.layers} "
                f"({self.model.config.num_hidden_layers - len(self.layers)} layers)..."
            )
            non_target_handles = self._patch_non_target_layers(
                num_heads=num_heads,
                num_sents=num_sents,
                token_to_sent=token_to_sent,
                gap_filter=gap_filter,
                custom_forward_fn=llama_attention_forward_with_differentiable_mask,
            )

        # 1. Pre-compute clean logits (with non-target layers ablated if enabled)
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        # 2. Integrated gradients
        accumulated_grads = {
            l: torch.zeros(num_heads, num_sents, num_sents) for l in self.layers
        }

        print(
            f"Running integrated gradients ({self.num_ig_steps} steps, "
            f"{len(continuations)} continuations)..."
        )
        for step in tqdm(range(1, self.num_ig_steps + 1), desc="IG steps"):
            alpha = step / self.num_ig_steps

            # Create per-layer mask tensors
            masks = {}
            for l in self.layers:
                m = torch.full(
                    (num_heads, num_sents, num_sents),
                    alpha,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                masks[l] = m

            # Patch model
            handles = self._patch_model(
                masks,
                token_to_sent,
                gap_filter,
                llama_attention_forward_with_differentiable_mask,
            )

            # Forward with grad for each continuation
            for cont_idx, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                position_mask = self._build_position_mask(full_len, prefix_len, device)
                clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

                # Zero grads from previous continuation
                for l in self.layers:
                    if masks[l].grad is not None:
                        masks[l].grad.detach_()
                        masks[l].grad.zero_()

                with torch.amp.autocast("cuda"):
                    logits = self.model(full_input).logits

                loss = self.objective_fn(clean_logits, logits, position_mask)
                loss.backward()

                # Accumulate grads
                for l in self.layers:
                    if masks[l].grad is not None:
                        accumulated_grads[l] += masks[l].grad.detach().cpu()

            # Cleanup
            self._unpatch_model(handles)

        # 3. Cleanup non-target layer ablation
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # 4. Average gradients → attribution scores
        # Raw IG scores: positive means including the node increases KL (hurts).
        # Negate (default) so positive means including the node *reduces* KL (helps).
        num_total = self.num_ig_steps * len(continuations)
        sign = -1.0 if self.negate_scores else 1.0
        scores = {}
        for l in self.layers:
            avg = sign * accumulated_grads[l] / num_total
            scores[l] = {h: avg[h].tolist() for h in range(num_heads)}

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_attribution",
            layers=self.layers,
            sentences=[
                {"start": s.start, "end": s.end} for s in sentences
            ],
            objective_name="kl_divergence",
            metadata={
                "num_ig_steps": self.num_ig_steps,
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "negate_scores": self.negate_scores,
            },
            scores=scores,
        )
