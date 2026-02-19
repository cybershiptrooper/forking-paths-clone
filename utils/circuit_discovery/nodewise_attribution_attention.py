"""Nodewise attribution via activation patching + integrated gradients.

Each attention head is treated as a separate node. Uses attribution patching
integrated gradients over attention-pattern activations to compute
per-(layer, head, source_sentence, target_sentence) attribution scores.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from tqdm import tqdm

from utils.circuit_discovery.base import CircuitDiscovery
from utils.masks import (
    NodeMask,
    apply_gap_filter,
    build_combined_filter,
    build_gap_filter,
    build_mode_filter,
)
from utils.utils import Sentence, get_attention_module


_ALLOWED_PAIR_AGGREGATIONS = {"sum", "mean", "median", "max"}


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

    if cache_position is not None:
        q_sent = token_to_sent[cache_position.long()]
    else:
        q_sent = token_to_sent[:q_len]
    k_sent = token_to_sent[:k_len]

    padded = torch.ones(
        num_heads, num_sents + 1, num_sents + 1, device=device, dtype=mask.dtype
    )
    effective_mask = apply_gap_filter(mask, gap_filter, fill_value=1.0)
    padded[:, :num_sents, :num_sents] = effective_mask

    q_idx = q_sent.clone().to(device)
    k_idx = k_sent.clone().to(device)
    q_idx[q_sent == -1] = num_sents
    k_idx[k_sent == -1] = num_sents

    token_mask = padded[:, q_idx][:, :, k_idx]
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
    """Patched LlamaAttention forward with masking + AP-IG attention overrides."""
    bsz, q_len, _ = hidden_states.size()

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

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

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

    mask = getattr(self, "_circuit_mask", None)
    token_to_sent = getattr(self, "_token_to_sent", None)
    gap_filter = getattr(self, "_gap_filter", None)

    if mask is not None and token_to_sent is not None:
        k_len = key_states.shape[-2]
        original_dtype = attn_weights.dtype
        token_mask = expand_sentence_mask_to_tokens(
            mask, token_to_sent, gap_filter, q_len, k_len, cache_position
        )
        attn_weights = (attn_weights.float() * token_mask.unsqueeze(0))
        renormalize = getattr(self, "_renormalize_masked_attn", True)
        if renormalize:
            row_sums = attn_weights.sum(dim=-1, keepdim=True) + 1e-12
            attn_weights = (attn_weights / row_sums).to(original_dtype)
        else:
            attn_weights = attn_weights.to(original_dtype)

    override = getattr(self, "_attn_override", None)
    if override is not None:
        attn_weights = override.to(attn_weights.dtype) + (attn_weights * 0)

    if getattr(self, "_capture_attn", False):
        self._last_attn_weights = attn_weights.detach()

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
    """Nodewise attribution using activation patching + integrated gradients."""

    def __init__(self, num_ig_steps: int = 10, negate_scores: bool = True, **kwargs):
        self.pair_aggregation = kwargs.pop("pair_aggregation", "sum")
        if self.pair_aggregation not in _ALLOWED_PAIR_AGGREGATIONS:
            raise ValueError(
                f"pair_aggregation must be one of {_ALLOWED_PAIR_AGGREGATIONS}, "
                f"got {self.pair_aggregation!r}"
            )
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
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for cont in continuations:
                full_input = torch.cat([input_ids, cont], dim=-1)
                outputs = self.model(full_input)
                clean_logits_list.append(outputs.logits.float().detach().cpu())
        return clean_logits_list

    def _build_position_mask(
        self, full_len: int, prefix_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build mask with 1 for continuation tokens, 0 for prefix."""
        mask = torch.zeros(1, full_len, device=device)
        mask[0, prefix_len - 1 : full_len - 1] = 1.0
        return mask

    def _make_layer_masks(
        self,
        value: float,
        num_heads: int,
        num_sents: int,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        return {
            l: torch.full(
                (num_heads, num_sents, num_sents),
                value,
                device=device,
                dtype=torch.float32,
            )
            for l in self.layers
        }

    def _clear_runtime_attrs(self):
        for layer in self.layers:
            attn_module = get_attention_module(self.model, layer)
            for attr in ["_capture_attn", "_last_attn_weights", "_attn_override"]:
                if hasattr(attn_module, attr):
                    delattr(attn_module, attr)

    def _capture_attention_maps(
        self,
        full_input: torch.Tensor,
        token_to_sent: torch.Tensor,
        combined_filter: torch.Tensor,
        mask_value: float,
        num_heads: int,
        num_sents: int,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        masks = self._make_layer_masks(mask_value, num_heads, num_sents, device)
        handles = self._patch_model(
            masks,
            token_to_sent,
            combined_filter,
            llama_attention_forward_with_differentiable_mask,
        )
        try:
            for layer in self.layers:
                attn_module = get_attention_module(self.model, layer)
                attn_module._capture_attn = True
                attn_module._last_attn_weights = None

            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = self.model(full_input)

            captured: Dict[int, torch.Tensor] = {}
            for layer in self.layers:
                attn_module = get_attention_module(self.model, layer)
                layer_attn = getattr(attn_module, "_last_attn_weights", None)
                if layer_attn is None:
                    raise RuntimeError(
                        f"Failed to capture attention weights for layer {layer}."
                    )
                captured[layer] = layer_attn.detach().float()
            return captured
        finally:
            self._clear_runtime_attrs()
            self._unpatch_model(handles)

    def _aggregate_token_scores(
        self,
        token_scores: torch.Tensor,
        q_sent: torch.Tensor,
        k_sent: torch.Tensor,
        num_sents: int,
    ) -> torch.Tensor:
        """Aggregate (head, q_tok, k_tok) into (head, q_sent, k_sent)."""
        num_heads, q_len, k_len = token_scores.shape
        dtype = token_scores.dtype
        device = token_scores.device

        q_sent = q_sent.to(device)
        k_sent = k_sent.to(device)

        q_one_hot = torch.zeros(q_len, num_sents, dtype=dtype, device=device)
        k_one_hot = torch.zeros(k_len, num_sents, dtype=dtype, device=device)

        q_valid = (q_sent >= 0) & (q_sent < num_sents)
        k_valid = (k_sent >= 0) & (k_sent < num_sents)

        if q_valid.any():
            q_one_hot[q_valid] = F.one_hot(
                q_sent[q_valid].long(), num_classes=num_sents
            ).to(dtype)
        if k_valid.any():
            k_one_hot[k_valid] = F.one_hot(
                k_sent[k_valid].long(), num_classes=num_sents
            ).to(dtype)

        if self.pair_aggregation in {"sum", "mean"}:
            summed = torch.einsum("hqk,qi,kj->hij", token_scores, q_one_hot, k_one_hot)
            if self.pair_aggregation == "sum":
                return summed
            counts = torch.einsum("qi,kj->ij", q_one_hot, k_one_hot)
            counts = counts.clamp_min(1.0)
            return summed / counts.unsqueeze(0)

        aggregated = torch.zeros(num_heads, num_sents, num_sents, dtype=dtype, device=device)
        for i in range(num_sents):
            q_idx = torch.nonzero(q_sent == i, as_tuple=False).squeeze(-1)
            if q_idx.numel() == 0:
                continue
            for j in range(num_sents):
                k_idx = torch.nonzero(k_sent == j, as_tuple=False).squeeze(-1)
                if k_idx.numel() == 0:
                    continue
                block = token_scores.index_select(1, q_idx).index_select(2, k_idx)
                flat = block.reshape(num_heads, -1)
                if flat.numel() == 0:
                    continue
                if self.pair_aggregation == "max":
                    aggregated[:, i, j] = flat.max(dim=-1).values
                else:
                    aggregated[:, i, j] = flat.median(dim=-1).values
        return aggregated

    def discover(
        self,
        input_ids: torch.Tensor,
        sentences: List[Sentence],
        continuations: List[torch.Tensor],
        mask_mode: str = "prefix",
        num_prefix_sentences: Optional[int] = None,
        **kwargs,
    ) -> NodeMask:
        """Run activation-patching integrated gradients circuit discovery."""
        device = next(self.model.parameters()).device
        num_sents = len(sentences)
        num_heads = self.model.config.num_attention_heads
        prefix_len = input_ids.shape[-1]
        num_prefix_sents = (
            num_prefix_sentences if num_prefix_sentences is not None else num_sents
        )

        max_cont_len = max(c.shape[-1] for c in continuations)
        total_seq_len = prefix_len + max_cont_len
        token_to_sent = self._build_token_to_sentence_map(sentences, total_seq_len)
        token_to_sent = token_to_sent.to(device)

        gap_filter = build_gap_filter(num_sents, self.sentence_gap, device=device)
        mode_filter = build_mode_filter(
            num_prefix_sents, num_sents, mask_mode, device=device
        )
        combined_filter = build_combined_filter(gap_filter, mode_filter)
        combined_filter_cpu = combined_filter.detach().cpu()

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
                gap_filter=combined_filter,
                custom_forward_fn=llama_attention_forward_with_differentiable_mask,
            )

        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        accumulated_scores = {
            layer: torch.zeros(num_heads, num_sents, num_sents, dtype=torch.float32)
            for layer in self.layers
        }

        print(
            f"Running AP+IG ({self.num_ig_steps} steps, {len(continuations)} continuations, "
            f"aggregation={self.pair_aggregation})..."
        )
        for cont_idx, cont in enumerate(tqdm(continuations, desc="Continuations")):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            position_mask = self._build_position_mask(full_len, prefix_len, device)
            clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

            clean_acts = self._capture_attention_maps(
                full_input,
                token_to_sent,
                combined_filter,
                mask_value=1.0,
                num_heads=num_heads,
                num_sents=num_sents,
                device=device,
            )
            corrupted_acts = self._capture_attention_maps(
                full_input,
                token_to_sent,
                combined_filter,
                mask_value=0.0,
                num_heads=num_heads,
                num_sents=num_sents,
                device=device,
            )

            deltas = {
                layer: clean_acts[layer] - corrupted_acts[layer]
                for layer in self.layers
            }

            q_sent = token_to_sent[:full_len].detach().cpu()
            k_sent = token_to_sent[:full_len].detach().cpu()

            for step in range(1, self.num_ig_steps + 1):
                alpha = step / self.num_ig_steps
                masks = self._make_layer_masks(1.0, num_heads, num_sents, device)
                handles = self._patch_model(
                    masks,
                    token_to_sent,
                    combined_filter,
                    llama_attention_forward_with_differentiable_mask,
                )

                step_overrides: Dict[int, torch.Tensor] = {}
                try:
                    for layer in self.layers:
                        attn_module = get_attention_module(self.model, layer)
                        override = (
                            corrupted_acts[layer] + alpha * deltas[layer]
                        ).detach().requires_grad_(True)
                        attn_module._attn_override = override
                        step_overrides[layer] = override

                    self.model.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda"):
                        logits = self.model(full_input).logits

                    loss = self.objective_fn(clean_logits, logits.float(), position_mask)
                    loss.backward()

                    for layer in self.layers:
                        grad = step_overrides[layer].grad
                        if grad is None:
                            continue
                        token_attr = (deltas[layer] * grad.float()).detach()
                        token_attr = token_attr.sum(dim=0).cpu()
                        aggregated = self._aggregate_token_scores(
                            token_attr, q_sent, k_sent, num_sents
                        )
                        accumulated_scores[layer] += aggregated
                finally:
                    self._clear_runtime_attrs()
                    self._unpatch_model(handles)

        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # Ensure frozen entries are not included in learned scores.
        frozen = combined_filter_cpu.unsqueeze(0).expand(num_heads, -1, -1)

        num_total = self.num_ig_steps * len(continuations)
        sign = -1.0 if self.negate_scores else 1.0
        scores = {}
        for layer in self.layers:
            avg = sign * accumulated_scores[layer] / max(num_total, 1)
            avg[frozen] = 0.0
            scores[layer] = {head: avg[head].tolist() for head in range(num_heads)}

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_attribution_attention",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name="kl_divergence",
            metadata={
                "num_ig_steps": self.num_ig_steps,
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "negate_scores": self.negate_scores,
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "pair_aggregation": self.pair_aggregation,
            },
            scores=scores,
        )
