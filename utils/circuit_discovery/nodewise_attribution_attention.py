"""Nodewise attribution via activation patching + integrated gradients.

Each attention head is treated as a separate node. Uses attribution patching
integrated gradients over attention-pattern activations to compute
per-(layer, head, source_sentence, target_sentence) attribution scores.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask
from utils.masks import (
    NodeMask,
    build_causal_filter,
    build_combined_filter,
    build_gap_filter,
    build_mode_filter,
)
from utils.utils import Sentence, get_attention_module


_ALLOWED_PAIR_AGGREGATIONS = {"sum", "mean", "median", "max"}


def ap_ig_attention_injection(
    module,
    attn_weights: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
) -> torch.Tensor:
    """Injection for activation-patching integrated gradients.

    1. Applies the standard sentence mask (via ``apply_sentence_mask``).
    2. Optionally overrides attention weights entirely with interpolated
       activations (``_attn_override``).
    3. Optionally captures attention weights for later retrieval
       (``_capture_attn``).
    """
    # Step 1: standard sentence mask
    attn_weights = apply_sentence_mask(module, attn_weights, q_len, k_len, cache_position)

    # Step 2: replace with interpolated activations for AP-IG
    override = getattr(module, "_attn_override", None)
    if override is not None:
        attn_weights = override.to(attn_weights.dtype) + (attn_weights * 0)

    # Step 3: snapshot for clean / corrupted capture
    if getattr(module, "_capture_attn", False):
        module._last_attn_weights = attn_weights.detach()

    return attn_weights


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
        forward_fn = make_attention_forward(self.model_type, ap_ig_attention_injection)
        masks = self._make_layer_masks(mask_value, num_heads, num_sents, device)
        handles = self._patch_model(
            masks,
            token_to_sent,
            combined_filter,
            forward_fn,
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
        branch_rewards: Optional[List[float]] = None,
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]] = None,
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
        causal_filter = build_causal_filter(num_sents, device=device)
        combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
        combined_filter_cpu = combined_filter.detach().cpu()

        # Build the patched forward with AP-IG injection
        forward_fn = make_attention_forward(self.model_type, ap_ig_attention_injection)

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
            )

        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        granularity = self.mask_granularity
        if granularity == "head":
            accumulated_scores = {
                layer: torch.zeros(num_heads, num_sents, num_sents, dtype=torch.float32)
                for layer in self.layers
            }
        elif granularity == "layer":
            accumulated_scores = {
                layer: torch.zeros(num_sents, num_sents, dtype=torch.float32)
                for layer in self.layers
            }
        else:  # "pair"
            accumulated_scores = torch.zeros(num_sents, num_sents, dtype=torch.float32)

        print(
            f"Running AP+IG ({self.num_ig_steps} steps, {len(continuations)} continuations, "
            f"aggregation={self.pair_aggregation}, granularity={granularity})..."
        )
        for cont_idx, cont in enumerate(tqdm(continuations, desc="Continuations")):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            position_mask = self._build_position_mask(full_len, prefix_len, device)
            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                position_mask = position_mask_overrides[cont_idx].to(device)
            # Keep clean_logits on CPU; moved to logits' device before objective
            clean_logits_cpu = clean_logits_list[cont_idx][:, :full_len]

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
                    forward_fn,
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

                    # Move clean logits and position mask to logits' device
                    # (handles device_map="auto" where lm_head may be on a different GPU)
                    out_device = logits.device
                    loss = self.objective_fn(
                        clean_logits_cpu.to(out_device), logits.float(),
                        position_mask.to(out_device), token_ids=full_input,
                    )
                    if branch_rewards is not None:
                        loss = loss * branch_rewards[cont_idx]
                    loss.backward()

                    for layer in self.layers:
                        grad = step_overrides[layer].grad
                        if grad is None:
                            continue
                        token_attr = (deltas[layer] * grad.float()).detach()
                        token_attr = token_attr.sum(dim=0).cpu()
                        # aggregated: (H, S, S) — per-head token→sentence scores
                        aggregated = self._aggregate_token_scores(
                            token_attr, q_sent, k_sent, num_sents
                        )
                        if granularity == "head":
                            accumulated_scores[layer] += aggregated
                        elif granularity == "layer":
                            accumulated_scores[layer] += aggregated.sum(dim=0)
                        else:  # "pair"
                            accumulated_scores += aggregated.sum(dim=0)
                finally:
                    self._clear_runtime_attrs()
                    self._unpatch_model(handles)

        if non_target_handles:
            self._unpatch_model(non_target_handles)

        num_total = self.num_ig_steps * len(continuations)
        sign = -1.0 if self.negate_scores else 1.0

        if granularity == "head":
            frozen = combined_filter_cpu.unsqueeze(0).expand(num_heads, -1, -1)
            scores = {}
            for layer in self.layers:
                avg = sign * accumulated_scores[layer] / max(num_total, 1)
                avg[frozen] = 0.0
                scores[layer] = {head: avg[head].tolist() for head in range(num_heads)}
        elif granularity == "layer":
            frozen_2d = combined_filter_cpu.bool()
            scores = {}
            for layer in self.layers:
                avg = sign * accumulated_scores[layer] / max(num_total, 1)
                avg[frozen_2d] = 0.0
                scores[layer] = avg.tolist()
        else:  # "pair"
            frozen_2d = combined_filter_cpu.bool()
            avg = sign * accumulated_scores / max(num_total, 1)
            avg[frozen_2d] = 0.0
            scores = avg.tolist()

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_attribution_attention",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name=getattr(self.objective_fn, "__name__", "unknown"),
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
                "mask_granularity": granularity,
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
            },
            scores=scores,
        )
