"""Nodewise activation patching with batched probe processing.

Instead of processing one probe (sentence-pair ablation) at a time,
batches multiple probes into a single forward pass.  All batch elements
share the same input tokens but have different attention masks — one
per probe — so the GPU processes them in parallel.

Speedup scales with the batch size (limited by GPU memory).
"""

from typing import List, Optional

import torch
from tqdm import tqdm

from utils.masks import (
    NodeMask,
    apply_gap_filter,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
)
from utils.utils import Sentence, get_attention_module
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import (
    make_attention_forward,
    expand_sentence_mask_to_tokens,
)


# ---------------------------------------------------------------------------
# Batched mask expansion and injection
# ---------------------------------------------------------------------------


def _expand_batched_mask(
    mask: torch.Tensor,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Expand mask to token-level, supporting both 3-D and 4-D masks.

    * 3-D ``(H, S, S)`` -> calls the standard ``expand_sentence_mask_to_tokens``.
    * 4-D ``(B, H, S, S)`` -> returns ``(B, H, q_len, k_len)``.
    """
    if mask.dim() == 3:
        return expand_sentence_mask_to_tokens(
            mask, token_to_sent, gap_filter, q_len, k_len,
            cache_position, out_dtype,
        )

    # mask: (B, H, S, S)
    B, H, S, _ = mask.shape
    device = mask.device

    token_to_sent = token_to_sent.to(device)
    if cache_position is not None:
        q_sent = token_to_sent[cache_position.to(device).long()]
    else:
        q_sent = token_to_sent[:q_len]
    k_sent = token_to_sent[:k_len]

    # Pad with sentinel row/col of 1s for tokens not in any sentence
    padded = torch.ones(B, H, S + 1, S + 1, device=device, dtype=mask.dtype)
    effective = apply_gap_filter(mask, gap_filter, fill_value=1.0)
    padded[:, :, :S, :S] = effective

    # Remap -1 -> S (sentinel index)
    q_idx = q_sent.clone().to(device)
    k_idx = k_sent.clone().to(device)
    q_idx[q_sent == -1] = S
    k_idx[k_sent == -1] = S

    if out_dtype is not None and padded.dtype != out_dtype:
        padded = padded.to(out_dtype)

    # Advanced indexing: (B, H, q_len, k_len)
    return padded[:, :, q_idx][:, :, :, k_idx]


def _batched_injection(
    module,
    attn_weights: torch.Tensor,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
) -> torch.Tensor:
    """Post-softmax injection that supports per-batch-element masks.

    When ``_circuit_mask`` is 3-D ``(H, S, S)``, broadcasts over the batch
    (identical behaviour to ``apply_sentence_mask``).  When 4-D
    ``(B, H, S, S)``, applies a different mask per batch element.
    """
    mask = getattr(module, "_circuit_mask", None)
    token_to_sent = getattr(module, "_token_to_sent", None)
    gap_filter = getattr(module, "_gap_filter", None)

    if mask is not None and token_to_sent is not None:
        token_mask = _expand_batched_mask(
            mask, token_to_sent, gap_filter, q_len, k_len, cache_position,
            out_dtype=attn_weights.dtype,
        )
        token_mask = token_mask.to(device=attn_weights.device)

        if token_mask.dim() == 3:
            # (H, q, k) -> broadcast over batch
            attn_weights.mul_(token_mask.unsqueeze(0))
        else:
            # (B, H, q, k) -> per-batch masks
            attn_weights.mul_(token_mask)

        if getattr(module, "_renormalize_masked_attn", True):
            row_sums = attn_weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
            row_sums.add_(1e-16)
            attn_weights.div_(row_sums.to(attn_weights.dtype))

    return attn_weights


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NodewiseActivationPatchingBatchedProbes(CircuitDiscovery):
    """Nodewise activation patching with batched probe processing.

    Processes multiple probes in a single forward pass by using per-batch-
    element attention masks.  Gives a significant speedup by exploiting GPU
    parallelism -- the model weights are loaded once and applied to B inputs
    simultaneously.
    """

    def __init__(self, probe_batch_size: int = 8, **kwargs):
        kwargs.pop("num_ig_steps", None)
        kwargs.pop("pair_aggregation", None)
        kwargs.pop("negate_scores", None)
        super().__init__(**kwargs)
        self.probe_batch_size = probe_batch_size

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _batched_forward(
        self,
        input_ids: torch.Tensor,
        cont: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward B copies of (prefix + continuation) with per-batch masks.

        All B copies share the same tokens; the attention masks on the
        model's attention modules provide per-batch differentiation.

        Returns logits ``(B, seq_len, vocab)``.
        """
        full_input = torch.cat([input_ids, cont], dim=-1)  # (1, seq)
        batch_input = full_input.expand(B, -1)  # (B, seq) — no copy
        with torch.amp.autocast("cuda"):
            return self.model(batch_input).logits

    def _compute_batch_metrics_local(
        self,
        B: int,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        clean_logits_list: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        branch_rewards: Optional[List[float]],
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]],
    ) -> List[float]:
        """Compute mean local objective for B probes across all continuations."""
        obj_sums = [0.0] * B
        for cont_idx, cont in enumerate(continuations):
            logits = self._batched_forward(input_ids, cont, B, device)
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            clean_logits_i = clean_logits_list[cont_idx][:, :full_len].to(device)

            if (
                position_mask_overrides is not None
                and position_mask_overrides[cont_idx] is not None
            ):
                position_mask = position_mask_overrides[cont_idx].to(device)
            else:
                position_mask = self._build_position_mask(full_len, prefix_len, device)

            weight = branch_rewards[cont_idx] if branch_rewards is not None else 1.0
            for b in range(B):
                obj = self.objective_fn(
                    clean_logits_i,
                    logits[b : b + 1].float(),
                    position_mask,
                    token_ids=full_input,
                )
                obj_sums[b] += obj.item() * weight
        return [s / len(continuations) for s in obj_sums]

    def _compute_batch_metrics_global(
        self,
        B: int,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        chain_logprobs_clean: torch.Tensor,
        answer_ids: torch.Tensor,
        num_answers: int,
        chain_lengths: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Compute global IS-based objective for B probes."""
        # Collect chain log-probs per probe across all continuations
        all_chain_lps: List[List[torch.Tensor]] = [[] for _ in range(B)]
        for cont in continuations:
            logits = self._batched_forward(input_ids, cont, B, device)
            full_input = torch.cat([input_ids, cont], dim=-1)
            for b in range(B):
                lp = chain_log_prob(
                    logits[b : b + 1].float(), full_input, prefix_len, temperature=self.temperature
                )
                all_chain_lps[b].append(lp.detach())

        metrics = []
        for b in range(B):
            chain_lps = torch.stack(all_chain_lps[b]).to(device)
            m = self.objective_fn(
                chain_lps, chain_logprobs_clean, answer_ids.to(device), num_answers,
                chain_lengths=chain_lengths,
                is_method=self.importance_sampling_method,
            ).item()
            metrics.append(m)
        return metrics

    def _compute_batch_metrics(
        self,
        B: int,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        clean_logits_list: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        branch_rewards: Optional[List[float]],
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]],
        use_global: bool,
        chain_logprobs_clean: Optional[torch.Tensor],
        answer_ids: Optional[torch.Tensor],
        num_answers: Optional[int],
        chain_lengths: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Dispatch to local or global batch metric computation."""
        if use_global:
            return self._compute_batch_metrics_global(
                B, input_ids, continuations, prefix_len, device,
                chain_logprobs_clean, answer_ids, num_answers,
                chain_lengths=chain_lengths,
            )
        return self._compute_batch_metrics_local(
            B, input_ids, continuations, clean_logits_list, prefix_len,
            device, branch_rewards, position_mask_overrides,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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
        device = next(self.model.parameters()).device
        num_sents = len(sentences)
        num_heads = self.model.config.num_attention_heads
        prefix_len = input_ids.shape[-1]
        num_prefix_sents = (
            num_prefix_sentences if num_prefix_sentences is not None else num_sents
        )
        granularity = self.mask_granularity
        B = self.probe_batch_size

        # Global objective setup
        objective_name = getattr(self.objective_fn, "__name__", "unknown")
        use_global = is_global_objective(objective_name)
        answer_ids = kwargs.get("answer_ids")
        num_answers = kwargs.get("num_answers")
        if use_global and (answer_ids is None or num_answers is None):
            raise ValueError(
                f"Global objective '{objective_name}' requires answer_ids and "
                f"num_answers to be passed to discover()."
            )

        # ----- Build mappings -----
        max_cont_len = max(c.shape[-1] for c in continuations)
        total_seq_len = prefix_len + max_cont_len
        token_to_sent = self._build_token_to_sentence_map(
            sentences, total_seq_len
        ).to(device)

        gap_filter = build_gap_filter(num_sents, self.sentence_gap, device=device)
        mode_filter = build_mode_filter(
            num_prefix_sents, num_sents, mask_mode, device=device
        )
        causal_filter = build_causal_filter(num_sents, device=device)
        combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
        combined_filter_cpu = combined_filter.cpu()

        active_pairs = torch.nonzero(~combined_filter, as_tuple=False)  # (N, 2)
        num_active = active_pairs.shape[0]

        if granularity == "head":
            total_probes = len(self.layers) * num_heads * num_active
        elif granularity == "layer":
            total_probes = len(self.layers) * num_active
        else:  # "pair"
            total_probes = num_active

        print(
            f"Running activation patching "
            f"({total_probes} probes x {len(continuations)} continuations, "
            f"granularity={granularity}, probe_batch={B})..."
        )

        # ----- Non-target layer ablation -----
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

        # ----- Clean logits -----
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        chain_logprobs_clean = None
        if use_global:
            chain_logprobs_clean = []
            for ci, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
                lp = chain_log_prob(clean_logits, full_input.cpu(), prefix_len, temperature=self.temperature)
                chain_logprobs_clean.append(lp.detach())
            chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

        # Per-chain continuation lengths — used by non-SNIS IS methods.
        chain_lengths = torch.tensor(
            [c.shape[-1] for c in continuations],
            dtype=torch.long, device=device,
        )

        # ----- Patch target layers (batched injection) -----
        forward_fn = make_attention_forward(self.model_type, _batched_injection)
        masks = {
            l: torch.ones(num_heads, num_sents, num_sents, device=device)
            for l in self.layers
        }
        handles = self._patch_model(
            masks, token_to_sent, combined_filter, forward_fn
        )

        # ----- Iterate (batched probes, no gradients) -----
        self.model.eval()

        if granularity == "head":
            accumulated = {
                l: torch.zeros(num_heads, num_sents, num_sents)
                for l in self.layers
            }
        elif granularity == "layer":
            accumulated = {
                l: torch.zeros(num_sents, num_sents) for l in self.layers
            }
        else:  # "pair"
            accumulated = torch.zeros(num_sents, num_sents)

        with torch.no_grad():
            if granularity == "pair":
                pbar = tqdm(total=num_active, desc="Activation patching (pair)")
                for batch_start in range(0, num_active, B):
                    batch_end = min(batch_start + B, num_active)
                    cur_B = batch_end - batch_start

                    # Build batched mask: (cur_B, H, S, S)
                    bmask = torch.ones(
                        cur_B, num_heads, num_sents, num_sents, device=device
                    )
                    pair_coords = []
                    for b in range(cur_B):
                        i, j = active_pairs[batch_start + b].tolist()
                        bmask[b, :, i, j] = 0.0
                        pair_coords.append((i, j))

                    # Set batched mask on all target layers
                    for l in self.layers:
                        get_attention_module(self.model, l)._circuit_mask = bmask

                    metrics = self._compute_batch_metrics(
                        cur_B, input_ids, continuations, clean_logits_list,
                        prefix_len, device, branch_rewards,
                        position_mask_overrides, use_global,
                        chain_logprobs_clean, answer_ids, num_answers,
                        chain_lengths=chain_lengths,
                    )

                    for b, (i, j) in enumerate(pair_coords):
                        accumulated[i, j] = metrics[b]

                    pbar.update(cur_B)

                # Restore unbatched masks
                for l in self.layers:
                    get_attention_module(self.model, l)._circuit_mask = masks[l]
                pbar.close()

            elif granularity == "layer":
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (layer)"
                )
                for l in self.layers:
                    for batch_start in range(0, num_active, B):
                        batch_end = min(batch_start + B, num_active)
                        cur_B = batch_end - batch_start

                        bmask = torch.ones(
                            cur_B, num_heads, num_sents, num_sents,
                            device=device,
                        )
                        pair_coords = []
                        for b in range(cur_B):
                            i, j = active_pairs[batch_start + b].tolist()
                            bmask[b, :, i, j] = 0.0
                            pair_coords.append((i, j))

                        # Only this layer gets the batched mask
                        get_attention_module(self.model, l)._circuit_mask = bmask

                        metrics = self._compute_batch_metrics(
                            cur_B, input_ids, continuations,
                            clean_logits_list, prefix_len, device,
                            branch_rewards, position_mask_overrides,
                            use_global, chain_logprobs_clean,
                            answer_ids, num_answers,
                            chain_lengths=chain_lengths,
                        )

                        for b, (i, j) in enumerate(pair_coords):
                            accumulated[l][i, j] = metrics[b]

                        # Restore this layer's mask
                        get_attention_module(
                            self.model, l
                        )._circuit_mask = masks[l]
                        pbar.update(cur_B)
                pbar.close()

            else:  # "head"
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (head)"
                )
                for l in self.layers:
                    for h in range(num_heads):
                        for batch_start in range(0, num_active, B):
                            batch_end = min(batch_start + B, num_active)
                            cur_B = batch_end - batch_start

                            bmask = torch.ones(
                                cur_B, num_heads, num_sents, num_sents,
                                device=device,
                            )
                            pair_coords = []
                            for b in range(cur_B):
                                i, j = active_pairs[batch_start + b].tolist()
                                bmask[b, h, i, j] = 0.0
                                pair_coords.append((i, j))

                            get_attention_module(
                                self.model, l
                            )._circuit_mask = bmask

                            metrics = self._compute_batch_metrics(
                                cur_B, input_ids, continuations,
                                clean_logits_list, prefix_len, device,
                                branch_rewards, position_mask_overrides,
                                use_global, chain_logprobs_clean,
                                answer_ids, num_answers,
                                chain_lengths=chain_lengths,
                            )

                            for b, (i, j) in enumerate(pair_coords):
                                accumulated[l][h, i, j] = metrics[b]

                            get_attention_module(
                                self.model, l
                            )._circuit_mask = masks[l]
                            pbar.update(cur_B)
                pbar.close()

        # ----- Cleanup -----
        self._unpatch_model(handles)
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # ----- Convert to NodeMask scores format -----
        if granularity == "head":
            scores = {}
            for l in self.layers:
                t = accumulated[l]
                t[combined_filter_cpu.unsqueeze(0).expand(num_heads, -1, -1)] = 0.0
                scores[l] = {h: t[h].tolist() for h in range(num_heads)}
        elif granularity == "layer":
            scores = {}
            for l in self.layers:
                t = accumulated[l]
                t[combined_filter_cpu] = 0.0
                scores[l] = t.tolist()
        else:  # "pair"
            accumulated[combined_filter_cpu] = 0.0
            scores = accumulated.tolist()

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_activation_patching",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name=getattr(self.objective_fn, "__name__", "unknown"),
            metadata={
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "mask_granularity": granularity,
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
            },
            scores=scores,
        )
