"""SDPA-based activation patching with batched continuations.

Replaces eager post-softmax multiplicative masking with SDPA pre-softmax
additive masking.  For binary masks (which activation patching always uses),
these are mathematically equivalent:

  Post-softmax:  softmax(QK^T) * mask,  then ratio-renormalize
  Pre-softmax:   softmax(QK^T + additive_mask)

Setting entries to -inf before softmax makes them 0, and the remaining
entries naturally sum to 1 — no renormalization needed.

Combines three optimizations:
1. **SDPA attention** — no attention weight matrix materialised
2. **KV cache** — prefix computed once per probe
3. **Batched continuations** — all B continuations in one forward pass
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from tqdm import tqdm

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.utils import Sentence, get_attention_module
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import (
    make_attention_forward,
    apply_sentence_mask,
    expand_sentence_mask_to_tokens,
)
# SDPA forward factories live in their own module; this file uses
# make_sdpa_attention_forward inside the patching algorithm below.
from utils.circuit_discovery.sdpa_forward import make_sdpa_attention_forward


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------


def _expand_kv_cache_for_batch(
    kv_cache: DynamicCache, batch_size: int,
) -> DynamicCache:
    """Expand a B=1 ``DynamicCache`` to *batch_size*.

    Creates contiguous copies so ``DynamicCache.update()`` works correctly
    during the batched forward (it concatenates along the seq dimension).

    Uses the ``DynamicCache.layers`` / ``DynamicLayer`` API
    (transformers ≥ 4.48).
    """
    expanded = DynamicCache()
    for layer in kv_cache.layers:
        k = layer.keys.expand(batch_size, -1, -1, -1).contiguous()
        v = layer.values.expand(batch_size, -1, -1, -1).contiguous()
        # update() appends a new DynamicLayer to the cache
        expanded.update(k, v, len(expanded), {})
    return expanded


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NodewiseActivationPatchingFlash(CircuitDiscovery):
    """SDPA activation patching with KV-cache + chunked batched continuations.

    Target layers use ``F.scaled_dot_product_attention`` with a pre-softmax
    additive sentence mask (no attention-weight matrix materialised).
    Non-target layers keep the standard eager forward with zero masks.

    Continuations are processed in chunks of ``batch_chunk_size`` (default 4)
    to avoid OOM on long sequences with many branches.
    """

    # Default chunk size — 4 is safe for 32-branch × 10K-token workloads on
    # 80 GB GPUs. Increase for shorter continuations / fewer branches.
    DEFAULT_BATCH_CHUNK_SIZE = 4

    def __init__(self, **kwargs):
        self.batch_chunk_size = kwargs.pop(
            "batch_chunk_size", self.DEFAULT_BATCH_CHUNK_SIZE,
        )
        kwargs.pop("num_ig_steps", None)
        kwargs.pop("pair_aggregation", None)
        kwargs.pop("negate_scores", None)
        kwargs.pop("include_zero_ablation", None)
        kwargs.pop("zero_ablation_epsilon", None)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    def _get_prefix_kv_cache(
        self, prefix_ids: torch.Tensor, device: torch.device,
    ) -> Tuple[DynamicCache, torch.Tensor]:
        """Forward ALL prefix tokens → (KV cache, last-position logits).

        Processing the full prefix here means the continuation forward only
        has tokens outside any sentence (``token_to_sent == -1``), so the
        SDPA forward skips the expensive sentence mask expansion entirely.
        The returned ``last_logits`` (1, 1, V) cover the prediction at
        ``prefix_len - 1`` (predicting the first continuation token).
        """
        kv = DynamicCache()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = self.model(
                prefix_ids.to(device), past_key_values=kv, use_cache=True,
            )
        last_logits = outputs.logits[:, -1:, :].float().detach()
        return kv, last_logits

    # ------------------------------------------------------------------
    # Clean logits via KV cache (avoids eager OOM on long sequences)
    # ------------------------------------------------------------------

    def _get_clean_logits_cached(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Compute clean logits using KV cache + chunked batching.

        Returns a list of (1, full_len, V) CPU logit tensors, matching the
        base class ``_get_clean_logits`` contract.  Only the continuation
        window (positions ``prefix_len-1`` onward) contains real logits;
        the prefix positions are zero-filled (unused by downstream code).

        Processes ALL prefix tokens in the KV cache so that the continuation
        forward only has tokens with ``token_to_sent == -1``, letting the
        SDPA forward skip the sentence mask entirely (fast path).
        """
        self.model.eval()
        chunk = self.batch_chunk_size

        # Full prefix → KV cache + logits for position prefix_len-1
        prefix_kv, last_prefix_logits = self._get_prefix_kv_cache(
            input_ids, device,
        )
        last_prefix_logits_cpu = last_prefix_logits.cpu()

        clean_logits_list: List[torch.Tensor] = [None] * len(continuations)
        B = len(continuations)

        with torch.no_grad():
            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                chunk_conts = continuations[start:end]
                chunk_size = end - start

                cont_lens = [c.shape[-1] for c in chunk_conts]
                max_cont_len = max(cont_lens)

                # Continuation-only input (no prefix token)
                batch_input = torch.zeros(
                    chunk_size, max_cont_len, dtype=input_ids.dtype, device=device,
                )
                attn_mask = torch.zeros(
                    chunk_size, prefix_len + max_cont_len, device=device,
                )
                for i, (cont, clen) in enumerate(zip(chunk_conts, cont_lens)):
                    batch_input[i, :clen] = cont[0, :clen]
                    attn_mask[i, : prefix_len + clen] = 1.0

                cache_position = torch.arange(
                    prefix_len, prefix_len + max_cont_len, device=device,
                )
                expanded_kv = _expand_kv_cache_for_batch(prefix_kv, chunk_size)

                with torch.amp.autocast("cuda"):
                    logits_batch = self.model(
                        batch_input,
                        attention_mask=attn_mask,
                        past_key_values=expanded_kv,
                        cache_position=cache_position,
                        use_cache=False,
                    ).logits  # (chunk_size, max_cont_len, V)

                del expanded_kv

                for ci, (cont, clen) in enumerate(zip(chunk_conts, cont_lens)):
                    global_ci = start + ci
                    full_len = prefix_len + clen
                    vocab_size = logits_batch.shape[-1]
                    full_logits = torch.zeros(1, full_len, vocab_size)
                    # Position prefix_len-1: from prefix forward
                    full_logits[:, prefix_len - 1 : prefix_len] = (
                        last_prefix_logits_cpu
                    )
                    # Positions prefix_len onward: from continuation forward
                    full_logits[:, prefix_len : prefix_len + clen] = (
                        logits_batch[ci : ci + 1, :clen].float().cpu()
                    )
                    clean_logits_list[global_ci] = full_logits

                del logits_batch

        del prefix_kv
        return clean_logits_list

    # ------------------------------------------------------------------
    # Local objective — batched continuations
    # ------------------------------------------------------------------

    def _compute_mean_kl(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        clean_logits_list: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        branch_rewards: Optional[List[float]] = None,
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]] = None,
        prefix_kv_cache=None,
        prefix_last_logits=None,
    ) -> float:
        if prefix_kv_cache is not None:
            return self._compute_mean_kl_batched(
                input_ids, continuations, clean_logits_list,
                prefix_len, device, branch_rewards, position_mask_overrides,
                prefix_kv_cache, prefix_last_logits,
            )
        # Fallback: sequential full-sequence forwards (no cache)
        return self._compute_mean_kl_sequential(
            input_ids, continuations, clean_logits_list,
            prefix_len, device, branch_rewards, position_mask_overrides,
        )

    def _compute_mean_kl_sequential(
        self,
        input_ids, continuations, clean_logits_list,
        prefix_len, device, branch_rewards, position_mask_overrides,
    ) -> float:
        batch_size = len(continuations)
        obj_sum = 0.0
        for ci, (cont, clean_logits) in enumerate(
            zip(continuations, clean_logits_list)
        ):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            with torch.amp.autocast("cuda"):
                logits_i = self.model(full_input).logits
            clean_logits_i = clean_logits[:, :full_len].to(device)
            if (
                position_mask_overrides is not None
                and position_mask_overrides[ci] is not None
            ):
                pmask = position_mask_overrides[ci].to(device)
            else:
                pmask = self._build_position_mask(full_len, prefix_len, device)
            obj = self.objective_fn(
                clean_logits_i, logits_i.float(), pmask, token_ids=full_input,
            )
            w = branch_rewards[ci] if branch_rewards is not None else 1.0
            obj_sum += obj.item() * w
        return obj_sum / batch_size

    def _compute_mean_kl_batched(
        self,
        input_ids, continuations, clean_logits_list,
        prefix_len, device, branch_rewards, position_mask_overrides,
        prefix_kv_cache, prefix_last_logits,
    ) -> float:
        """Chunked batched forward over continuations using expanded KV cache.

        ``prefix_kv_cache`` covers ALL prefix tokens.  The continuation
        forward only processes actual continuation tokens (all with
        ``token_to_sent == -1``), so the SDPA fast path skips sentence
        mask expansion.  ``prefix_last_logits`` (1, 1, V) provides the
        logit at position ``prefix_len - 1``.
        """
        B = len(continuations)
        chunk = self.batch_chunk_size

        obj_sum = 0.0
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            chunk_conts = continuations[start:end]
            chunk_clean = clean_logits_list[start:end]
            chunk_size = end - start

            cont_lens = [c.shape[-1] for c in chunk_conts]
            max_cont_len = max(cont_lens)

            # Continuation-only input (no prefix token)
            batch_input = torch.zeros(
                chunk_size, max_cont_len, dtype=input_ids.dtype, device=device,
            )
            attn_mask = torch.zeros(
                chunk_size, prefix_len + max_cont_len, device=device,
            )
            for i, (cont, clen) in enumerate(zip(chunk_conts, cont_lens)):
                batch_input[i, :clen] = cont[0, :clen]
                attn_mask[i, : prefix_len + clen] = 1.0

            cache_position = torch.arange(
                prefix_len, prefix_len + max_cont_len, device=device,
            )

            expanded_kv = _expand_kv_cache_for_batch(prefix_kv_cache, chunk_size)

            with torch.amp.autocast("cuda"):
                logits_batch = self.model(
                    batch_input,
                    attention_mask=attn_mask,
                    past_key_values=expanded_kv,
                    cache_position=cache_position,
                    use_cache=False,
                ).logits  # (chunk_size, max_cont_len, V)

            del expanded_kv

            # Extract per-continuation logits and free the batch tensor early
            cont_logits_list = [
                logits_batch[ci : ci + 1, :clen].clone()
                for ci, clen in enumerate(cont_lens)
            ]
            del logits_batch

            for ci, (clen, clean_logits) in enumerate(
                zip(cont_lens, chunk_clean)
            ):
                global_ci = start + ci
                # Stitch prefix-last logit + continuation logits
                logits_i = torch.cat(
                    [prefix_last_logits, cont_logits_list[ci]], dim=1,
                )  # (1, 1+clen, V)
                cont_logits_list[ci] = None
                clean_logits_i = clean_logits[
                    :, prefix_len - 1 : prefix_len + clen
                ].to(device)
                if (
                    position_mask_overrides is not None
                    and position_mask_overrides[global_ci] is not None
                ):
                    pm_full = position_mask_overrides[global_ci].to(device)
                    pmask = pm_full[:, prefix_len - 1 : prefix_len + clen]
                else:
                    pmask = torch.ones(1, 1 + clen, device=device)
                    pmask[0, clen] = 0.0

                cont_input_tokens = continuations[global_ci][0, :clen]
                full_token_window = torch.cat([
                    input_ids[0, -1:], cont_input_tokens,
                ]).unsqueeze(0)
                obj = self.objective_fn(
                    clean_logits_i, logits_i.float(), pmask,
                    token_ids=full_token_window,
                )
                del clean_logits_i, logits_i
                w = branch_rewards[global_ci] if branch_rewards is not None else 1.0
                obj_sum += obj.item() * w

        return obj_sum / B

    # ------------------------------------------------------------------
    # Global objective — batched continuations
    # ------------------------------------------------------------------

    def _compute_global_metric(
        self,
        input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers,
        prefix_kv_cache=None,
        prefix_last_logits=None,
        chain_lengths: Optional[torch.Tensor] = None,
    ) -> float:
        if prefix_kv_cache is not None:
            return self._compute_global_metric_batched(
                input_ids, continuations, prefix_len, device,
                chain_logprobs_clean, answer_ids, num_answers,
                prefix_kv_cache, prefix_last_logits,
                chain_lengths=chain_lengths,
            )
        return self._compute_global_metric_sequential(
            input_ids, continuations, prefix_len, device,
            chain_logprobs_clean, answer_ids, num_answers,
            chain_lengths=chain_lengths,
        )

    def _compute_global_metric_sequential(
        self, input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers,
        chain_lengths: Optional[torch.Tensor] = None,
    ) -> float:
        chain_lps = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits
            lp = chain_log_prob(
                logits.float(), full_input, prefix_len,
                temperature=self.temperature,
            )
            chain_lps.append(lp.detach())
        chain_lps = torch.stack(chain_lps).to(device)
        return self.objective_fn(
            chain_lps, chain_logprobs_clean, answer_ids, num_answers,
            chain_lengths=chain_lengths,
            is_method=self.importance_sampling_method,
            is_temperature=self.importance_sampling_temperature,
        ).item()

    def _compute_global_metric_batched(
        self, input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers,
        prefix_kv_cache, prefix_last_logits,
        chain_lengths: Optional[torch.Tensor] = None,
    ) -> float:
        """Chunked batched forward, then per-branch chain log-prob extraction."""
        B = len(continuations)
        chunk = self.batch_chunk_size

        chain_lps = []
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            chunk_conts = continuations[start:end]
            chunk_size = end - start

            cont_lens = [c.shape[-1] for c in chunk_conts]
            max_cont_len = max(cont_lens)

            batch_input = torch.zeros(
                chunk_size, max_cont_len, dtype=input_ids.dtype, device=device,
            )
            attn_mask = torch.zeros(
                chunk_size, prefix_len + max_cont_len, device=device,
            )
            for i, (cont, clen) in enumerate(zip(chunk_conts, cont_lens)):
                batch_input[i, :clen] = cont[0, :clen]
                attn_mask[i, : prefix_len + clen] = 1.0

            cache_position = torch.arange(
                prefix_len, prefix_len + max_cont_len, device=device,
            )

            expanded_kv = _expand_kv_cache_for_batch(prefix_kv_cache, chunk_size)

            with torch.amp.autocast("cuda"):
                logits_batch = self.model(
                    batch_input,
                    attention_mask=attn_mask,
                    past_key_values=expanded_kv,
                    cache_position=cache_position,
                    use_cache=False,
                ).logits

            del expanded_kv

            for ci, (cont, clen) in enumerate(zip(chunk_conts, cont_lens)):
                # Stitch prefix-last logit + continuation logits
                cont_logits = logits_batch[ci : ci + 1, :clen]
                logits_i = torch.cat(
                    [prefix_last_logits, cont_logits], dim=1,
                )  # (1, 1+clen, V)
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                vocab_size = logits_i.shape[-1]
                full_logits = torch.zeros(
                    1, full_len, vocab_size, device=device,
                )
                full_logits[:, prefix_len - 1 : prefix_len + clen] = logits_i
                lp = chain_log_prob(
                    full_logits.float(), full_input, prefix_len,
                    temperature=self.temperature,
                )
                chain_lps.append(lp.detach())

            del logits_batch

        chain_lps = torch.stack(chain_lps).to(device)
        return self.objective_fn(
            chain_lps, chain_logprobs_clean, answer_ids, num_answers,
            chain_lengths=chain_lengths,
            is_method=self.importance_sampling_method,
            is_temperature=self.importance_sampling_temperature,
        ).item()

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
        num_frozen_prompt_sentences: int = 0,
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
            sentences, total_seq_len,
        ).to(device)

        gap_filter = build_gap_filter(num_sents, self.sentence_gap, device=device)
        mode_filter = build_mode_filter(
            num_prefix_sents, num_sents, mask_mode, device=device,
        )
        causal_filter = build_causal_filter(num_sents, device=device)
        prompt_filter = (
            build_prompt_filter(num_frozen_prompt_sentences, num_sents, device=device)
            if num_frozen_prompt_sentences
            else None
        )
        combined_filter = build_combined_filter(
            gap_filter, mode_filter, causal_filter, prompt_filter
        )
        combined_filter_cpu = combined_filter.cpu()

        active_pairs = torch.nonzero(~combined_filter, as_tuple=False)
        num_active = active_pairs.shape[0]

        if granularity == "head":
            total_probes = len(self.layers) * num_heads * num_active
        elif granularity == "layer":
            total_probes = len(self.layers) * num_active
        else:
            total_probes = num_active

        print(
            f"Running activation patching [flash/SDPA] "
            f"({total_probes} probes x {len(continuations)} continuations, "
            f"granularity={granularity})..."
        )

        # ----- Patch ALL layers with SDPA forward -----
        # Non-target layers get zero masks (fully ablated); target layers
        # get all-ones (unmasked).  Using SDPA everywhere avoids the O(n²)
        # eager attention matrix that OOMs on long sequences.
        sdpa_fwd = make_sdpa_attention_forward(self.model_type)

        non_target_handles = []
        if self.ablate_non_target_layers:
            num_all_layers = self.model.config.num_hidden_layers
            target_set = set(self.layers)
            non_target = [l for l in range(num_all_layers) if l not in target_set]
            print(
                f"Ablating all layers outside {self.layers} "
                f"({len(non_target)} layers)..."
            )
            import types
            for layer_idx in non_target:
                attn_module = get_attention_module(self.model, layer_idx)
                original_forward = attn_module.forward
                zero_mask = torch.zeros(
                    num_heads, num_sents, num_sents, device=device,
                )
                attn_module._circuit_mask = zero_mask
                attn_module._token_to_sent = token_to_sent
                attn_module._gap_filter = combined_filter
                attn_module._renormalize_masked_attn = self.renormalize_masked_attention
                attn_module.forward = types.MethodType(sdpa_fwd, attn_module)
                from utils.circuit_discovery.base import AblationHandle
                non_target_handles.append(AblationHandle(attn_module, original_forward))

        # Target layers: SDPA with all-ones masks (= no masking)
        masks = {
            l: torch.ones(num_heads, num_sents, num_sents, device=device)
            for l in self.layers
        }
        handles = self._patch_model(
            masks, token_to_sent, combined_filter, sdpa_fwd,
        )

        # ----- Clean logits via KV cache (memory-efficient) -----
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits_cached(
            input_ids, continuations, prefix_len, device,
        )

        # Clean chain logprobs for global objectives
        chain_logprobs_clean = None
        if use_global:
            chain_logprobs_clean = []
            for ci, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
                lp = chain_log_prob(
                    clean_logits, full_input.cpu(), prefix_len,
                    temperature=self.temperature,
                )
                chain_logprobs_clean.append(lp.detach())
            chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

        # Per-chain continuation lengths — used by non-SNIS IS methods.
        chain_lengths = torch.tensor(
            [c.shape[-1] for c in continuations],
            dtype=torch.long, device=device,
        )

        # ----- Probe loop (no gradients) -----
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
        else:
            accumulated = torch.zeros(num_sents, num_sents)

        def _compute_metric(prefix_kv, prefix_last_logits) -> float:
            if use_global:
                return self._compute_global_metric(
                    input_ids, continuations, prefix_len, device,
                    chain_logprobs_clean, answer_ids.to(device), num_answers,
                    prefix_kv_cache=prefix_kv,
                    prefix_last_logits=prefix_last_logits,
                    chain_lengths=chain_lengths,
                )
            return self._compute_mean_kl(
                input_ids, continuations, clean_logits_list,
                prefix_len, device,
                branch_rewards=branch_rewards,
                position_mask_overrides=position_mask_overrides,
                prefix_kv_cache=prefix_kv,
                prefix_last_logits=prefix_last_logits,
            )

        with torch.no_grad():
            if granularity == "pair":
                for pair_idx in tqdm(
                    range(num_active), desc="Activation patching (pair)",
                ):
                    i, j = active_pairs[pair_idx].tolist()
                    for l in self.layers:
                        masks[l][:, i, j] = 0.0
                    prefix_kv, prefix_last_logits = self._get_prefix_kv_cache(
                        input_ids, device,
                    )
                    accumulated[i, j] = _compute_metric(
                        prefix_kv, prefix_last_logits,
                    )
                    for l in self.layers:
                        masks[l][:, i, j] = 1.0

            elif granularity == "layer":
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (layer)",
                )
                for l in self.layers:
                    for pair_idx in range(num_active):
                        i, j = active_pairs[pair_idx].tolist()
                        masks[l][:, i, j] = 0.0
                        prefix_kv, prefix_last_logits = self._get_prefix_kv_cache(
                            input_ids, device,
                        )
                        accumulated[l][i, j] = _compute_metric(
                            prefix_kv, prefix_last_logits,
                        )
                        masks[l][:, i, j] = 1.0
                        pbar.update(1)
                pbar.close()

            else:  # head
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (head)",
                )
                for l in self.layers:
                    for h in range(num_heads):
                        for pair_idx in range(num_active):
                            i, j = active_pairs[pair_idx].tolist()
                            masks[l][h, i, j] = 0.0
                            prefix_kv, prefix_last_logits = self._get_prefix_kv_cache(
                                input_ids, device,
                            )
                            accumulated[l][h, i, j] = _compute_metric(
                                prefix_kv, prefix_last_logits,
                            )
                            masks[l][h, i, j] = 1.0
                            pbar.update(1)
                pbar.close()

        # ----- Cleanup (decompile is handled by _unpatch_model) -----
        self._unpatch_model(handles)
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # ----- Convert to NodeMask -----
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
        else:
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
                "num_frozen_prompt_sentences": num_frozen_prompt_sentences,
                "mask_granularity": granularity,
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
                "score_readout": "raw_score",
            },
            scores=scores,
        )
