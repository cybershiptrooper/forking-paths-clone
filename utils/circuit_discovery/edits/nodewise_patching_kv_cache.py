"""Nodewise activation patching for circuit discovery.

For each sentence pair at the chosen granularity, zeroes out that single
attention edge, runs a forward pass (no gradients), and records the KL
divergence as the importance score.  Higher score = more important edge.
"""

from typing import List, Optional

import torch
from tqdm import tqdm

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
)
from utils.utils import Sentence
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask


class NodewiseActivationPatchingKVCache(CircuitDiscovery):
    """Nodewise activation patching (leave-one-out ablation scanning).

    For each (layer, head, src_sentence, tgt_sentence) edge — at the chosen
    granularity — zero out that single attention connection and measure the
    resulting KL divergence vs clean logits.

    Scores are always stored as +KL(ablated): positive = important edge
    (ablating it hurts the model's output).  No negate flag.
    """

    def __init__(self, **kwargs):
        # Pop kwargs that the factory may forward but we don't use.
        kwargs.pop("num_ig_steps", None)
        kwargs.pop("pair_aggregation", None)
        kwargs.pop("negate_scores", None)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_prefix_kv_cache(self, prefix_ids: torch.Tensor, device: torch.device):
        """Run the prefix (all but last token) through the patched model → DynamicCache.

        The cache stores K/V states for positions 0..prefix_len-2 under the
        current ablation mask.  The continuation forward (starting from the
        last prefix token) is then much cheaper than re-running the full
        (prefix + continuation) sequence for every probe.
        """
        from transformers.cache_utils import DynamicCache
        kv = DynamicCache()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            self.model(prefix_ids.to(device), past_key_values=kv, use_cache=True)
        return kv

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
    ) -> float:
        """Batch all continuations into a forward pass and return mean objective.

        All continuations share the same ablation mask for a given probe, so batching
        is valid — the hook broadcasts the sentence mask over the batch dimension.
        Continuations of unequal length are right-padded; an explicit attention_mask
        prevents real tokens from attending to padding positions.

        Args:
            prefix_kv_cache: Optional DynamicCache from _get_prefix_kv_cache.
                When provided, the forward runs only over the continuation tokens
                (starting from the last prefix token), skipping the O(prefix_len²)
                prefix attention recomputation.  Pass None to fall back to the full
                sequence forward.
        """
        cont_lens = [c.shape[-1] for c in continuations]
        max_cont_len = max(cont_lens)
        batch_size = len(continuations)

        if prefix_kv_cache is not None:
            return self._compute_mean_kl_cached(
                input_ids, continuations, clean_logits_list,
                prefix_len, device, cont_lens, max_cont_len, batch_size,
                branch_rewards, position_mask_overrides, prefix_kv_cache,
            )

        # --- Full-sequence batched forward (no KV cache) ---
        full_len = prefix_len + max_cont_len
        batch_input = torch.zeros(batch_size, full_len, dtype=input_ids.dtype, device=device)
        attn_mask = torch.zeros(batch_size, full_len, device=device)
        for i, (cont, clen) in enumerate(zip(continuations, cont_lens)):
            actual = prefix_len + clen
            batch_input[i, :prefix_len] = input_ids[0]
            batch_input[i, prefix_len:actual] = cont[0]
            attn_mask[i, :actual] = 1.0

        with torch.amp.autocast("cuda"):
            logits_batch = self.model(batch_input, attention_mask=attn_mask).logits  # (B, full_len, V)

        obj_sum = 0.0
        for cont_idx, (clen, clean_logits) in enumerate(zip(cont_lens, clean_logits_list)):
            actual = prefix_len + clen
            logits_i = logits_batch[cont_idx : cont_idx + 1, :actual]
            clean_logits_i = clean_logits[:, :actual].to(device)
            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                position_mask = position_mask_overrides[cont_idx].to(device)
            else:
                position_mask = self._build_position_mask(actual, prefix_len, device)
            obj = self.objective_fn(
                clean_logits_i, logits_i.float(), position_mask,
                token_ids=batch_input[cont_idx : cont_idx + 1, :actual],
            )
            weight = branch_rewards[cont_idx] if branch_rewards is not None else 1.0
            obj_sum += obj.item() * weight
        return obj_sum / batch_size

    def _compute_mean_kl_cached(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        clean_logits_list: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        cont_lens: List[int],
        max_cont_len: int,
        batch_size: int,
        branch_rewards: Optional[List[float]],
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]],
        prefix_kv_cache,
    ) -> float:
        """Forward only the continuation tokens using a pre-computed prefix KV cache.

        Continuations are processed one at a time (B=1) to avoid the OOM that
        arises from expanding the prefix cache to B=N_conts and running a batched
        forward (each DynamicLayer.update() would materialise a
        (N_conts, n_kv_heads, prefix_len+cont_len, head_dim) tensor per layer).

        Instead we let DynamicLayer.update() grow the cache in-place during the
        forward pass, then **slice it back** to the prefix length afterwards —
        a zero-copy reset that avoids deepcopy overhead.

        Continuation input per forward: [p_{L-1}, c_0, ..., c_{M-1}]
        (last prefix token + continuation, length = 1 + clen).  Logit at local
        position k predicts global token at prefix_len-1+k, so local positions
        0..clen-1 cover the entire continuation window.
        """
        prefix_seq_len = prefix_len - 1  # tokens currently stored in the cache
        last_prefix_tok = input_ids[0, -1]  # scalar

        obj_sum = 0.0
        for cont_idx, (cont, clen, clean_logits) in enumerate(
            zip(continuations, cont_lens, clean_logits_list)
        ):
            # Build [p_{L-1}, c_0, ..., c_{clen-1}] (exact length, no padding)
            cont_input = torch.empty(1, 1 + clen, dtype=input_ids.dtype, device=device)
            cont_input[0, 0] = last_prefix_tok
            cont_input[0, 1:] = cont[0, :clen]

            # Attention mask: cached prefix positions + new tokens (all real, no padding)
            attn_mask = torch.ones(1, prefix_len + clen, device=device)

            # cache_position: global indices of the new tokens
            cache_position = torch.arange(prefix_len - 1, prefix_len + clen, device=device)

            with torch.amp.autocast("cuda"):
                logits_i = self.model(
                    cont_input,
                    attention_mask=attn_mask,
                    past_key_values=prefix_kv_cache,
                    cache_position=cache_position,
                    use_cache=False,
                ).logits  # (1, 1+clen, V)

            # Slice cache back to prefix length — undoes DynamicLayer.update() in-place growth
            for layer in prefix_kv_cache.layers:
                layer.keys = layer.keys[:, :, :prefix_seq_len, :]
                layer.values = layer.values[:, :, :prefix_seq_len, :]

            # Clean logits window: global positions prefix_len-1..prefix_len+clen-1 (length 1+clen)
            clean_logits_i = clean_logits[:, prefix_len - 1 : prefix_len + clen].to(device)

            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                pm_full = position_mask_overrides[cont_idx].to(device)
                position_mask = pm_full[:, prefix_len - 1 : prefix_len + clen]
            else:
                # Active for predictions 0..clen-1; exclude position clen (past continuation end)
                position_mask = torch.ones(1, 1 + clen, device=device)
                position_mask[0, clen] = 0.0

            obj = self.objective_fn(
                clean_logits_i, logits_i.float(), position_mask,
                token_ids=cont_input,
            )
            weight = branch_rewards[cont_idx] if branch_rewards is not None else 1.0
            obj_sum += obj.item() * weight
        return obj_sum / batch_size

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
        """Run activation patching circuit discovery.

        Args:
            input_ids: (1, prompt_len) tokenized prompt
            sentences: Sentence boundaries (prefix + optional generation)
            continuations: List of (1, cont_len) continuation token tensors
            mask_mode: "prefix", "generation", or "both"
            num_prefix_sentences: Number of prefix sentences in *sentences*.
            branch_rewards: Optional per-branch scalar rewards.
            position_mask_overrides: Optional per-branch position masks.

        Returns:
            NodeMask with importance scores per edge.
        """
        device = next(self.model.parameters()).device
        num_sents = len(sentences)
        num_heads = self.model.config.num_attention_heads
        prefix_len = input_ids.shape[-1]
        num_prefix_sents = (
            num_prefix_sentences if num_prefix_sentences is not None else num_sents
        )
        granularity = self.mask_granularity

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

        # Active (learnable) pairs — positions NOT frozen by the combined filter
        active_pairs = torch.nonzero(~combined_filter, as_tuple=False)  # (N, 2)
        num_active = active_pairs.shape[0]

        # Total probes for progress reporting
        if granularity == "head":
            total_probes = len(self.layers) * num_heads * num_active
        elif granularity == "layer":
            total_probes = len(self.layers) * num_active
        else:  # "pair"
            total_probes = num_active

        print(
            f"Running activation patching "
            f"({total_probes} probes x {len(continuations)} continuations, "
            f"granularity={granularity})..."
        )

        # ----- Non-target layer ablation (optional) -----
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

        # ----- Patch target layers with all-ones masks -----
        forward_fn = make_attention_forward(self.model_type, apply_sentence_mask)
        masks = {
            l: torch.ones(num_heads, num_sents, num_sents, device=device)
            for l in self.layers
        }
        handles = self._patch_model(
            masks, token_to_sent, combined_filter, forward_fn
        )

        # ----- Iterate over edges (no gradients) -----
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

        # Pre-compute prefix ids for KV caching (all but the last prefix token).
        # The cache captures K/V states for positions 0..prefix_len-2 under the
        # current ablation mask; the continuation forward starts from p_{L-1}.
        prefix_ids_for_cache = input_ids[:, :-1]

        with torch.no_grad():
            if granularity == "pair":
                for pair_idx in tqdm(
                    range(num_active), desc="Activation patching (pair)"
                ):
                    i, j = active_pairs[pair_idx].tolist()
                    # Ablate this pair across all layers
                    for l in self.layers:
                        masks[l][:, i, j] = 0.0
                    prefix_kv = self._get_prefix_kv_cache(prefix_ids_for_cache, device)
                    mean_kl = self._compute_mean_kl(
                        input_ids, continuations, clean_logits_list,
                        prefix_len, device,
                        branch_rewards=branch_rewards,
                        position_mask_overrides=position_mask_overrides,
                        prefix_kv_cache=prefix_kv,
                    )
                    accumulated[i, j] = mean_kl
                    # Restore
                    for l in self.layers:
                        masks[l][:, i, j] = 1.0

            elif granularity == "layer":
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (layer)"
                )
                for l in self.layers:
                    for pair_idx in range(num_active):
                        i, j = active_pairs[pair_idx].tolist()
                        masks[l][:, i, j] = 0.0
                        prefix_kv = self._get_prefix_kv_cache(prefix_ids_for_cache, device)
                        mean_kl = self._compute_mean_kl(
                            input_ids, continuations, clean_logits_list,
                            prefix_len, device,
                            prefix_kv_cache=prefix_kv,
                        )
                        accumulated[l][i, j] = mean_kl
                        masks[l][:, i, j] = 1.0
                        pbar.update(1)
                pbar.close()

            else:  # "head"
                pbar = tqdm(
                    total=total_probes, desc="Activation patching (head)"
                )
                for l in self.layers:
                    for h in range(num_heads):
                        for pair_idx in range(num_active):
                            i, j = active_pairs[pair_idx].tolist()
                            masks[l][h, i, j] = 0.0
                            prefix_kv = self._get_prefix_kv_cache(prefix_ids_for_cache, device)
                            mean_kl = self._compute_mean_kl(
                                input_ids, continuations, clean_logits_list,
                                prefix_len, device,
                                prefix_kv_cache=prefix_kv,
                            )
                            accumulated[l][h, i, j] = mean_kl
                            masks[l][h, i, j] = 1.0
                            pbar.update(1)
                pbar.close()

        # ----- Cleanup patches -----
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
            },
            scores=scores,
        )
