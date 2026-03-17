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
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask


class NodewiseActivationPatching(CircuitDiscovery):
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

    def _compute_mean_kl(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        clean_logits_list: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        branch_rewards: Optional[List[float]] = None,
        position_mask_overrides: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> float:
        """Forward all continuations through the (patched) model, return mean objective."""
        obj_sum = 0.0
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            position_mask = self._build_position_mask(full_len, prefix_len, device)
            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                position_mask = position_mask_overrides[cont_idx].to(device)
            clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits

            obj = self.objective_fn(
                clean_logits, logits.float(), position_mask, token_ids=full_input
            )
            weight = branch_rewards[cont_idx] if branch_rewards is not None else 1.0
            obj_sum += obj.item() * weight
        return obj_sum / len(continuations)

    def _compute_global_metric(
        self,
        input_ids: torch.Tensor,
        continuations: List[torch.Tensor],
        prefix_len: int,
        device: torch.device,
        chain_logprobs_clean: torch.Tensor,
        answer_ids: torch.Tensor,
        num_answers: int,
    ) -> float:
        """Forward all chains, compute global IS-based objective metric."""
        chain_lps = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits
            lp = chain_log_prob(logits.float(), full_input, prefix_len)
            chain_lps.append(lp.detach())
        chain_lps = torch.stack(chain_lps).to(device)
        return self.objective_fn(
            chain_lps, chain_logprobs_clean, answer_ids, num_answers
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

        # Determine if we're using a global objective
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

        # For global objectives: compute clean chain logprobs (proposal)
        chain_logprobs_clean = None
        if use_global:
            chain_logprobs_clean = []
            for ci, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
                lp = chain_log_prob(clean_logits, full_input.cpu(), prefix_len)
                chain_logprobs_clean.append(lp.detach())
            chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

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

        # Helper to compute the metric for the current mask configuration
        def _compute_metric() -> float:
            if use_global:
                return self._compute_global_metric(
                    input_ids, continuations, prefix_len, device,
                    chain_logprobs_clean, answer_ids.to(device), num_answers,
                )
            else:
                return self._compute_mean_kl(
                    input_ids, continuations, clean_logits_list,
                    prefix_len, device,
                    branch_rewards=branch_rewards,
                    position_mask_overrides=position_mask_overrides,
                )

        with torch.no_grad():
            if granularity == "pair":
                for pair_idx in tqdm(
                    range(num_active), desc="Activation patching (pair)"
                ):
                    i, j = active_pairs[pair_idx].tolist()
                    # Ablate this pair across all layers
                    for l in self.layers:
                        masks[l][:, i, j] = 0.0
                    accumulated[i, j] = _compute_metric()
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
                        accumulated[l][i, j] = _compute_metric()
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
                            accumulated[l][h, i, j] = _compute_metric()
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
