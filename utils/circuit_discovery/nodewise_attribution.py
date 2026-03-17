"""Nodewise attribution via integrated gradients for circuit discovery.

Each attention head is treated as a separate node. Uses integrated gradients
to compute per-(layer, head, source_sentence, target_sentence) attribution
scores measuring how important each attention edge is for the model's output.
"""

from typing import List, Optional

import torch
from tqdm import tqdm

from utils.masks import NodeMask, build_gap_filter, build_mode_filter, build_combined_filter, build_causal_filter
from utils.utils import Sentence
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import make_attention_forward, apply_sentence_mask


_ALLOWED_PAIR_AGGREGATIONS = {"sum", "mean"}


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
        self.pair_aggregation = kwargs.pop("pair_aggregation", "sum")
        if self.pair_aggregation not in _ALLOWED_PAIR_AGGREGATIONS:
            raise ValueError(
                f"pair_aggregation for mask-IG must be one of {_ALLOWED_PAIR_AGGREGATIONS}, "
                f"got {self.pair_aggregation!r}"
            )
        super().__init__(**kwargs)
        self.num_ig_steps = num_ig_steps
        self.negate_scores = negate_scores

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
        """Run integrated gradients circuit discovery.

        Args:
            input_ids: (1, prompt_len) tokenized prompt
            sentences: Sentence boundaries (prefix + optional generation sentences)
            continuations: List of (1, cont_len) continuation token tensors
            mask_mode: "prefix" (MASK 1), "generation" (MASK 2), or "both"
            num_prefix_sentences: Number of prefix sentences in *sentences*.
                Generation sentences start at this index. Defaults to len(sentences)
                (all are prefix, original behaviour).

        Returns:
            NodeMask with attribution scores per (layer, head, src_sent, tgt_sent)
        """
        device = next(self.model.parameters()).device
        num_sents = len(sentences)
        num_heads = self.model.config.num_attention_heads
        prefix_len = input_ids.shape[-1]
        num_prefix_sents = num_prefix_sentences if num_prefix_sentences is not None else num_sents

        # Build mappings
        # token_to_sent needs to cover the full sequence (prompt + longest continuation)
        max_cont_len = max(c.shape[-1] for c in continuations)
        total_seq_len = prefix_len + max_cont_len
        token_to_sent = self._build_token_to_sentence_map(sentences, total_seq_len)
        token_to_sent = token_to_sent.to(device)
        gap_filter = build_gap_filter(num_sents, self.sentence_gap, device=device)

        # Build combined filter (gap + mode + causal) — True = frozen at 1.0
        mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode, device=device)
        causal_filter = build_causal_filter(num_sents, device=device)
        combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

        # Build the patched forward with sentence-mask injection
        forward_fn = make_attention_forward(self.model_type, apply_sentence_mask)

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
                gap_filter=combined_filter,
            )

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

        # 1. Pre-compute clean logits (with non-target layers ablated if enabled)
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        # For global objectives: also compute clean chain logprobs (proposal)
        chain_logprobs_clean = None
        if use_global:
            chain_logprobs_clean = []
            for ci, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
                lp = chain_log_prob(clean_logits, full_input.cpu(), prefix_len)
                chain_logprobs_clean.append(lp.detach())
            chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

        # 2. Integrated gradients
        granularity = self.mask_granularity
        if granularity == "head":
            accumulated_grads = {
                l: torch.zeros(num_heads, num_sents, num_sents) for l in self.layers
            }
        elif granularity == "layer":
            accumulated_grads = {
                l: torch.zeros(1, num_sents, num_sents) for l in self.layers
            }
        else:  # "pair"
            accumulated_grads = torch.zeros(1, num_sents, num_sents)

        print(
            f"Running integrated gradients ({self.num_ig_steps} steps, "
            f"{len(continuations)} continuations, "
            f"aggregation={self.pair_aggregation}, granularity={granularity})..."
        )
        for step in tqdm(range(1, self.num_ig_steps + 1), desc="IG steps"):
            alpha = step / self.num_ig_steps

            # Create mask tensors at the appropriate granularity.
            # For per-layer/per-pair, we create a smaller learnable tensor
            # and .expand() it to (H, S, S) so expand_sentence_mask_to_tokens
            # receives the shape it expects. Autograd sums gradients over
            # expanded dimensions.
            masks = {}
            raw_masks: dict | torch.Tensor = {}
            if granularity == "head":
                for l in self.layers:
                    m = torch.full(
                        (num_heads, num_sents, num_sents),
                        alpha, device=device, dtype=torch.float32,
                        requires_grad=True,
                    )
                    masks[l] = m
                raw_masks = masks
            elif granularity == "layer":
                raw_masks = {}
                for l in self.layers:
                    m = torch.full(
                        (1, num_sents, num_sents),
                        alpha, device=device, dtype=torch.float32,
                        requires_grad=True,
                    )
                    raw_masks[l] = m
                    masks[l] = m.expand(num_heads, -1, -1)
            else:  # "pair"
                shared_m = torch.full(
                    (1, num_sents, num_sents),
                    alpha, device=device, dtype=torch.float32,
                    requires_grad=True,
                )
                raw_masks = shared_m
                expanded = shared_m.expand(num_heads, -1, -1)
                for l in self.layers:
                    masks[l] = expanded

            # Patch model
            handles = self._patch_model(
                masks,
                token_to_sent,
                combined_filter,
                forward_fn,
            )

            if use_global:
                # --- Global objective: two-pass approach ---
                # Pass 1: forward all chains without grad to get chain logprobs
                chain_lps_detached = []
                for cont in continuations:
                    full_input = torch.cat([input_ids, cont], dim=-1)
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        logits = self.model(full_input).logits
                    lp = chain_log_prob(logits.float(), full_input, prefix_len)
                    chain_lps_detached.append(lp.detach())
                chain_lps_detached = torch.stack(chain_lps_detached)

                # Compute per-chain gradient weights via autograd on small graph
                chain_lps_param = chain_lps_detached.clone().requires_grad_(True)
                global_loss = self.objective_fn(
                    chain_lps_param, chain_logprobs_clean,
                    answer_ids.to(device), num_answers,
                )
                global_loss.backward()
                per_chain_weights = chain_lps_param.grad.detach()  # (N,)

                # Pass 2: forward each chain with grad, weight by per-chain gradient
                for cont_idx, cont in enumerate(continuations):
                    full_input = torch.cat([input_ids, cont], dim=-1)

                    # Zero grads from previous continuation
                    if granularity == "pair":
                        if raw_masks.grad is not None:
                            raw_masks.grad.detach_()
                            raw_masks.grad.zero_()
                    else:
                        for l in self.layers:
                            rm = raw_masks[l]
                            if rm.grad is not None:
                                rm.grad.detach_()
                                rm.grad.zero_()

                    with torch.amp.autocast("cuda"):
                        logits = self.model(full_input).logits

                    lp = chain_log_prob(logits.float(), full_input, prefix_len)
                    weighted_loss = lp * per_chain_weights[cont_idx]
                    weighted_loss.backward()

                    # Accumulate grads
                    if granularity == "pair":
                        if raw_masks.grad is not None:
                            accumulated_grads += raw_masks.grad.detach().cpu()
                    else:
                        for l in self.layers:
                            rm = raw_masks[l]
                            if rm.grad is not None:
                                accumulated_grads[l] += rm.grad.detach().cpu()
            else:
                # --- Local objective: per-chain forward + backward ---
                for cont_idx, cont in enumerate(continuations):
                    full_input = torch.cat([input_ids, cont], dim=-1)
                    full_len = full_input.shape[-1]
                    position_mask = self._build_position_mask(full_len, prefix_len, device)
                    if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                        position_mask = position_mask_overrides[cont_idx].to(device)
                    clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

                    # Zero grads from previous continuation
                    if granularity == "pair":
                        if raw_masks.grad is not None:
                            raw_masks.grad.detach_()
                            raw_masks.grad.zero_()
                    else:
                        for l in self.layers:
                            rm = raw_masks[l]
                            if rm.grad is not None:
                                rm.grad.detach_()
                                rm.grad.zero_()

                    with torch.amp.autocast("cuda"):
                        logits = self.model(full_input).logits

                    # Cast to float32 to match clean_logits precision — avoids
                    # spurious KL from bfloat16-vs-float32 log_softmax differences.
                    loss = self.objective_fn(
                        clean_logits, logits.float(), position_mask, token_ids=full_input
                    )
                    if branch_rewards is not None:
                        loss = loss * branch_rewards[cont_idx]
                    loss.backward()

                    # Accumulate grads
                    if granularity == "pair":
                        if raw_masks.grad is not None:
                            accumulated_grads += raw_masks.grad.detach().cpu()
                    else:
                        for l in self.layers:
                            rm = raw_masks[l]
                            if rm.grad is not None:
                                accumulated_grads[l] += rm.grad.detach().cpu()

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

        # Normalize by sentence-pair token counts if pair_aggregation == "mean".
        # Autograd sums over all (t_q, t_k) pairs; dividing by |sent_i|*|sent_j|
        # converts the sum to a mean so longer sentences don't dominate.
        if self.pair_aggregation == "mean":
            sent_lens = torch.tensor(
                [s.end - s.start + 1 for s in sentences], dtype=torch.float32
            )
            pair_counts = (sent_lens.unsqueeze(1) * sent_lens.unsqueeze(0)).clamp_min(1.0)
        else:
            pair_counts = None

        if granularity == "head":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                if pair_counts is not None:
                    avg /= pair_counts.unsqueeze(0)
                scores[l] = {h: avg[h].tolist() for h in range(num_heads)}
        elif granularity == "layer":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                if pair_counts is not None:
                    avg /= pair_counts.unsqueeze(0)
                scores[l] = avg[0].tolist()
        else:  # "pair"
            avg = sign * accumulated_grads / num_total
            if pair_counts is not None:
                avg /= pair_counts.unsqueeze(0)
            scores = avg[0].tolist()

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_attribution",
            layers=self.layers,
            sentences=[
                {"start": s.start, "end": s.end} for s in sentences
            ],
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
            },
            scores=scores,
        )
