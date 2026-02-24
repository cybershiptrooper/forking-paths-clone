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
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import make_llama_attention_forward, apply_sentence_mask


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

    def discover(
        self,
        input_ids: torch.Tensor,
        sentences: List[Sentence],
        continuations: List[torch.Tensor],
        mask_mode: str = "prefix",
        num_prefix_sentences: Optional[int] = None,
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
        forward_fn = make_llama_attention_forward(apply_sentence_mask)

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

        # 1. Pre-compute clean logits (with non-target layers ablated if enabled)
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

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
            f"{len(continuations)} continuations, granularity={granularity})..."
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

            # Forward with grad for each continuation
            for cont_idx, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                position_mask = self._build_position_mask(full_len, prefix_len, device)
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
                loss = self.objective_fn(clean_logits, logits.float(), position_mask)
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

        if granularity == "head":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                scores[l] = {h: avg[h].tolist() for h in range(num_heads)}
        elif granularity == "layer":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                scores[l] = avg[0].tolist()
        else:  # "pair"
            avg = sign * accumulated_grads / num_total
            scores = avg[0].tolist()

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
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "mask_granularity": granularity,
            },
            scores=scores,
        )
