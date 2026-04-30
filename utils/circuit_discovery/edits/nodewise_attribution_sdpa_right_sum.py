"""SDPA-based integrated gradients for sentence-level attention mask attribution.

Drop-in equivalent of ``nodewise_attribution.NodewiseAttribution`` that
replaces the eager post-softmax multiply-and-renormalize with an SDPA
pre-softmax additive log-mask. This avoids materializing the O(B·H·S²)
attention weight matrix, which at 10k tokens on an 8B model costs ~6 GB
per layer and dominates the OOM profile of the eager path.

Mathematical equivalence (used to justify the swap):

    post-softmax:  (softmax(QK^T) ⊙ M) / Σ softmax(QK^T) ⊙ M
    pre-softmax:    softmax(QK^T + log M)

Both are differentiable in M ∈ (0, 1].

Memory-efficient SDPA accepts a dense additive ``attn_mask`` without
materialising the attention weights. The mask itself is O(H·Q·K) in
bf16, so at 10k tokens × 32 heads we allocate ~6 GB per layer forward —
but only one layer is in flight at a time under gradient checkpointing,
which is strongly recommended.

Not reused from ``nodewise_patching_flash``:
    - ``_get_prefix_kv_cache`` / ``_expand_kv_cache_for_batch``. Those
      assume the prefix attention is independent of the mask, which is
      only true for ``mask_mode="generation"``. A ``prefix``-mode IG step
      varies the prefix attention, so a fixed prefix KV cache would
      silently break gradients. Omitted for now; re-add guarded by
      ``mask_mode != "prefix"`` later if useful.
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
from utils.utils import Sentence
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.sdpa_forward import (
    make_sdpa_attention_forward,
)


_ALLOWED_PAIR_AGGREGATIONS = {"sum", "mean"}
# See nodewise_attribution_sdpa._LOG_MASK_EPS for rationale.
_LOG_MASK_EPS = 1e-30


def _expand_mask_to_log_additive(
    module,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Differentiable log-mask for SDPA pre-softmax additive bias.

    Reads ``_circuit_mask`` (continuous, requires_grad) from *module* and
    returns ``log(mask)`` expanded to token level as ``(1, H_src, q_len, k_len)``
    where ``H_src`` matches the source mask's first dim (``num_heads`` for
    head granularity, ``1`` for layer / pair — SDPA broadcasts the 1-dim to
    H_attention_heads without materialising anything).

    Memory-tight: log is applied to the tiny ``(H_src, S, S)`` source
    tensor; then a single gather writes directly into the target dtype.
    At ``pair`` granularity with ``H_src=1`` and 10k tokens this is an
    ~180 MB bf16 allocation instead of the multi-GB fp32 intermediates a
    naive implementation produces.

    Fast path: returns *None* when every query is outside any sentence
    (token_to_sent == -1) — the mask is identically 1.0 and the bias
    identically 0.0, so SDPA can skip the additive term entirely.
    """
    mask = getattr(module, "_circuit_mask", None)
    token_to_sent = getattr(module, "_token_to_sent", None)
    gap_filter = getattr(module, "_gap_filter", None)

    if mask is None or token_to_sent is None:
        return None

    device = mask.device
    token_to_sent = token_to_sent.to(device)

    if cache_position is not None:
        q_sent = token_to_sent[cache_position.to(device).long()]
        if (q_sent == -1).all():
            return None
    else:
        q_sent = token_to_sent[:q_len]
    k_sent = token_to_sent[:k_len]

    # Log on the source (H_src, S, S) — tiny, stays in float32 for stability.
    log_source = torch.log(mask.float().clamp_min(_LOG_MASK_EPS))
    # Gap entries → log(1) = 0 in log-space (additive identity).
    log_source = apply_gap_filter(log_source, gap_filter, fill_value=0.0)

    num_heads_src, num_sents, _ = log_source.shape

    # Pad with sentinel 0.0 (= log(1)) for tokens outside any sentence.
    # Cast to target dtype BEFORE the big gather so the (H_src, Q, K)
    # result lives in bf16 only, not float32.
    padded = torch.zeros(
        num_heads_src, num_sents + 1, num_sents + 1,
        device=device, dtype=log_source.dtype,
    )
    padded[:, :num_sents, :num_sents] = log_source
    padded = padded.to(dtype)

    q_idx = q_sent.clone()
    k_idx = k_sent.clone()
    q_idx[q_sent == -1] = num_sents
    k_idx[k_sent == -1] = num_sents

    # Single allocation of shape (H_src, q_len, k_len) in target dtype.
    log_token = padded[:, q_idx][:, :, k_idx]
    return log_token.unsqueeze(0)  # (1, H_src, q_len, k_len) — SDPA broadcasts


class NodewiseAttributionSDPA(CircuitDiscovery):
    """Integrated gradients over sentence masks using SDPA attention.

    Behaviourally identical to ``NodewiseAttribution`` (same IG formula,
    same sign convention, same aggregation options) but runs on an SDPA
    backend with a differentiable log-additive mask. Scales to ~10k-token
    contexts on an 8B model that the eager path OOMs on.

    Recommended: enable gradient checkpointing on the model before
    calling ``discover`` to further reduce per-layer activation memory::

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
    """

    def __init__(self, num_ig_steps: int = 10, negate_scores: bool = True, **kwargs):
        self.pair_aggregation = kwargs.pop("pair_aggregation", "sum")
        if self.pair_aggregation not in _ALLOWED_PAIR_AGGREGATIONS:
            raise ValueError(
                f"pair_aggregation must be one of {_ALLOWED_PAIR_AGGREGATIONS}, "
                f"got {self.pair_aggregation!r}"
            )
        kwargs.pop("batch_chunk_size", None)
        super().__init__(**kwargs)
        self.num_ig_steps = num_ig_steps
        self.negate_scores = negate_scores

    # ------------------------------------------------------------------
    # Patching helpers — SDPA forward with log-additive mask converter
    # ------------------------------------------------------------------

    def _sdpa_forward(self):
        return make_sdpa_attention_forward(
            self.model_type, mask_converter=_expand_mask_to_log_additive,
        )

    def _patch_non_target_layers_sdpa(
        self,
        num_heads: int,
        num_sents: int,
        token_to_sent: torch.Tensor,
        gap_filter: torch.Tensor,
    ):
        """Zero-ablate every layer outside ``self.layers`` using SDPA forward.

        A zero mask maps to ``log(eps)`` which SDPA treats as -inf, giving
        exact zero attention weight on those edges after softmax.
        """
        import types
        from utils.utils import get_attention_module
        from utils.circuit_discovery.base import AblationHandle

        device = next(self.model.parameters()).device
        num_all = self.model.config.num_hidden_layers
        target_set = set(self.layers)
        non_target = [l for l in range(num_all) if l not in target_set]
        sdpa_fwd = self._sdpa_forward()
        handles = []
        for layer_idx in non_target:
            attn_module = get_attention_module(self.model, layer_idx)
            original_forward = attn_module.forward
            # All heads ablated identically → (1, S, S) suffices; SDPA
            # broadcasts the additive bias across the head dim for free.
            zero_mask = torch.zeros(
                1, num_sents, num_sents, device=device,
            )
            attn_module._circuit_mask = zero_mask
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module._renormalize_masked_attn = self.renormalize_masked_attention
            attn_module.forward = types.MethodType(sdpa_fwd, attn_module)
            handles.append(AblationHandle(attn_module, original_forward))
        return handles

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
        combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

        forward_fn = self._sdpa_forward()

        # ----- Ablate non-target layers (optional) -----
        non_target_handles = []
        if self.ablate_non_target_layers:
            print(
                f"Ablating all layers outside {self.layers} "
                f"({self.model.config.num_hidden_layers - len(self.layers)} layers)..."
            )
            non_target_handles = self._patch_non_target_layers_sdpa(
                num_heads=num_heads,
                num_sents=num_sents,
                token_to_sent=token_to_sent,
                gap_filter=combined_filter,
            )

        # ----- Global-objective setup -----
        objective_name = getattr(self.objective_fn, "__name__", "unknown")
        use_global = is_global_objective(objective_name)
        answer_ids = kwargs.get("answer_ids")
        num_answers = kwargs.get("num_answers")
        if use_global and (answer_ids is None or num_answers is None):
            raise ValueError(
                f"Global objective '{objective_name}' requires answer_ids and "
                f"num_answers to be passed to discover()."
            )

        # ----- Clean logits (no grad) -----
        print("Computing clean logits...")
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

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

        chain_lengths = torch.tensor(
            [c.shape[-1] for c in continuations],
            dtype=torch.long, device=device,
        )

        # ----- Accumulator shape mirrors the granularity -----
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
            f"Running integrated gradients [SDPA] ({self.num_ig_steps} steps, "
            f"{len(continuations)} continuations, "
            f"aggregation={self.pair_aggregation}, granularity={granularity})..."
        )

        # ----- IG loop -----
        for step in tqdm(range(1, self.num_ig_steps + 1), desc="IG steps"):
            alpha = step / self.num_ig_steps
            masks, raw_masks = self._make_masks_at_alpha(
                alpha, granularity, num_heads, num_sents, device,
            )

            handles = self._patch_model(
                masks, token_to_sent, combined_filter, forward_fn,
            )

            if use_global:
                self._ig_step_global(
                    input_ids, continuations, prefix_len, device,
                    chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
                    raw_masks, granularity, accumulated_grads,
                )
            else:
                self._ig_step_local(
                    input_ids, continuations, clean_logits_list,
                    prefix_len, device, branch_rewards, position_mask_overrides,
                    raw_masks, granularity, accumulated_grads,
                )

            self._unpatch_model(handles)

        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # ----- Average + aggregate -----
        num_total = self.num_ig_steps * len(continuations)
        sign = -1.0 if self.negate_scores else 1.0

        if self.pair_aggregation == "mean":
            sent_lens = torch.tensor(
                [s.end - s.start + 1 for s in sentences], dtype=torch.float32,
            )
            pair_counts = (sent_lens.unsqueeze(1) * sent_lens.unsqueeze(0)).clamp_min(1.0)
        else:
            pair_counts = None

        if granularity == "head":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                if pair_counts is not None:
                    avg = avg / pair_counts.unsqueeze(0)
                scores[l] = {h: avg[h].tolist() for h in range(num_heads)}
        elif granularity == "layer":
            scores = {}
            for l in self.layers:
                avg = sign * accumulated_grads[l] / num_total
                if pair_counts is not None:
                    avg = avg / pair_counts.unsqueeze(0)
                scores[l] = avg[0].tolist()
        else:  # "pair"
            avg = sign * accumulated_grads / num_total
            if pair_counts is not None:
                avg = avg / pair_counts.unsqueeze(0)
            scores = avg[0].tolist()

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_attribution_sdpa",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name=objective_name,
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
                "attention_backend": "sdpa",
            },
            scores=scores,
        )

    # ------------------------------------------------------------------
    # IG step helpers
    # ------------------------------------------------------------------

    def _make_masks_at_alpha(
        self,
        alpha: float,
        granularity: str,
        num_heads: int,
        num_sents: int,
        device: torch.device,
    ):
        """Return ``(masks, raw_masks)`` at IG interpolation ``alpha``.

        For ``layer`` / ``pair`` granularity the installed mask keeps its
        ``(1, S, S)`` shape — the SDPA log-mask expansion then produces a
        ``(1, 1, q, k)`` additive bias that SDPA broadcasts across heads
        without materialising per-head memory.
        """
        masks = {}
        if granularity == "head":
            for l in self.layers:
                m = torch.full(
                    (num_heads, num_sents, num_sents),
                    alpha, device=device, dtype=torch.float32, requires_grad=True,
                )
                masks[l] = m
            return masks, masks
        if granularity == "layer":
            raw = {}
            for l in self.layers:
                m = torch.full(
                    (1, num_sents, num_sents),
                    alpha, device=device, dtype=torch.float32, requires_grad=True,
                )
                raw[l] = m
                masks[l] = m
            return masks, raw
        # "pair" — one shared tensor across all target layers
        shared = torch.full(
            (1, num_sents, num_sents),
            alpha, device=device, dtype=torch.float32, requires_grad=True,
        )
        for l in self.layers:
            masks[l] = shared
        return masks, shared

    @staticmethod
    def _zero_mask_grads(raw_masks, granularity: str, layers):
        if granularity == "pair":
            if raw_masks.grad is not None:
                raw_masks.grad.detach_()
                raw_masks.grad.zero_()
        else:
            for l in layers:
                g = raw_masks[l].grad
                if g is not None:
                    g.detach_()
                    g.zero_()

    @staticmethod
    def _accumulate_mask_grads(raw_masks, granularity: str, layers, accumulated):
        if granularity == "pair":
            if raw_masks.grad is not None:
                accumulated += raw_masks.grad.detach().cpu()
        else:
            for l in layers:
                g = raw_masks[l].grad
                if g is not None:
                    accumulated[l] += g.detach().cpu()

    def _ig_step_local(
        self,
        input_ids, continuations, clean_logits_list,
        prefix_len, device, branch_rewards, position_mask_overrides,
        raw_masks, granularity, accumulated,
    ):
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]
            position_mask = self._build_position_mask(full_len, prefix_len, device)
            if (
                position_mask_overrides is not None
                and position_mask_overrides[cont_idx] is not None
            ):
                position_mask = position_mask_overrides[cont_idx].to(device)
            clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

            self._zero_mask_grads(raw_masks, granularity, self.layers)

            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits

            loss = self.objective_fn(
                clean_logits, logits.float(), position_mask, token_ids=full_input,
            )
            if branch_rewards is not None:
                loss = loss * branch_rewards[cont_idx]
            loss.backward()

            self._accumulate_mask_grads(
                raw_masks, granularity, self.layers, accumulated,
            )
            del logits, loss

    def _ig_step_global(
        self,
        input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
        raw_masks, granularity, accumulated,
    ):
        # Pass 1: no-grad chain logprobs under the current mask.
        chain_lps_detached = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits
            lp = chain_log_prob(
                logits.float(), full_input, prefix_len, temperature=self.temperature,
            )
            chain_lps_detached.append(lp.detach())
        chain_lps_detached = torch.stack(chain_lps_detached)

        # Per-chain gradient weights via a small autograd graph.
        chain_lps_param = chain_lps_detached.clone().requires_grad_(True)
        global_loss = self.objective_fn(
            chain_lps_param, chain_logprobs_clean,
            answer_ids.to(device), num_answers,
            chain_lengths=chain_lengths,
            is_method=self.importance_sampling_method,
            is_temperature=self.importance_sampling_temperature,
        )
        global_loss.backward()
        per_chain_weights = chain_lps_param.grad.detach()

        # Pass 2: per-chain forward with grad, weighted.
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            self._zero_mask_grads(raw_masks, granularity, self.layers)

            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits
            lp = chain_log_prob(
                logits.float(), full_input, prefix_len, temperature=self.temperature,
            )
            (lp * per_chain_weights[cont_idx]).backward()

            self._accumulate_mask_grads(
                raw_masks, granularity, self.layers, accumulated,
            )
            del logits, lp
