"""Subnetwork probing with Hard-Concrete samples batched along the batch dim.

``NodewiseSubnetworkProbingSDPA`` with ``num_hc_samples_per_step = K > 1``
runs K *sequential* forward+backward passes per training step (one per
Hard-Concrete mask sample), so wall-clock scales linearly with K
(measured: ~300 s / 598 s / 1178 s / 2339 s per 1000-step Qwen3-8B run at
K = 1 / 2 / 4 / 8). This subclass computes the same K-sample-averaged
gradient with the samples stacked along the *batch* dimension of a single
forward pass instead:

- ``_sample_masks`` draws all K samples at once, stacked as ``(K, S, S)``
  (pair / layer granularity; head granularity is not supported).
- The SDPA mask converter transposes the expanded log-additive bias from
  ``(1, K, q, k)`` to ``(K, 1, q, k)`` so each batch row gets its own
  sample's bias, broadcast over heads.
- ``_step_local`` replicates the input row K times (in micro-batches of
  ``hc_samples_per_forward`` rows when K rows don't fit in memory) and
  backpropagates the mean loss over the batch rows.

The resulting ``log_alpha.grad`` is mathematically identical to the
sequential implementation's 1/K-scaled accumulated gradient (verified in
``tests/test_snp_hc_batched.py``); only the RNG stream consumption
differs, so individual runs are not bit-reproductions of sequential runs
with the same seed.

Config keys (via the standard sweep YAML → run.py plumbing):

- ``num_hc_samples_per_step``: K, as before.
- ``batch_chunk_size``: samples per forward micro-batch (this key already
  flows through run.py/learn.py for other algorithms; here it means
  "HC samples per forward"). Default: all K in one forward.

Only local objectives are supported (``_step_global`` raises). The
training-metrics JSONL logs ``num_hc_samples_per_step: 1`` (the base
class's inner loop runs once); the saved mask metadata records the true
K and the micro-batch size.

Registered as ``nodewise_subnetwork_probing_hc_batched``.
"""

from typing import Optional

import torch

from utils.circuit_discovery.edits.nodewise_attribution_sdpa import (
    _expand_mask_to_log_additive,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _HC_BETA,
    _hard_concrete_sample,
)
from utils.circuit_discovery.sdpa_forward import make_sdpa_attention_forward


def _expand_mask_to_log_additive_hc_batched(
    module,
    q_len: int,
    k_len: int,
    cache_position,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Batched variant of ``_expand_mask_to_log_additive``.

    The base converter treats ``_circuit_mask``'s first dim as heads and
    returns ``(1, H_src, q, k)``. Here the first dim is the HC-sample
    batch, so the result is transposed to ``(B, 1, q, k)``: one bias per
    batch row, broadcast over heads. For an unbatched ``(1, S, S)`` mask
    (clean-logits pass, non-target-layer ablation) the transpose is a
    no-op and the bias broadcasts over both batch and heads, exactly as
    before.
    """
    out = _expand_mask_to_log_additive(module, q_len, k_len, cache_position, dtype)
    if out is None:
        return None
    return out.transpose(0, 1)


# Local objectives whose value depends on the logits only through the
# rows selected by ``position_mask`` *after* their own internal float()
# cast — for these the (B, L, V) logits can be handed over in bf16
# without the base class's full-tensor ``.float()`` copy.
_SELECT_THEN_FLOAT_OBJECTIVES = {
    "answer_probe_kl_loss",
    "answer_probe_target_kl_loss",
}


class NodewiseSubnetworkProbingHCBatched(NodewiseSubnetworkProbingSDPA):
    """SNP with the K Hard-Concrete samples batched into one forward."""

    def __init__(self, batch_chunk_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        # The base training loop's inner sample loop must run exactly once;
        # all K samples ride along the batch dimension instead.
        self._hc_total_samples = self.num_hc_samples_per_step
        self.num_hc_samples_per_step = 1
        if batch_chunk_size is not None and int(batch_chunk_size) < 1:
            raise ValueError(
                f"batch_chunk_size must be >= 1, got {batch_chunk_size}"
            )
        self.hc_samples_per_forward = (
            int(batch_chunk_size) if batch_chunk_size is not None
            else self._hc_total_samples
        )
        self._installed_mask_dict = None

    # -- SDPA forward with the batch-transposing converter ---------------

    def _sdpa_forward(self):
        return make_sdpa_attention_forward(
            self.model_type,
            mask_converter=_expand_mask_to_log_additive_hc_batched,
        )

    # -- K samples stacked on dim 0 --------------------------------------

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        K = self._hc_total_samples
        if K == 1:
            return super()._sample_masks(log_alpha, granularity, beta=beta)
        if granularity == "head":
            raise NotImplementedError(
                "nodewise_subnetwork_probing_hc_batched supports pair and "
                "layer granularity only (head masks are (H, S, S); stacking "
                "K samples would need a (K, H, S, S) converter)."
            )
        if isinstance(log_alpha, torch.Tensor):
            # pair: (1, S, S) -> (K, S, S), one independent sample per row;
            # grad flows back through the expand (summing over rows).
            expanded = log_alpha.expand(K, -1, -1)
            sampled = self._apply_dropout(
                _hard_concrete_sample(expanded, beta=beta)
            )
            return {l: sampled for l in self.layers}
        return {
            l: self._apply_dropout(
                _hard_concrete_sample(log_alpha[l].expand(K, -1, -1), beta=beta)
            )
            for l in self.layers
        }

    def _install_masks(self, masks):
        # Stash the full (K, S, S) dict so _step_local can install
        # per-micro-batch slices (views of the same leaves).
        self._installed_mask_dict = masks
        super()._install_masks(masks)

    # -- One batched forward per step (micro-batched) --------------------

    def _step_local(
        self,
        input_ids, continuations, clean_logits_list,
        prefix_len, device, branch_rewards, position_mask_overrides,
    ):
        K = self._hc_total_samples
        if K == 1:
            return super()._step_local(
                input_ids, continuations, clean_logits_list,
                prefix_len, device, branch_rewards, position_mask_overrides,
            )
        masks_full = self._installed_mask_dict
        assert masks_full is not None, "_install_masks must run before _step_local"
        B = max(1, min(self.hc_samples_per_forward, K))
        obj_name = getattr(self.objective_fn, "__name__", "")
        select_then_float = obj_name in _SELECT_THEN_FLOAT_OBJECTIVES

        task_loss_total = 0.0
        try:
            for cont_idx, cont in enumerate(continuations):
                full_input = torch.cat([input_ids, cont], dim=-1)
                full_len = full_input.shape[-1]
                position_mask = self._build_position_mask(
                    full_len, prefix_len, device,
                )
                if (
                    position_mask_overrides is not None
                    and position_mask_overrides[cont_idx] is not None
                ):
                    position_mask = position_mask_overrides[cont_idx].to(device)
                clean_logits = clean_logits_list[cont_idx][:, :full_len].to(device)

                for start in range(0, K, B):
                    b = min(B, K - start)
                    # Slices are views of the installed leaves, so grads
                    # accumulate on the full (K, S, S) leaf tensors.
                    super()._install_masks(
                        {l: m[start:start + b] for l, m in masks_full.items()}
                    )
                    inp = full_input.expand(b, -1)
                    with torch.amp.autocast("cuda"):
                        logits = self.model(inp).logits  # (b, L, V)

                    # The objectives above select the probed rows before
                    # their own float() cast, so skipping the full-tensor
                    # .float() here is numerically identical to the base
                    # class (which upcasts logits that were computed in
                    # bf16 anyway) and avoids a (b, L, V) fp32 copy.
                    masked = logits if select_then_float else logits.float()
                    loss = self.objective_fn(
                        clean_logits.expand(b, -1, -1),
                        masked,
                        position_mask.expand(b, -1),
                        token_ids=inp,
                    )
                    if branch_rewards is not None:
                        loss = loss * branch_rewards[cont_idx]
                    # objective averages over the b rows; weight by b/K so
                    # the micro-batches sum to the mean over all K samples
                    # (matching the sequential 1/K-scaled accumulation).
                    loss = loss * (b / K)
                    task_loss_total += float(loss.detach().item())
                    loss.backward()
                    del logits, masked, loss
        finally:
            super()._install_masks(masks_full)
        return task_loss_total

    def _step_global(self, *args, **kwargs):
        raise NotImplementedError(
            "nodewise_subnetwork_probing_hc_batched only supports local "
            "objectives; use nodewise_subnetwork_probing_sdpa for global "
            "(importance-sampling) objectives."
        )

    # -- Metadata fix-up ---------------------------------------------------

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        node_mask.algorithm = "nodewise_subnetwork_probing_hc_batched"
        node_mask.metadata["num_hc_samples_per_step"] = self._hc_total_samples
        node_mask.metadata["hc_samples_per_forward"] = self.hc_samples_per_forward
        node_mask.metadata["hc_sample_batching"] = True
        return node_mask
