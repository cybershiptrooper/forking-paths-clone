"""Column-granularity subnetwork probing.

Learns a per-sentence (column) mask instead of the per-pair (S x S) mask
used by :class:`NodewiseSubnetworkProbingSDPA`.  Each sentence j gets one
learnable logit ``log_alpha[j]``; the mask is broadcast column-wise so
``mask[i][j] = m[j]`` for all rows i.  The intervention is "zero all
attention to sentence j" — identical to thought anchors' column-zero, but
trained jointly across all columns via gradient descent.

Two ``training_gap_mode`` settings control how the gap filter is applied
during training:

- ``"matched"`` (default): the real gap filter is applied during training,
  freezing near-diagonal cells at 1.  The gradient for column j never sees
  the self-attention removal effect — exactly matching the eval setting.
- ``"unrestricted"``: no gap filter during training (zeros everywhere, like
  thought anchors).  The mask can freely zero any column including the
  diagonal.  At eval the real gap filter is still applied, creating a
  train/eval mismatch on the diagonal (same as thought anchors).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from utils.masks import NodeMask
from utils.circuit_eval import Sentence
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _hard_concrete_sample,
    _hard_concrete_mean,
    _HC_BETA,
)


class ColumnSubnetworkProbing(NodewiseSubnetworkProbingSDPA):
    """Column-granularity subnetwork probing.

    Overrides the parent to use a 1D ``log_alpha`` of shape ``(1, S)``
    instead of ``(1, S, S)``.  The parent's ``isinstance(log_alpha, Tensor)``
    checks route column tensors through the same branches as pair tensors.
    ``_score_readout`` is overridden to save a 1D list.
    """

    def __init__(
        self,
        *args,
        training_gap_mode: str = "matched",
        uniform_column_l0: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("mask_granularity", "column")
        super().__init__(*args, **kwargs)
        if training_gap_mode not in ("matched", "unrestricted"):
            raise ValueError(
                f"training_gap_mode must be 'matched' or 'unrestricted', "
                f"got {training_gap_mode!r}"
            )
        self.training_gap_mode = training_gap_mode
        # If True, the target_size_relu L0 counts each COLUMN (sentence)
        # equally: expected number of active columns vs a column-count
        # target.  The inherited behaviour counts expected active 2D cells,
        # which weighs sentence j by its number of valid attention cells —
        # later (causal) columns have fewer valid rows and are cheaper to
        # keep open, biasing both the achieved column count and which
        # sentences are kept.
        self.uniform_column_l0 = uniform_column_l0

    # ------------------------------------------------------------------
    # Overrides — always use column logic regardless of granularity param
    # ------------------------------------------------------------------

    def _init_log_alpha(self, granularity, num_heads, num_sents, device):
        if isinstance(self.log_alpha_init, str):
            if self.log_alpha_init != "random":
                raise ValueError(
                    f"log_alpha_init must be a float or 'random', "
                    f"got {self.log_alpha_init!r}"
                )
            # Uniform(-2, 2) per gate: spans hard-closed (≤ -1.6, HC mean 0)
            # to fully open (≥ +1.6, HC mean 1).
            init = torch.empty(
                (1, num_sents), device=device, dtype=torch.float32,
            ).uniform_(-2.0, 2.0)
            return init.requires_grad_(True)
        return torch.full(
            (1, num_sents),
            self.log_alpha_init,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        sampled_col = _hard_concrete_sample(log_alpha, beta=beta)  # (1, S)
        sampled_col = self._apply_dropout(sampled_col)
        num_sents = log_alpha.shape[-1]
        # Broadcast column-wise: mask[i][j] = m[j] for all rows i
        sampled_2d = sampled_col.unsqueeze(1).expand(-1, num_sents, -1)  # (1, S, S)
        return {l: sampled_2d for l in self.layers}

    def _current_sparsity(self, log_alpha, granularity, combined_filter=None) -> float:
        with torch.no_grad():
            m = _hard_concrete_mean(log_alpha).flatten()  # (S,)
            return (m == 0).sum().item() / m.numel() if m.numel() > 0 else 0.0

    # NOTE: by default _l0 falls through to the parent's "pair" branch:
    # probs = _hard_concrete_l0_probs(log_alpha) → (1, S) broadcast against
    # valid (1, S, S), i.e. the expected active 2D cells — sentence j is
    # weighed by its number of valid cells.  With ``uniform_column_l0=True``
    # each column counts equally instead (see _l0 below).

    def _l0(self, log_alpha, granularity, combined_filter=None):
        if not self.uniform_column_l0:
            return super()._l0(log_alpha, granularity, combined_filter)
        assert self.sparsity_loss_mode == "target_size_relu", (
            "uniform_column_l0 is only defined for target_size_relu"
        )
        assert combined_filter is not None
        import torch as _torch
        from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
            _hard_concrete_l0_probs,
        )
        # A column is budget-valid iff it has at least one learnable cell
        # (columns with none — the frozen prompt, and the final rankable
        # sentence at gap>=1 — receive no gradient and are excluded from
        # both the count and the target).
        learnable_cols = (~combined_filter.bool()).any(dim=0)      # (S,)
        probs = _hard_concrete_l0_probs(log_alpha).flatten()        # (S,)
        valid = learnable_cols.to(probs.dtype)
        n_active = (probs * valid).sum()
        n_valid = valid.sum()
        target_n = (1.0 - self.target_sparsity) * n_valid
        return _torch.relu(n_active - target_n)

    # NOTE: _params_as_list is NOT overridden.  The parent's "pair" branch
    # returns [log_alpha], which works for our single (1, S) tensor.

    # NOTE: _ones_masks is NOT overridden.  The parent returns (1, S, S)
    # all-ones per layer, which is correct for the clean reference.

    # ------------------------------------------------------------------
    # Score readout — 1D column scores
    # ------------------------------------------------------------------

    def _score_readout(self, readout_source, readout_fn, granularity, num_heads):
        with torch.no_grad():
            r = readout_fn(readout_source)  # (1, S)
            return r[0].tolist()            # [float] length S

    # ------------------------------------------------------------------
    # discover() — gap filter swap + column metadata fixup
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
        real_gap = self.sentence_gap
        if self.training_gap_mode == "unrestricted":
            self.sentence_gap = 0

        result = super().discover(
            input_ids,
            sentences,
            continuations,
            mask_mode=mask_mode,
            num_prefix_sentences=num_prefix_sentences,
            branch_rewards=branch_rewards,
            position_mask_overrides=position_mask_overrides,
            **kwargs,
        )

        self.sentence_gap = real_gap

        result.algorithm = "column_subnetwork_probing"
        result.metadata["sentence_gap"] = real_gap
        result.metadata["training_gap_mode"] = self.training_gap_mode
        result.metadata["uniform_column_l0"] = self.uniform_column_l0

        return result
