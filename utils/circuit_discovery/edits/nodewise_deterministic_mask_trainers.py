"""Two sampling-free mask trainers for the answer-preservation (KL) task.

Both are subclasses of ``NodewiseSubnetworkProbingHCBatched`` and reuse its
training loop unchanged: the rollout-bank plumbing (``continuations_per_step``,
``clean_logits_dtype``), the λ schedule, the hybrid optimiser (Adam on the
task gradient, plain SGD on the penalty), checkpoints and the JSONL / wandb
logging. The only methods overridden are the ones that produce the mask for
a step (``_sample_masks``), the sparsity penalty (``_l0``) and the sparsity
diagnostics. Nothing in the existing trainers is changed.

1. ``NodewiseStraightThroughTopK`` — registered as ``nodewise_straight_through_topk``

   One real-valued score per learnable edge (stored in ``log_alpha`` so the
   saved mask and the evaluator's top-k readout work unchanged).

   Forward pass: the binary mask that keeps the ``n_keep`` highest-scoring
   learnable edges and removes the rest. ``n_keep`` follows a linear ramp:
   nothing removed at step 0, the target number removed at step
   ``st_ramp_frac * num_training_steps``, held there afterwards.

   Backward pass (straight-through): the loss gradient with respect to every
   learnable edge's mask value, kept or removed, is passed to that edge's
   score, either unchanged (``st_surrogate: identity``, the default) or
   multiplied by sigmoid'(score) (``st_surrogate: sigmoid``). This is the
   first-order estimate of how the loss would change if the edge flipped, so
   removed edges keep receiving a signal and can re-enter the kept set.

   Removed edges take the mask value ``st_removed_value`` (default 1e-4)
   instead of 0 during training: the SDPA converter applies log(mask) with a
   clamp at 1e-30, and the clamp passes no gradient to an exact zero. A
   value of 1e-4 scales the attention to a removed sentence by 1e-4, which is
   an ablation for every practical purpose, while its gradient stays
   numerically healthy. The periodic bank evaluation and the saved mask use
   exact zeros.

   No sparsity penalty: the budget is exact by construction. ``l0_lambda``
   must be 0. Initial scores are ``log_alpha_init`` plus Gaussian noise of
   standard deviation ``st_init_noise`` (tie-breaking for the first cut; also
   breaks ties among equal scores of a warm-start mask).

2. ``NodewiseDeterministicContinuousMask`` — registered as ``nodewise_deterministic_continuous``

   Mask m = sigmoid(score) in (0, 1), used directly as the attention scaling
   in every step. No sampling. Penalty, multiplied by the trainer's λ(t)
   schedule (use ``l0_warmup_frac: 0`` so the ramp starts at step 0, since at
   an all-open mask the KL and its gradient are exactly zero):

       (Σ m − n_target)² / n_valid  +  c · Σ m (1 − m) / n_valid

   over learnable edges, with ``n_target = (1 − target_sparsity) · n_valid``
   as in ``target_size_l2`` and ``c = dc_polarization``. The first term is
   the two-sided quadratic budget of the subnetwork-probing trainer applied
   to the mask values instead of the expected gate count; the second pushes
   every value toward 0 or 1 so the top-k readout changes the trained mask
   little. Requires ``sparsity_loss_mode: target_size_l2``.

Both trainers: every ``bank_eval_every`` steps and at the last step, the
binary top-k mask at the target sparsity (exact zeros, the evaluator's rule)
is installed and the task loss is computed on every continuation of the
bank; one line per evaluation goes to ``<log_dir>/bank_topk_eval.jsonl``.
Membership churn of the forward mask (edges entering / leaving the kept set
between logged steps) goes to ``<log_dir>/deterministic_mask_metrics.jsonl``.

Pair granularity and local objectives only. Config keys not known to
``run.py`` are passed through the ``algorithm_kwargs`` mapping of the YAML.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _HC_BETA,
)


def n_keep_at_target(n_valid: int, target_sparsity: float) -> int:
    """Number of learnable edges the evaluator keeps at ``target_sparsity``
    (``eval_log_alpha._build_binary_mask``: round((1 − s) · n_valid))."""
    return max(0, int(round((1.0 - float(target_sparsity)) * n_valid)))


def n_keep_on_ramp(n_valid: int, target_sparsity: float, step: int,
                   ramp_steps: float) -> int:
    """Edges kept at ``step`` of a linear ramp that removes nothing at step 0
    and the full target count at ``ramp_steps``, held afterwards."""
    n_removed_final = n_valid - n_keep_at_target(n_valid, target_sparsity)
    frac = 1.0 if ramp_steps <= 0 else min(1.0, max(0.0, step / ramp_steps))
    return n_valid - int(round(n_removed_final * frac))


def topk_keep_mask(scores: torch.Tensor, learnable: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Boolean (S, S) tensor: True on the ``n_keep`` highest-scoring learnable
    edges. Non-learnable cells are False. Ties are broken by ``torch.topk``."""
    flat = torch.where(
        learnable.flatten(), scores.detach().flatten(),
        torch.full_like(scores.flatten(), float("-inf")),
    )
    keep = torch.zeros_like(flat, dtype=torch.bool)
    if n_keep > 0:
        keep[torch.topk(flat, min(n_keep, int(learnable.sum()))).indices] = True
    return keep.view_as(scores)


def straight_through_mask(scores: torch.Tensor, keep: torch.Tensor, learnable: torch.Tensor,
                          removed_value: float, surrogate: str) -> torch.Tensor:
    """Mask whose value is binary (1 kept, ``removed_value`` removed, 1 on
    non-learnable cells) and whose gradient w.r.t. ``scores`` is that of the
    surrogate (identity or sigmoid) on every learnable cell."""
    hard = torch.where(keep, torch.ones_like(scores), torch.full_like(scores, removed_value))
    hard = torch.where(learnable, hard, torch.ones_like(scores))
    if surrogate == "identity":
        soft = scores
    elif surrogate == "sigmoid":
        soft = torch.sigmoid(scores)
    else:
        raise ValueError(f"st_surrogate must be 'identity' or 'sigmoid', got {surrogate!r}")
    soft = torch.where(learnable, soft, soft.detach())
    return hard.detach() + (soft - soft.detach())


class _DeterministicMaskTrainerBase(NodewiseSubnetworkProbingHCBatched):
    """Shared plumbing: step tracking, learnable filter, periodic binary
    top-k evaluation on the whole bank, membership-churn logging."""

    ALGORITHM_NAME = "deterministic_mask_trainer"

    def __init__(self, bank_eval_every: int = 100, **kwargs):
        kwargs.setdefault("num_hc_samples_per_step", 1)
        super().__init__(**kwargs)
        if self._hc_total_samples != 1:
            raise ValueError(f"{self.ALGORITHM_NAME}: the mask is deterministic, "
                             f"num_hc_samples_per_step must be 1 (got {self._hc_total_samples})")
        if self.mask_granularity != "pair":
            raise NotImplementedError(f"{self.ALGORITHM_NAME} supports pair granularity only")
        if self.target_sparsity is None:
            raise ValueError(f"{self.ALGORITHM_NAME} requires target_sparsity")
        if self.hc_beta_anneal or self.dropout_p > 0.0:
            raise ValueError(f"{self.ALGORITHM_NAME}: hc_beta_anneal and dropout_p do not apply")
        self.bank_eval_every = int(bank_eval_every)
        self._step = 0
        self._combined_filter = None
        self._theta_ref = None
        self._bank_input_ids = None
        self._prev_keep = None
        self._flips_on = 0
        self._flips_off = 0
        self._last_hard_sparsity = 0.0
        self._last_logged_step = None

    # -- hooks into the base training loop (no base changes) ---------------

    def _current_beta(self, step: int) -> float:
        # The loop calls this first in every step; it is our step counter.
        self._step = int(step)
        return _HC_BETA

    def _init_log_alpha(self, granularity, num_heads, num_sents, device, combined_filter=None):
        self._combined_filter = combined_filter
        return super()._init_log_alpha(granularity, num_heads, num_sents, device,
                                       combined_filter=combined_filter)

    def _get_clean_logits(self, input_ids, continuations):
        self._bank_input_ids = input_ids
        return super()._get_clean_logits(input_ids, continuations)

    def _learnable(self, device) -> torch.Tensor:
        assert self._combined_filter is not None, "_init_log_alpha must run first"
        return (~self._combined_filter.bool()).to(device)

    def _n_valid(self, device) -> int:
        return int(self._learnable(device).sum().item())

    # -- diagnostics the base loop logs ---------------------------------------

    def _expected_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        return self._current_sparsity(log_alpha, granularity, combined_filter, beta)

    def _record_membership(self, keep: torch.Tensor, n_valid: int):
        with torch.no_grad():
            if self._prev_keep is not None:
                self._flips_off += int((self._prev_keep & ~keep).sum().item())
                self._flips_on += int((~self._prev_keep & keep).sum().item())
            self._prev_keep = keep.clone()
            self._last_hard_sparsity = 1.0 - float(keep.sum().item()) / max(1, n_valid)

    def _write_jsonl(self, name: str, row: dict):
        if self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, name), "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- periodic binary evaluation on the whole bank -------------------------

    def _continuations_for_step(self, step, continuations, clean_logits_list, branch_rewards,
                                position_mask_overrides):
        last = step + 1 == self.num_training_steps
        if self.bank_eval_every > 0 and (step % self.bank_eval_every == 0 or last):
            self._bank_topk_eval(step, continuations, clean_logits_list, position_mask_overrides)
        if step % self.log_every == 0 or last:
            self._write_jsonl("deterministic_mask_metrics.jsonl", {
                "step": step, "forward_mask_sparsity": self._last_hard_sparsity,
                "flips_on_interval": self._flips_on, "flips_off_interval": self._flips_off,
            })
            self._flips_on = 0
            self._flips_off = 0
        return super()._continuations_for_step(step, continuations, clean_logits_list,
                                               branch_rewards, position_mask_overrides)

    def _scores_2d(self) -> torch.Tensor:
        assert self._theta_ref is not None, "_sample_masks must run first"
        return self._theta_ref[0].detach()

    def _bank_topk_eval(self, step, continuations, clean_logits_list, position_mask_overrides):
        """Task loss of the binary top-k mask at the target (exact zeros, the
        evaluator's rule) on every continuation of the bank."""
        device = self._scores_2d().device
        learnable = self._learnable(device)
        n_valid = int(learnable.sum().item())
        n_keep = n_keep_at_target(n_valid, self.target_sparsity)
        keep = topk_keep_mask(self._scores_2d(), learnable, n_keep)
        hard = torch.where(learnable, keep.float(), torch.ones_like(keep, dtype=torch.float32))
        input_ids = self._bank_input_ids
        prefix_len = input_ids.shape[-1]
        losses = []
        try:
            NodewiseSubnetworkProbingSDPA._install_masks(self, {l: hard.unsqueeze(0) for l in self.layers})
            with torch.no_grad():
                for i, cont in enumerate(continuations):
                    full_input = torch.cat([input_ids, cont], dim=-1)
                    full_len = full_input.shape[-1]
                    position_mask = self._build_position_mask(full_len, prefix_len, device)
                    if position_mask_overrides is not None and position_mask_overrides[i] is not None:
                        position_mask = position_mask_overrides[i].to(device)
                    clean_logits = clean_logits_list[i][:, :full_len].to(device)
                    with torch.amp.autocast("cuda"):
                        logits = self.model(full_input).logits
                    loss = self.objective_fn(clean_logits, logits.float(), position_mask, token_ids=full_input)
                    losses.append(float(loss.detach().item()))
                    del logits, loss
        finally:
            if self._installed_mask_dict is not None:
                NodewiseSubnetworkProbingSDPA._install_masks(self, self._installed_mask_dict)
        t = torch.tensor(losses)
        row = {"step": step, "n_keep": n_keep, "n_valid": n_valid, "n_continuations": len(losses),
               "mean": float(t.mean()), "median": float(t.median()), "max": float(t.max()),
               "per_continuation": losses}
        self._write_jsonl("bank_topk_eval.jsonl", row)
        print(f"  [bank top-k eval] step {step}: n_keep {n_keep}/{n_valid}, mean KL {row['mean']:.5f}, "
              f"median {row['median']:.5f} over {len(losses)} continuations")

    def _step_global(self, *args, **kwargs):
        raise NotImplementedError(f"{self.ALGORITHM_NAME} supports local objectives only")

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        node_mask.algorithm = self.ALGORITHM_NAME
        node_mask.metadata["bank_eval_every"] = self.bank_eval_every
        node_mask.metadata["deterministic_mask"] = True
        node_mask.metadata["hc_sample_batching"] = False
        return node_mask


class NodewiseStraightThroughTopK(_DeterministicMaskTrainerBase):
    """Binary top-k mask in the forward pass, straight-through gradient to the
    scores, linear ramp of the number of removed edges."""

    ALGORITHM_NAME = "nodewise_straight_through_topk"

    def __init__(self, st_ramp_frac: float = 0.5, st_surrogate: str = "identity",
                 st_removed_value: float = 1e-4, st_init_noise: float = 0.01,
                 st_record_membership_every: int = 0, **kwargs):
        super().__init__(**kwargs)
        # Diagnostic: every N steps, store the kept set of the forward mask (bit-packed over
        # the (S, S) grid) and the scores; written to <log_dir>/membership_trace.npz at the end.
        self.st_record_membership_every = int(st_record_membership_every)
        self._trace_steps, self._trace_keep, self._trace_scores = [], [], []
        if self.l0_lambda != 0.0:
            raise ValueError("nodewise_straight_through_topk has no sparsity penalty: set l0_lambda: 0")
        if not (0.0 <= st_ramp_frac <= 1.0):
            raise ValueError(f"st_ramp_frac must be in [0, 1], got {st_ramp_frac}")
        if not (0.0 < st_removed_value < 1.0):
            raise ValueError(f"st_removed_value must be in (0, 1), got {st_removed_value}")
        if st_surrogate not in ("identity", "sigmoid"):
            raise ValueError(f"st_surrogate must be 'identity' or 'sigmoid', got {st_surrogate!r}")
        self.st_ramp_frac = float(st_ramp_frac)
        self.st_surrogate = st_surrogate
        self.st_removed_value = float(st_removed_value)
        self.st_init_noise = float(st_init_noise)

    def _init_log_alpha(self, granularity, num_heads, num_sents, device, combined_filter=None):
        theta = super()._init_log_alpha(granularity, num_heads, num_sents, device,
                                        combined_filter=combined_filter)
        if self.st_init_noise > 0.0:
            with torch.no_grad():
                theta.add_(self.st_init_noise * torch.randn_like(theta))
        return theta

    def _ramp_steps(self) -> float:
        return self.st_ramp_frac * self.num_training_steps

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        self._theta_ref = log_alpha
        theta = log_alpha[0]
        learnable = self._learnable(theta.device)
        n_valid = int(learnable.sum().item())
        n_keep = n_keep_on_ramp(n_valid, self.target_sparsity, self._step, self._ramp_steps())
        keep = topk_keep_mask(theta, learnable, n_keep)
        self._record_membership(keep, n_valid)
        if self.st_record_membership_every > 0 and self._step % self.st_record_membership_every == 0:
            self._trace_steps.append(self._step)
            self._trace_keep.append(np.packbits(keep.detach().cpu().numpy().astype(np.uint8)))
            self._trace_scores.append(theta.detach().float().cpu().numpy().astype(np.float16))
        mask = straight_through_mask(theta, keep, learnable, self.st_removed_value, self.st_surrogate)
        mask = mask.unsqueeze(0)
        return {l: mask for l in self.layers}

    def _current_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        return self._last_hard_sparsity

    def _l0(self, *args, **kwargs):
        raise RuntimeError("nodewise_straight_through_topk has no sparsity penalty (l0_lambda must be 0)")

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        if self._trace_steps and self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.log_dir, "membership_trace.npz"), steps=np.array(self._trace_steps),
                                keep_packed=np.stack(self._trace_keep), scores=np.stack(self._trace_scores),
                                shape=np.array(self._trace_scores[0].shape))
        node_mask.metadata.update({
            "st_record_membership_every": self.st_record_membership_every,
            "st_ramp_frac": self.st_ramp_frac, "st_surrogate": self.st_surrogate,
            "st_removed_value": self.st_removed_value, "st_init_noise": self.st_init_noise,
            "sparsity_loss_mode": "none (exact top-k budget)",
        })
        return node_mask


class NodewiseDeterministicContinuousMask(_DeterministicMaskTrainerBase):
    """Mask sigmoid(score) used directly; two-sided quadratic budget plus a
    polarisation term, both scaled by the trainer's λ(t) schedule."""

    ALGORITHM_NAME = "nodewise_deterministic_continuous"

    def __init__(self, dc_polarization: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if self.sparsity_loss_mode != "target_size_l2":
            raise ValueError("nodewise_deterministic_continuous requires sparsity_loss_mode: target_size_l2")
        if self.l0_lambda <= 0.0:
            raise ValueError("nodewise_deterministic_continuous needs l0_lambda > 0 (the budget term)")
        if dc_polarization < 0.0:
            raise ValueError(f"dc_polarization must be >= 0, got {dc_polarization}")
        self.dc_polarization = float(dc_polarization)

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        self._theta_ref = log_alpha
        m = torch.sigmoid(log_alpha)
        with torch.no_grad():
            learnable = self._learnable(m.device)
            keep = (m[0] >= 0.5) & learnable
            self._record_membership(keep, int(learnable.sum().item()))
        return {l: m for l in self.layers}

    def _l0(self, log_alpha, granularity, combined_filter=None, beta: float = _HC_BETA):
        learnable = (~combined_filter.bool()).to(log_alpha.dtype)
        m = torch.sigmoid(log_alpha[0])
        n_valid = learnable.sum()
        n_active = (m * learnable).sum()
        target_n = (1.0 - self.target_sparsity) * n_valid
        budget = (n_active - target_n) ** 2 / n_valid
        polar = (m * (1.0 - m) * learnable).sum() / n_valid
        return budget + self.dc_polarization * polar

    def _current_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        with torch.no_grad():
            learnable = self._learnable(log_alpha.device)
            m = torch.sigmoid(log_alpha[0])
            return 1.0 - float(((m >= 0.5) & learnable).sum().item()) / max(1, int(learnable.sum().item()))

    def _expected_sparsity(self, log_alpha, granularity, combined_filter=None, beta=_HC_BETA):
        with torch.no_grad():
            learnable = self._learnable(log_alpha.device).float()
            m = torch.sigmoid(log_alpha[0])
            return 1.0 - float((m * learnable).sum().item()) / max(1.0, float(learnable.sum().item()))

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        node_mask.metadata.update({
            "dc_polarization": self.dc_polarization,
            "mask_parameterisation": "sigmoid(log_alpha)",
            "sparsity_loss_mode": "target_size_l2 on mask values + polarisation",
        })
        return node_mask
