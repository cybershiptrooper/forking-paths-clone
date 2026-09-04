"""Subnetwork probing for sentence-level attention mask discovery (SDPA).

Adapts the method of Cao, Sanh & Rush, "Low-Complexity Probing via Finding
Subnetworks" (2021, https://arxiv.org/abs/2104.03514) to sentence-to-sentence
attention edges. The paper learns a binary pruning mask over model
parameters via the Hard Concrete relaxation of Louizos, Welling & Kingma
(2018, https://arxiv.org/abs/1712.01312). We apply the same relaxation,
but the parameters being pruned are the (layer, head, src_sent, tgt_sent)
attention edges, installed as an SDPA pre-softmax log-additive bias —
the same mechanism used by ``NodewiseAttributionSDPA``.

Training objective:

    L = task_loss(mask-sampled model, branches) + lambda * E[||m||_0]

where the L0 expectation is Hard-Concrete's closed-form edge-active
probability. Gradient descent on the Hard-Concrete location parameters
``log_alpha`` trades off fit against sparsity; edges whose active
probability survives training are those important for the task.

Final score per edge = deterministic (no-noise) Hard-Concrete mean,
which is in [0, 1] and directly interpretable as "keep probability".
Higher = more important. ``negate_scores`` is not applied.
"""

import json
import os
from typing import List, Optional

import torch
from tqdm import tqdm

from utils.wandb_logging import init_wandb_run, log_step, finish_wandb_run

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.utils import Sentence, clear_cuda
from utils.objectives import is_global_objective
from utils.importance_sampling import chain_log_prob, chain_log_prob_chunked
from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.edits.nodewise_attribution_sdpa import (
    _expand_mask_to_log_additive,
)
from utils.circuit_discovery.sdpa_forward import (
    make_sdpa_attention_forward,
)


# Hard Concrete hyperparameters from Louizos et al. 2018 (Table 1).
_HC_BETA = 2.0 / 3.0
_HC_GAMMA = -0.1
_HC_ZETA = 1.1
_HC_EPS = 1e-6


def _hard_concrete_sample(
    log_alpha: torch.Tensor,
    beta: float = _HC_BETA,
    gamma: float = _HC_GAMMA,
    zeta: float = _HC_ZETA,
) -> torch.Tensor:
    """Reparameterised Hard-Concrete sample in [0, 1], differentiable in log_alpha.

    ``beta`` controls the temperature: smaller β → harder (closer to {0,1})
    samples; larger β → softer. Default 2/3 is the Louizos et al. value.
    Anneal β from a high value to a low value over training to widen
    exploration early and sharpen the mask late.
    """
    u = torch.empty_like(log_alpha).uniform_(_HC_EPS, 1.0 - _HC_EPS)
    s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0.0, 1.0)


def _hard_concrete_mean(
    log_alpha: torch.Tensor,
    beta: float = _HC_BETA,
    gamma: float = _HC_GAMMA,
    zeta: float = _HC_ZETA,
) -> torch.Tensor:
    """Deterministic Hard-Concrete readout (noise-free mean), clamped to [0, 1]."""
    s = torch.sigmoid(log_alpha / beta)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0.0, 1.0)


def _hard_concrete_l0_probs(
    log_alpha: torch.Tensor,
    beta: float = _HC_BETA,
    gamma: float = _HC_GAMMA,
    zeta: float = _HC_ZETA,
) -> torch.Tensor:
    """Closed-form P(mask > 0) per entry, same shape as ``log_alpha``.

    Louizos et al. eq. (13): P(z != 0) = sigmoid(log_alpha - beta * log(-gamma/zeta)).
    """
    return torch.sigmoid(
        log_alpha - beta * torch.log(torch.tensor(-gamma / zeta, device=log_alpha.device))
    )


def _hard_concrete_l0_mean(
    log_alpha: torch.Tensor, beta: float = _HC_BETA,
) -> torch.Tensor:
    """Average per-entry active probability — Cao et al. 2021's R(θ) = (1/d) Σ_i ..."""
    return _hard_concrete_l0_probs(log_alpha, beta=beta).mean()


def _hard_concrete_l0_count(
    log_alpha: torch.Tensor, beta: float = _HC_BETA,
) -> torch.Tensor:
    """Expected number of active entries (sum of per-entry probs); for diagnostics."""
    return _hard_concrete_l0_probs(log_alpha, beta=beta).sum()


class NodewiseSubnetworkProbingSDPA(CircuitDiscovery):
    """Hard-Concrete subnetwork probing over sentence-level attention masks.

    Mask parameters ``log_alpha`` are optimised with Adam; each step samples
    a fresh Hard-Concrete mask, installs it into SDPA via the same
    log-additive bias used by ``NodewiseAttributionSDPA``, evaluates the
    task loss across branches, adds an L0 penalty, and backpropagates into
    ``log_alpha``.

    Gradient checkpointing on the model is strongly recommended, identical
    to ``NodewiseAttributionSDPA`` — each training step is a full forward +
    backward through every branch.
    """

    def __init__(
        self,
        num_training_steps: int = 150,
        learning_rate: float = 0.05,
        l0_lambda: float = 1e-3,
        l0_lambda_schedule: bool = True,
        l0_warmup_frac: float = 0.25,
        l0_ramp_frac: float = 0.50,
        log_alpha_init: float = 2.0,
        log_alpha_init_mask_path: Optional[str] = None,
        log_alpha_init_mask_alpha: float = 1.0,
        log_dir: Optional[str] = None,
        log_every: int = 5,
        plot_every: int = 20,
        wandb_project: Optional[str] = "cot_interp",
        wandb_run_name: Optional[str] = None,
        # New flags (defaults preserve current behaviour) ----------------
        sparsity_loss_mode: str = "l0_mean",
        l0_normalize_hinge: bool = False,
        target_sparsity: Optional[float] = None,
        optimizer: str = "adam",
        momentum: float = 0.9,
        l0_lr_multiplier: float = 1.0,
        dropout_p: float = 0.0,
        save_log_alpha: bool = False,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
        resume_from_checkpoint: bool = False,
        # D2 — HC variance reduction ------------------------------------
        num_hc_samples_per_step: int = 1,
        polyak_ema_log_alpha: float = 0.0,
        hc_beta_anneal: bool = False,
        hc_beta_start: float = _HC_BETA,
        hc_beta_end: float = _HC_BETA / 2.0,
        hc_beta_anneal_end_frac: float = 1.0,
        # D3 — LR scheduler --------------------------------------------
        lr_schedule: str = "constant",
        lr_min_ratio: float = 0.1,
        lr_plateau_patience: int = 50,
        lr_plateau_factor: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            num_training_steps: Adam steps over the full branch set.
            learning_rate: Adam learning rate on ``log_alpha``.
            l0_lambda: Max weight on the L0 sparsity penalty (λ_max in Cao et al.).
                With ``l0_lambda_schedule=True`` this is reached after the
                warmup+ramp phases; otherwise it is applied constantly.
            l0_lambda_schedule: If True, apply Cao et al.'s schedule: λ stays
                at 0 for the first ``l0_warmup_frac`` of steps, linearly ramps
                to ``l0_lambda`` over the next ``l0_ramp_frac``, then held.
            l0_warmup_frac: Fraction of training with λ = 0. Default 0.25.
            l0_ramp_frac: Fraction of training over which λ linearly ramps
                from 0 to ``l0_lambda``. Default 0.50.
            log_alpha_init: Initial value for every edge's location parameter.
                NOTE: not the paper's default — Cao et al. do not state an
                initialization but the Louizos convention is 0 (mean ≈ 0.5).
                Default 2.0 here gives noise-free mean ≈ 0.99 (near fully-on)
                so the first step is close to the unmasked model, which works
                well with our short training budget and the reward_gap
                objective but is a deliberate deviation from the paper.
            log_alpha_init: also accepts the string ``"random"``, matching
                ``ColumnSubnetworkProbing``: every gate drawn independently
                from Uniform(-2, 2), which spans hard-closed (log_alpha
                <= -1.6, gate mean 0) to fully open (>= +1.6, gate mean 1).
                The named initializations used by the sentence-grading
                hyperparameter search are plain floats on this scale —
                closed = -3, half = 0 (gate mean 0.5), open = +2 — so no
                extra mode parameter is needed for them.
            log_alpha_init_mask_path: Optional mask JSON. When set, the
                initialization instead mixes that mask with the all-open
                mask: it is binarized by top-k at this run's
                ``target_sparsity`` over the learnable pool (the
                evaluator's rule), and edge ``(i, j)`` starts at gate mean
                ``m = alpha + (1 - alpha) * kept(i, j)``, converted to
                log_alpha by ``_mean_to_log_alpha``. alpha = 1 is all-open
                (identical to ``log_alpha_init = 2.0``); alpha = 0 starts
                exactly at the supplied mask.
            log_alpha_init_mask_alpha: The mixing weight ``alpha`` above.
        """
        self.pair_aggregation = kwargs.pop("pair_aggregation", "mean")
        kwargs.pop("batch_chunk_size", None)
        # IG-specific kwargs passed by the shared learn_circuit.py CLI;
        # accepted and ignored so this method is a drop-in in the factory.
        kwargs.pop("num_ig_steps", None)
        kwargs.pop("negate_scores", None)
        kwargs.pop("include_zero_ablation", None)
        kwargs.pop("zero_ablation_epsilon", None)
        super().__init__(**kwargs)
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.l0_lambda = l0_lambda
        self.l0_lambda_schedule = l0_lambda_schedule
        self.l0_warmup_frac = l0_warmup_frac
        self.l0_ramp_frac = l0_ramp_frac
        self.log_alpha_init = log_alpha_init
        self.log_alpha_init_mask_path = log_alpha_init_mask_path
        self.log_alpha_init_mask_alpha = log_alpha_init_mask_alpha
        # log_dir / plot_every kept for backward-compat; training curves now
        # go to wandb. log_every controls wandb scalar cadence.
        self.log_dir = log_dir
        self.log_every = log_every
        self.plot_every = plot_every
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        if sparsity_loss_mode not in ("l0_mean", "target_size_relu", "target_size_l2"):
            raise ValueError(
                f"sparsity_loss_mode must be 'l0_mean', 'target_size_relu' or "
                f"'target_size_l2', got {sparsity_loss_mode!r}"
            )
        if sparsity_loss_mode in ("target_size_relu", "target_size_l2") and target_sparsity is None:
            raise ValueError(
                f"{sparsity_loss_mode} sparsity loss requires target_sparsity in [0, 1)."
            )
        if optimizer not in ("adam", "sgd", "sgd_momentum", "hybrid"):
            raise ValueError(
                f"optimizer must be one of {{adam, sgd, sgd_momentum, hybrid}}, "
                f"got {optimizer!r}"
            )
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        self.sparsity_loss_mode = sparsity_loss_mode
        self.l0_normalize_hinge = l0_normalize_hinge
        self.target_sparsity = target_sparsity
        self.optimizer = optimizer
        self.momentum = momentum
        self.l0_lr_multiplier = l0_lr_multiplier
        self.dropout_p = dropout_p
        self.save_log_alpha = save_log_alpha
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.resume_from_checkpoint = resume_from_checkpoint
        # D2 — HC variance reduction
        if num_hc_samples_per_step < 1:
            raise ValueError(
                f"num_hc_samples_per_step must be >= 1, got {num_hc_samples_per_step}"
            )
        if not (0.0 <= polyak_ema_log_alpha < 1.0):
            raise ValueError(
                f"polyak_ema_log_alpha must be in [0, 1), got {polyak_ema_log_alpha}"
            )
        self.num_hc_samples_per_step = num_hc_samples_per_step
        self.polyak_ema_log_alpha = polyak_ema_log_alpha
        self.hc_beta_anneal = hc_beta_anneal
        self.hc_beta_start = hc_beta_start
        self.hc_beta_end = hc_beta_end
        if not (0.0 < hc_beta_anneal_end_frac <= 1.0):
            raise ValueError(
                f"hc_beta_anneal_end_frac must be in (0, 1], got "
                f"{hc_beta_anneal_end_frac}"
            )
        self.hc_beta_anneal_end_frac = hc_beta_anneal_end_frac
        # D3 — LR scheduler
        if lr_schedule not in ("constant", "cosine", "linear", "on_plateau"):
            raise ValueError(
                f"lr_schedule must be one of {{constant, cosine, linear, "
                f"on_plateau}}, got {lr_schedule!r}"
            )
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.lr_plateau_patience = lr_plateau_patience
        self.lr_plateau_factor = lr_plateau_factor

    # ------------------------------------------------------------------
    # Patching helpers
    # ------------------------------------------------------------------

    def _sdpa_forward(self):
        return make_sdpa_attention_forward(
            self.model_type, mask_converter=_expand_mask_to_log_additive,
        )

    def _install_masks(self, masks):
        """Overwrite ``_circuit_mask`` on each target attention module in place.

        Called between optimiser steps to avoid re-patching forward every
        iteration (forwards were registered once at the top of ``discover``).
        """
        from utils.utils import get_attention_module
        for layer_idx in self.layers:
            attn = get_attention_module(self.model, layer_idx)
            attn._circuit_mask = masks[layer_idx]

    def _patch_non_target_layers_sdpa(
        self,
        num_sents: int,
        token_to_sent: torch.Tensor,
        gap_filter: torch.Tensor,
    ):
        """Zero-ablate every layer outside ``self.layers`` with SDPA forward."""
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
            zero_mask = torch.zeros(1, num_sents, num_sents, device=device)
            attn_module._circuit_mask = zero_mask
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module._renormalize_masked_attn = self.renormalize_masked_attention
            attn_module.forward = types.MethodType(sdpa_fwd, attn_module)
            handles.append(AblationHandle(attn_module, original_forward))
        return handles

    # ------------------------------------------------------------------
    # log_alpha bookkeeping (granularity-aware)
    # ------------------------------------------------------------------

    def _mean_to_log_alpha(self, m, device):
        """Invert the deterministic Hard-Concrete readout: gate mean -> log_alpha.

        The readout is ``m = clamp(sigmoid(la / beta) * (zeta - gamma) + gamma, 0, 1)``,
        so for an interior mean the inverse is
        ``la = beta * logit((m - gamma) / (zeta - gamma))``. The readout
        saturates outside ``[gamma, zeta]``: every ``la`` above
        ``beta * logit((1 - gamma) / (zeta - gamma))`` gives m = 1 exactly.
        We therefore map the endpoints to +/- ``log_alpha_init`` so that
        m = 1 reproduces the canonical fully-open initialization rather
        than sitting at the smallest log_alpha that happens to saturate.
        """
        m = torch.as_tensor(m, dtype=torch.float32, device=device)
        mag = abs(self.log_alpha_init)
        beta = (self.hc_beta_start if getattr(self, "hc_beta_anneal", False)
                else _HC_BETA)
        s = (m - _HC_GAMMA) / (_HC_ZETA - _HC_GAMMA)
        s = s.clamp(1e-6, 1 - 1e-6)
        la = beta * (torch.log(s) - torch.log1p(-s))
        la = la.clamp(-mag, mag)
        la = torch.where(m >= 1.0, torch.full_like(la, mag), la)
        la = torch.where(m <= 0.0, torch.full_like(la, -mag), la)
        return la

    def _init_means(self, num_sents, device, combined_filter=None):
        """Per-edge initial gate means when initializing from a mask.

        Returns an (S, S) tensor of means in [0, 1], or None when no mask
        was supplied (the float / "random" paths handle those).
        """
        if not self.log_alpha_init_mask_path:
            return None
        if self.target_sparsity is None:
            raise ValueError(
                "log_alpha_init_mask_path requires target_sparsity (the mask "
                "is binarized by top-k at that sparsity)"
            )
        kept = self._load_mask_topk(
            self.log_alpha_init_mask_path, num_sents, device, combined_filter,
        )
        a = float(self.log_alpha_init_mask_alpha)
        return a + (1.0 - a) * kept.to(torch.float32)

    def _load_mask_topk(self, path, num_sents, device, combined_filter):
        """Binary (S, S) top-k selection of a saved mask at target_sparsity.

        Same rule as the matched-target evaluator
        (``expts/direct_answer_circuit_discovery/eval_log_alpha.py``):
        keep the ``round((1 - s) * n_valid)`` highest-scoring learnable
        edges.
        """
        with open(path) as f:
            d = json.load(f)
        scores = torch.tensor(d["scores"], dtype=torch.float32, device=device)
        if scores.shape != (num_sents, num_sents):
            raise ValueError(
                f"init mask {path} has shape {tuple(scores.shape)}, expected "
                f"{(num_sents, num_sents)}"
            )
        readout = d.get("metadata", {}).get("score_readout")
        if readout not in ("log_alpha", "raw_score") and (
            float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
        ):
            # Hard-Concrete means: invert so the ranking matches the evaluator.
            ss = ((scores - _HC_GAMMA) / (_HC_ZETA - _HC_GAMMA)).clamp(1e-6, 1 - 1e-6)
            scores = _HC_BETA * (torch.log(ss) - torch.log1p(-ss))
        valid = (
            torch.ones(num_sents, num_sents, dtype=torch.bool, device=device)
            if combined_filter is None
            else ~combined_filter.bool()
        )
        n_valid = int(valid.sum())
        n_keep = max(0, int(round((1.0 - float(self.target_sparsity)) * n_valid)))
        flat = torch.where(
            valid.flatten(), scores.flatten(),
            torch.full_like(scores.flatten(), float("-inf")),
        )
        keep = torch.zeros_like(flat, dtype=torch.bool)
        if n_keep > 0:
            keep[torch.topk(flat, n_keep).indices] = True
        print(
            f"  init mask: kept {int(keep.sum())} of {n_valid} learnable edges "
            f"from {os.path.basename(path)} at target sparsity "
            f"{float(self.target_sparsity):g}"
        )
        return keep.view(num_sents, num_sents)

    def _init_log_alpha(self, granularity, num_heads, num_sents, device,
                        combined_filter=None):
        means = self._init_means(num_sents, device, combined_filter)
        if means is not None:
            base = self._mean_to_log_alpha(means, device)
        elif isinstance(self.log_alpha_init, str):
            # Same convention as ColumnSubnetworkProbing.
            if self.log_alpha_init != "random":
                raise ValueError(
                    f"log_alpha_init must be a float or 'random', "
                    f"got {self.log_alpha_init!r}"
                )
            base = torch.empty(
                (num_sents, num_sents), device=device, dtype=torch.float32,
            ).uniform_(-2.0, 2.0)
        else:
            base = torch.full(
                (num_sents, num_sents), float(self.log_alpha_init),
                device=device, dtype=torch.float32,
            )
        if granularity == "head":
            return {
                l: base.unsqueeze(0).expand(num_heads, -1, -1)
                    .clone().requires_grad_(True)
                for l in self.layers
            }
        if granularity == "layer":
            return {
                l: base.unsqueeze(0).clone().requires_grad_(True)
                for l in self.layers
            }
        return base.unsqueeze(0).clone().requires_grad_(True)

    def _params_as_list(self, log_alpha, granularity):
        if isinstance(log_alpha, torch.Tensor):
            return [log_alpha]
        return list(log_alpha.values())

    def _current_beta(self, step: int) -> float:
        """Annealed Hard-Concrete temperature for this training step.

        Linear interpolation from ``hc_beta_start`` at step 0 to
        ``hc_beta_end``, reached after ``hc_beta_anneal_end_frac`` of
        training (default 1.0 = the final step) and held there. Ending the
        anneal early leaves a hardened-gate phase in which the mask is
        optimized at its final temperature. Returns the constant
        ``_HC_BETA`` when annealing is disabled.
        """
        if not self.hc_beta_anneal:
            return _HC_BETA
        total = max(1, self.num_training_steps - 1)
        anneal_steps = max(1.0, self.hc_beta_anneal_end_frac * total)
        frac = min(1.0, max(0.0, step / anneal_steps))
        return self.hc_beta_start + (self.hc_beta_end - self.hc_beta_start) * frac

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        if isinstance(log_alpha, torch.Tensor):
            sampled = _hard_concrete_sample(log_alpha, beta=beta)
            sampled = self._apply_dropout(sampled)
            return {l: sampled for l in self.layers}
        return {
            l: self._apply_dropout(_hard_concrete_sample(log_alpha[l], beta=beta))
            for l in self.layers
        }

    def _apply_dropout(self, sampled: torch.Tensor) -> torch.Tensor:
        """Bernoulli dropout on the sampled mask during training.

        With probability ``dropout_p`` per entry, force the sampled mask to 0
        for this training step (extra regulariser; AutoCircuit-style).
        Active only when ``dropout_p > 0``. Surviving entries are *not*
        rescaled — we want the L0 budget to reflect "active in expectation",
        not an inflated value.
        """
        if self.dropout_p <= 0.0:
            return sampled
        keep = (torch.rand_like(sampled) >= self.dropout_p).float()
        return sampled * keep

    def _ones_masks(self, granularity, num_heads, num_sents, device):
        """Deterministic all-ones mask per target layer.

        Used for clean-logits reference so KL target isn't perturbed by the
        stochastic HC sample at initialization. Shape matches ``_sample_masks``.
        """
        if granularity == "head":
            shape = (num_heads, num_sents, num_sents)
        else:
            shape = (1, num_sents, num_sents)
        ones = torch.ones(shape, device=device)
        return {l: ones for l in self.layers}

    def _l0(self, log_alpha, granularity, combined_filter=None, beta: float = _HC_BETA):
        """Sparsity loss in one of three modes (see also ``target_size_l2``
        in the body: two-sided quadratic toward the target size).

        - ``"l0_mean"`` (default, matches Cao et al.): mean per-entry active
          probability across the parameter tensor(s). Filter-agnostic for
          backward compatibility with prior runs.
        - ``"target_size_relu"``: ``ReLU(n_active − target_n)``, where the
          counts are taken **only over filter-valid entries** so that
          ``target_sparsity`` is interpretable as "fraction of *valid* edges
          to prune" (gap/causal/mode-filtered cells are not counted).
          ``λ`` then scales how hard we push *down* once the budget is
          exceeded; below budget, gradient is zero.

        ``beta`` must be the same Hard-Concrete temperature used for
        sampling this step, so the budget measures the distribution
        actually being sampled (matters when ``hc_beta_anneal`` is on).

        With ``l0_normalize_hinge=True`` the hinge is divided by the
        detached excess (in cells, floored at 1), so the *total* per-step
        crush force is bounded at ~λ regardless of how far over budget the
        mask is: per-cell force ≈ λ·p′/excess instead of λ·p′ for every
        over-budget cell simultaneously.
        """
        if self.sparsity_loss_mode == "l0_mean":
            if isinstance(log_alpha, torch.Tensor):
                return _hard_concrete_l0_mean(log_alpha, beta=beta)
            return sum(
                _hard_concrete_l0_mean(log_alpha[l], beta=beta) for l in self.layers
            ) / len(self.layers)

        # target_size_relu — filter-aware count, ReLU hinge against budget.
        # NOTE: ``combined_filter == True`` means *frozen* at 1.0 (gap/mode/
        # causal-excluded). Learnable / valid entries are the *complement*.
        assert combined_filter is not None, (
            "target_size_relu requires combined_filter for valid-edge counting"
        )
        learnable_2d = (~combined_filter.bool()).to(torch.float32)
        if isinstance(log_alpha, torch.Tensor):
            probs = _hard_concrete_l0_probs(log_alpha, beta=beta)
            valid = learnable_2d.unsqueeze(0).to(probs.dtype)
            n_active = (probs * valid).sum()
            n_valid = valid.sum()
        else:
            n_active = log_alpha[self.layers[0]].new_zeros(())
            n_valid = log_alpha[self.layers[0]].new_zeros(())
            for l in self.layers:
                probs = _hard_concrete_l0_probs(log_alpha[l], beta=beta)
                valid = learnable_2d.unsqueeze(0).to(probs.dtype)
                n_active = n_active + (probs * valid).sum()
                n_valid = n_valid + valid.sum() * probs.shape[0]
        target_n = (1.0 - self.target_sparsity) * n_valid
        if self.sparsity_loss_mode == "target_size_l2":
            # Two-sided quadratic: penalises being sparser than the target
            # as well as denser, so the mask has a restoring force toward
            # landing AT the target (the one-sided hinge lets it drift
            # past). Normalised by n_valid so the per-cell push is
            # λ·2·(excess fraction)·p′ — the same scale as the hinge's
            # λ·p′ at 50% excess, vanishing smoothly at the target.
            return (n_active - target_n) ** 2 / n_valid
        hinge = torch.relu(n_active - target_n)
        if self.l0_normalize_hinge:
            excess = (n_active - target_n).detach().clamp(min=1.0)
            hinge = hinge / excess
        return hinge

    def _lambda_at_step(self, step: int) -> float:
        """Cao et al.'s L0 schedule: 0 → λ_max ramp, then held.

        Without this schedule the mask gets penalised hard from step 0 and
        often collapses before it has explored useful configurations.
        """
        if not self.l0_lambda_schedule:
            return self.l0_lambda
        total = max(1, self.num_training_steps - 1)
        frac = step / total
        if frac < self.l0_warmup_frac:
            return 0.0
        ramp_end = self.l0_warmup_frac + self.l0_ramp_frac
        if frac < ramp_end:
            return self.l0_lambda * (frac - self.l0_warmup_frac) / self.l0_ramp_frac
        return self.l0_lambda

    def _update_lr(self, optim, step: int, total_steps: int) -> float:
        """Apply LR schedule for this step; return the resulting lr.

        ``constant``: returns ``self.learning_rate`` unchanged.
        ``cosine``: half-cosine decay from ``learning_rate`` down to
        ``learning_rate * lr_min_ratio`` over ``total_steps``.
        ``linear``: linear decay over the same range.
        ``on_plateau``: handled separately via ReduceLROnPlateau; this
        function returns the optimiser's current lr without modifying it.
        """
        if self.lr_schedule == "constant":
            return self.learning_rate
        if self.lr_schedule == "on_plateau":
            return float(optim.param_groups[0]["lr"])
        progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
        lr_max = self.learning_rate
        lr_min = self.learning_rate * self.lr_min_ratio
        if self.lr_schedule == "cosine":
            import math
            new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        elif self.lr_schedule == "linear":
            new_lr = lr_max + (lr_min - lr_max) * progress
        else:
            new_lr = lr_max
        for pg in optim.param_groups:
            pg["lr"] = new_lr
        return float(new_lr)

    def _build_optimizer(self, params):
        """Build optimiser per ``self.optimizer``.

        ``hybrid``: returns the Adam used for the *task* gradient. The L0
        update is applied manually outside this optimiser so its magnitude
        is not adaptively normalised away (the entire reason this mode
        exists).
        """
        if self.optimizer == "adam" or self.optimizer == "hybrid":
            return torch.optim.Adam(params, lr=self.learning_rate)
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate)
        if self.optimizer == "sgd_momentum":
            return torch.optim.SGD(
                params, lr=self.learning_rate, momentum=self.momentum,
            )
        raise ValueError(f"Unknown optimizer {self.optimizer!r}")

    def _save_checkpoint(self, path, step, log_alpha, granularity, optim,
                         ema_log_alpha=None):
        """Save full training state for resume-and-extend."""
        if isinstance(log_alpha, torch.Tensor):
            la_state = {"_tensor": log_alpha.detach().cpu()}
        else:
            la_state = {str(l): log_alpha[l].detach().cpu() for l in self.layers}
        ema_state = None
        if ema_log_alpha is not None:
            if isinstance(ema_log_alpha, torch.Tensor):
                ema_state = {"_tensor": ema_log_alpha.detach().cpu()}
            else:
                ema_state = {
                    str(l): ema_log_alpha[l].detach().cpu() for l in self.layers
                }
        state = {
            "step": step,
            "granularity": granularity,
            "log_alpha": la_state,
            "ema_log_alpha": ema_state,
            "optimizer": optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic-ish write to avoid corrupted files on preemption.
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def _load_checkpoint(self, path, log_alpha, granularity, optim, device,
                         ema_log_alpha=None):
        """Load checkpoint into existing log_alpha + optimizer; return start_step.

        Loaded with ``map_location='cpu'`` so that RNG-state byte tensors stay
        on CPU (``set_rng_state`` requires CPU ByteTensors); we then move
        only the parameter tensors to the target device.

        If ``ema_log_alpha`` is passed and the checkpoint holds an EMA state,
        the EMA tensors are restored in place too (older checkpoints without
        the key leave the EMA at its re-initialized value).
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["granularity"] != granularity:
            raise ValueError(
                f"checkpoint granularity={state['granularity']} != "
                f"current granularity={granularity}"
            )
        if isinstance(log_alpha, torch.Tensor):
            la_key = "_tensor" if "_tensor" in state["log_alpha"] else "pair"
            with torch.no_grad():
                log_alpha.copy_(state["log_alpha"][la_key].to(device))
        else:
            for l in self.layers:
                with torch.no_grad():
                    log_alpha[l].copy_(state["log_alpha"][str(l)].to(device))
        ema_state = state.get("ema_log_alpha")
        if ema_log_alpha is not None and ema_state is not None:
            if isinstance(ema_log_alpha, torch.Tensor):
                ema_key = "_tensor" if "_tensor" in ema_state else "pair"
                with torch.no_grad():
                    ema_log_alpha.copy_(ema_state[ema_key].to(device))
            else:
                for l in self.layers:
                    with torch.no_grad():
                        ema_log_alpha[l].copy_(ema_state[str(l)].to(device))
        optim.load_state_dict(state["optimizer"])
        # Optim state tensors may need a device move (Adam's exp_avg etc).
        for st in optim.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
            except Exception as e:
                print(f"  (could not restore CUDA RNG state: {e})")
        return int(state["step"])

    def _expected_active_count(self, log_alpha, granularity) -> float:
        """Interpretable diagnostic: expected number of active edges."""
        with torch.no_grad():
            if isinstance(log_alpha, torch.Tensor):
                return float(
                    (_hard_concrete_l0_count(log_alpha)).item()
                )
            return float(
                sum(_hard_concrete_l0_count(log_alpha[l]) for l in self.layers).item()
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

        forward_fn = self._sdpa_forward()
        granularity = self.mask_granularity

        # ----- Init learnable parameters -----
        log_alpha = self._init_log_alpha(
            granularity, num_heads, num_sents, device,
            combined_filter=combined_filter,
        )
        params = self._params_as_list(log_alpha, granularity)
        optim = self._build_optimizer(params)

        # ----- Optional Polyak EMA on log_alpha (D2) -----
        # Maintain a smoothed copy of log_alpha that's updated each step.
        # When polyak_ema_log_alpha > 0, the readout uses the EMA copy
        # rather than the noisy live log_alpha — empirically reduces
        # variance from late-training HC sampling jitter. Created before
        # the resume path so checkpoints can restore it.
        ema_log_alpha = None
        if self.polyak_ema_log_alpha > 0.0:
            with torch.no_grad():
                if isinstance(log_alpha, torch.Tensor):
                    ema_log_alpha = log_alpha.detach().clone()
                else:
                    ema_log_alpha = {
                        l: log_alpha[l].detach().clone() for l in self.layers
                    }

        start_step = 0
        if self.resume_from_checkpoint and self.checkpoint_path is not None \
                and os.path.exists(self.checkpoint_path):
            print(f"  Resuming from checkpoint: {self.checkpoint_path}")
            start_step = self._load_checkpoint(
                self.checkpoint_path, log_alpha, granularity, optim, device,
                ema_log_alpha=ema_log_alpha,
            )
            print(f"    -> resumed at step {start_step}")

        # ----- Patch target layers; use deterministic all-ones for clean logits -----
        # Clean-logits reference must be fully kept on target layers (not a
        # stochastic HC sample), else the KL target is perturbed by ~log(init_mean)
        # per edge. We install a deterministic all-ones mask to compute clean
        # logits, then swap in the stochastic init sample before training starts.
        ones_masks = self._ones_masks(granularity, num_heads, num_sents, device)
        handles = self._patch_model(ones_masks, token_to_sent, combined_filter, forward_fn)

        non_target_handles = []
        if self.ablate_non_target_layers:
            print(
                f"Ablating all layers outside {self.layers} "
                f"({self.model.config.num_hidden_layers - len(self.layers)} layers)..."
            )
            non_target_handles = self._patch_non_target_layers_sdpa(
                num_sents=num_sents,
                token_to_sent=token_to_sent,
                gap_filter=combined_filter,
            )

        # ----- Objective setup -----
        objective_name = getattr(self.objective_fn, "__name__", "unknown")
        use_global = is_global_objective(objective_name)
        answer_ids = kwargs.get("answer_ids")
        num_answers = kwargs.get("num_answers")
        if use_global and (answer_ids is None or num_answers is None):
            raise ValueError(
                f"Global objective '{objective_name}' requires answer_ids and "
                f"num_answers."
            )

        print("Computing clean logits (target all-ones, non-target zero-ablated)...")
        # Local objectives still need the full (seq, V) clean-logits cache.
        # Global objectives only need the per-chain scalar log-probs, which
        # we compute directly via chain_log_prob_chunked from hidden states —
        # avoiding the (seq, V) materialisation entirely for global runs.
        if use_global:
            clean_logits_list = None
            chain_logprobs_clean = []
            lm_head_weight = self.model.lm_head.weight
            lm_head_bias = getattr(self.model.lm_head, "bias", None)
            self.model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for cont in continuations:
                    full_input = torch.cat([input_ids, cont], dim=-1)
                    hidden = self.model.model(full_input).last_hidden_state
                    lp = chain_log_prob_chunked(
                        hidden, lm_head_weight, full_input, prefix_len,
                        temperature=self.temperature, lm_head_bias=lm_head_bias,
                    )
                    chain_logprobs_clean.append(lp.detach())
                    del hidden
            chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)
        else:
            clean_logits_list = self._get_clean_logits(input_ids, continuations)
            chain_logprobs_clean = None

        # Swap in the initial stochastic HC sample for training.
        # Uses the step-0 annealed temperature so sampling and the L0
        # budget agree from the first step (hc_beta_anneal fix).
        init_masks = self._sample_masks(
            log_alpha, granularity, beta=self._current_beta(start_step),
        )
        self._install_masks(init_masks)

        # ----- Optional LR scheduler (D3) -----
        # constant: no-op; cosine / linear: per-step lr update on Adam's
        # param_group; on_plateau: ReduceLROnPlateau driven by smoothed
        # task_loss. Hybrid mode also reads ``current_lr`` for its manual
        # SGD step (so the L0 push tracks Adam's lr decay too).
        plateau_sched = None
        plateau_loss_buf = []
        if self.lr_schedule == "on_plateau":
            plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=self.lr_plateau_factor,
                patience=self.lr_plateau_patience,
            )

        chain_lengths = torch.tensor(
            [c.shape[-1] for c in continuations], dtype=torch.long, device=device,
        )

        sched_desc = (
            f"schedule: warmup={self.l0_warmup_frac:.2f} + ramp={self.l0_ramp_frac:.2f}"
            if self.l0_lambda_schedule
            else "constant (no schedule)"
        )
        print(
            f"Running subnetwork probing "
            f"({self.num_training_steps} steps, lr={self.learning_rate}, "
            f"l0_lambda={self.l0_lambda} [{sched_desc}], "
            f"{len(continuations)} continuations, granularity={granularity})..."
        )

        # ----- Training loop -----
        run_name = self.wandb_run_name
        if run_name is None and self.log_dir is not None:
            # Re-use the per-run subdir name as the wandb run name so the
            # mapping from a saved NodeMask to its training curves is obvious.
            run_name = os.path.basename(os.path.normpath(self.log_dir))
        wandb_run = init_wandb_run(
            project=self.wandb_project,
            run_name=run_name,
            config={
                "algorithm": "nodewise_subnetwork_probing_sdpa",
                "num_training_steps": self.num_training_steps,
                "learning_rate": self.learning_rate,
                "l0_lambda": self.l0_lambda,
                "l0_lambda_schedule": self.l0_lambda_schedule,
                "l0_warmup_frac": self.l0_warmup_frac,
                "l0_ramp_frac": self.l0_ramp_frac,
                "log_alpha_init": self.log_alpha_init,
                "log_alpha_init_mask_path": self.log_alpha_init_mask_path,
                "log_alpha_init_mask_alpha": self.log_alpha_init_mask_alpha,
                "mask_granularity": granularity,
                "num_continuations": len(continuations),
                "num_layers": len(self.layers),
                "num_heads": num_heads,
                "num_sentences": num_sents,
                "objective": objective_name,
                "mask_mode": mask_mode,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
            },
        )

        if start_step >= self.num_training_steps:
            print(
                f"  Checkpoint already at step {start_step} >= "
                f"num_training_steps={self.num_training_steps}; skipping training."
            )
        # HF gradient checkpointing only engages in train mode
        # (GradientCheckpointingLayer gates on `self.training`); without this
        # the flag is a silent no-op and long-context 32B training OOMs. The
        # model has no dropout, so train() is numerically identical here.
        if getattr(self.model, "is_gradient_checkpointing", False):
            self.model.train()
        # Gross node-flip tracking: per training step, count learnable cells
        # whose deterministic clamped-to-zero status changed (on→off and
        # off→on separately), accumulated between logged steps. The scalar
        # `sparsity` series only recovers the NET change per interval; these
        # two counters give the gross flux.
        _learnable_flat = (~combined_filter.bool()).flatten()
        _prev_clamped = None
        _flips_off_accum = 0
        _flips_on_accum = 0
        for step in tqdm(
            range(start_step, self.num_training_steps),
            desc="SNP steps", initial=start_step, total=self.num_training_steps,
        ):
            optim.zero_grad(set_to_none=True)
            beta = self._current_beta(step)
            K = self.num_hc_samples_per_step
            # K-sample variance reduction: average gradient over K fresh HC
            # samples before stepping. Each inner iteration scales its
            # backward grad by 1/K so the accumulated log_alpha.grad equals
            # the mean over samples (matches a single-sample step in
            # magnitude; just lower variance).
            task_loss_val_accum = 0.0
            for k_idx in range(K):
                sampled = self._sample_masks(log_alpha, granularity, beta=beta)
                # Decouple model graph from log_alpha graph (see comment
                # in single-sample version below).
                unique_leaves = {}
                sampled_leaf = {}
                for k_key, v in sampled.items():
                    leaf = unique_leaves.get(id(v))
                    if leaf is None:
                        leaf = v.detach().requires_grad_(True)
                        unique_leaves[id(v)] = leaf
                    sampled_leaf[k_key] = leaf
                self._install_masks(sampled_leaf)

                if use_global:
                    task_loss_val_k = self._step_global(
                        input_ids, continuations, prefix_len, device,
                        chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
                    )
                else:
                    task_loss_val_k = self._step_local(
                        input_ids, continuations, clean_logits_list,
                        prefix_len, device, branch_rewards, position_mask_overrides,
                    )
                task_loss_val_accum += task_loss_val_k / K

                # Push accumulated leaf grads through m_sample → log_alpha,
                # scaled by 1/K so the K-sample average lands on log_alpha.grad.
                scale = 1.0 / K
                for orig_id, leaf in unique_leaves.items():
                    if leaf.grad is None:
                        continue
                    orig = next(t for t in sampled.values() if id(t) == orig_id)
                    orig.backward(gradient=leaf.grad * scale)
                # Discard model graph + leaf state for this sample before the
                # next one. Python will GC; we just drop the references.
                del sampled, sampled_leaf, unique_leaves
            task_loss_val = task_loss_val_accum

            # ----- Task-gradient diagnostics -----
            # At this point .grad on the params holds ONLY the task-term
            # gradient (the L0 penalty is applied below: manually for the
            # hybrid optimizer, via backward for the others), so its norm
            # and step-to-step direction stability measure whether the task
            # objective is still supplying signal — the saturation check.
            with torch.no_grad():
                grad_parts = [
                    p.grad.flatten() for p in params if p.grad is not None
                ]
                if grad_parts:
                    task_grad_flat = torch.cat(grad_parts)
                    task_grad_norm = float(task_grad_flat.norm().item())
                    prev = getattr(self, "_prev_task_grad_flat", None)
                    if prev is not None and prev.numel() == task_grad_flat.numel():
                        denom = float(prev.norm().item()) * task_grad_norm
                        task_grad_cosine = (
                            float((prev @ task_grad_flat).item() / denom)
                            if denom > 0.0 else None
                        )
                    else:
                        task_grad_cosine = None
                    self._prev_task_grad_flat = task_grad_flat.detach().clone()
                else:
                    task_grad_norm = 0.0
                    task_grad_cosine = None

            # ----- LR schedule update (D3) -----
            # Refreshes Adam's lr in-place; the manual L0 SGD below reads
            # the same `current_lr` value so both stay in sync.
            current_lr = self._update_lr(
                optim, step, self.num_training_steps,
            )

            # Sparsity penalty (independent of the task forward), scheduled λ.
            lam = self._lambda_at_step(step)
            l0_val = 0.0
            if lam > 0.0:
                l0 = lam * self._l0(log_alpha, granularity, combined_filter, beta=beta)
                l0_val = float(l0.detach().item())
                if self.optimizer == "hybrid":
                    # Hybrid: Adam handles task gradient only. Apply the L0
                    # gradient *manually* outside Adam so |λ| is not
                    # adaptively normalised away. The Adam .step() below
                    # only sees the task gradient already accumulated on
                    # the leaves; we then take a vanilla SGD step on
                    # log_alpha using the L0 gradient computed here.
                    l0_grads = torch.autograd.grad(
                        l0,
                        params,
                        retain_graph=False,
                        allow_unused=True,
                    )
                else:
                    l0.backward()
                    l0_grads = None
            else:
                l0_grads = None

            optim.step()

            if l0_grads is not None:
                # Manual SGD on L0 grad; lr scaled by l0_lr_multiplier so we
                # can independently tune the L0 push without disturbing Adam.
                # ``current_lr`` already reflects the LR schedule.
                with torch.no_grad():
                    l0_lr = current_lr * self.l0_lr_multiplier
                    for p, g in zip(params, l0_grads):
                        if g is None:
                            continue
                        p.add_(g, alpha=-l0_lr)

            # ----- Polyak EMA on log_alpha (D2) -----
            if ema_log_alpha is not None:
                ema_decay = self.polyak_ema_log_alpha
                with torch.no_grad():
                    if isinstance(ema_log_alpha, torch.Tensor):
                        ema_log_alpha.mul_(ema_decay).add_(
                            log_alpha.detach(), alpha=1.0 - ema_decay,
                        )
                    else:
                        for l in self.layers:
                            ema_log_alpha[l].mul_(ema_decay).add_(
                                log_alpha[l].detach(), alpha=1.0 - ema_decay,
                            )

            # ----- Gross flip counters (see init above the loop) -----
            with torch.no_grad():
                if isinstance(log_alpha, torch.Tensor):
                    m_now = _hard_concrete_mean(log_alpha, beta=beta).flatten()
                    lrn = _learnable_flat.to(m_now.device)
                    n_rep = m_now.numel() // lrn.numel() if lrn.numel() else 1
                    lrn = lrn.repeat(max(1, n_rep))[:m_now.numel()]
                else:
                    m_parts = [
                        _hard_concrete_mean(log_alpha[l], beta=beta)
                        for l in self.layers
                    ]
                    m_now = torch.cat([m.flatten() for m in m_parts])
                    lrn_2d = _learnable_flat.to(m_now.device)
                    lrn = torch.cat([
                        lrn_2d.view(*m_parts[0].shape[-2:]).unsqueeze(0)
                        .expand_as(m).flatten()
                        for m in m_parts
                    ])
                clamped_now = (m_now == 0.0) & lrn
                if _prev_clamped is not None:
                    _flips_off_accum += int((clamped_now & ~_prev_clamped).sum().item())
                    _flips_on_accum += int((~clamped_now & _prev_clamped).sum().item())
                _prev_clamped = clamped_now

            # ReduceLROnPlateau is metric-driven; step it on smoothed task_loss.
            if plateau_sched is not None:
                plateau_loss_buf.append(task_loss_val)
                if len(plateau_loss_buf) > 10:
                    plateau_loss_buf.pop(0)
                plateau_sched.step(sum(plateau_loss_buf) / len(plateau_loss_buf))

            if (
                self.checkpoint_path is not None
                and self.checkpoint_every > 0
                and ((step + 1) % self.checkpoint_every == 0
                     or (step + 1) == self.num_training_steps)
            ):
                self._save_checkpoint(
                    self.checkpoint_path, step + 1, log_alpha, granularity, optim,
                    ema_log_alpha=ema_log_alpha,
                )

            if (step % self.log_every == 0) or (step + 1 == self.num_training_steps):
                sparsity_val = self._current_sparsity(
                    log_alpha, granularity, combined_filter, beta=beta,
                )
                # Budget-side view of the same quantity: expected active
                # count under the closed-form P(z > 0) at the current β,
                # restricted to learnable cells — the value the
                # target_size_relu hinge actually constrains. Logged so
                # per-step node add/remove analyses can compare the two.
                sparsity_expected = self._expected_sparsity(
                    log_alpha, granularity, combined_filter, beta=beta,
                )
                metrics = {
                    "step": step,
                    "task_loss": task_loss_val,
                    "l0_loss": l0_val,
                    "sparsity": sparsity_val,
                    "sparsity_expected": sparsity_expected,
                    "flips_off_interval": _flips_off_accum,
                    "flips_on_interval": _flips_on_accum,
                    "l0_lambda": lam,
                    "lr": current_lr,
                    "hc_beta": beta,
                    "num_hc_samples_per_step": K,
                    "task_grad_norm": task_grad_norm,
                    "task_grad_cosine": task_grad_cosine,
                }
                _flips_off_accum = 0
                _flips_on_accum = 0
                # Objective-specific saturation stats (ESS, softmax entropy,
                # pair-saturation fraction, hazard levels, ...) written by
                # the loss function on its last call this step.
                from utils import objectives as _objectives_mod
                metrics.update({
                    f"diag_{k}": v
                    for k, v in _objectives_mod.LAST_DIAGNOSTICS.items()
                })
                # Per-chain-weight stats from the global two-pass step.
                metrics.update(getattr(self, "_last_global_diag", {}))
                if wandb_run is not None:
                    log_step(wandb_run, step=step, metrics={k: v for k, v in metrics.items() if k != "step"})
                # Local JSONL fallback so training curves survive wandb outages
                # and can be replotted offline. Writes one line per logged step.
                if self.log_dir is not None:
                    os.makedirs(self.log_dir, exist_ok=True)
                    jl_path = os.path.join(self.log_dir, "training_metrics.jsonl")
                    with open(jl_path, "a") as _jlf:
                        _jlf.write(json.dumps(metrics) + "\n")

        finish_wandb_run(wandb_run)
        self._unpatch_model(handles)
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # ----- Readout -----
        # Default: deterministic HC mean ``m`` in [0, 1] (back-compat).
        # If ``save_log_alpha`` is True, save raw log_alpha values instead;
        # eval-time recovers any of {m, m>0, m>0.5, top-K} from log_alpha.
        # When Polyak EMA is enabled (D2), the readout source is the EMA
        # tensor instead of the noisy live ``log_alpha``.
        if ema_log_alpha is not None:
            readout_source = ema_log_alpha
        else:
            readout_source = log_alpha

        # The readout (and the saved hard_concrete_beta below) use the β in
        # effect at the *final* step, so eval-side threshold constants and
        # HC-mean inversion match the trained distribution when
        # hc_beta_anneal is on.
        final_beta = self._current_beta(self.num_training_steps - 1)
        if self.save_log_alpha:
            readout_fn = lambda t: t.detach().float().cpu()
            score_readout_kind = "log_alpha"
        else:
            readout_fn = (
                lambda t: _hard_concrete_mean(t, beta=final_beta).detach().cpu()
            )
            score_readout_kind = "hard_concrete_mean"

        scores = self._score_readout(
            readout_source, readout_fn, granularity, num_heads,
        )

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_subnetwork_probing_sdpa",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name=objective_name,
            metadata={
                "num_training_steps": self.num_training_steps,
                "learning_rate": self.learning_rate,
                "l0_lambda": self.l0_lambda,
                "l0_lambda_schedule": self.l0_lambda_schedule,
                "l0_warmup_frac": self.l0_warmup_frac,
                "l0_ramp_frac": self.l0_ramp_frac,
                "l0_normalization": "mean",
                "sparsity_loss_mode": self.sparsity_loss_mode,
                "l0_normalize_hinge": self.l0_normalize_hinge,
                "target_sparsity": self.target_sparsity,
                "optimizer": self.optimizer,
                "momentum": self.momentum,
                "l0_lr_multiplier": self.l0_lr_multiplier,
                "dropout_p": self.dropout_p,
                "save_log_alpha": self.save_log_alpha,
                "log_alpha_init": self.log_alpha_init,
                "log_alpha_init_mask_path": self.log_alpha_init_mask_path,
                "log_alpha_init_mask_alpha": self.log_alpha_init_mask_alpha,
                "hard_concrete_beta": final_beta,
                "hard_concrete_gamma": _HC_GAMMA,
                "hard_concrete_zeta": _HC_ZETA,
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "num_frozen_prompt_sentences": num_frozen_prompt_sentences,
                "pair_aggregation": self.pair_aggregation,
                "mask_granularity": granularity,
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
                "attention_backend": "sdpa",
                "score_readout": score_readout_kind,
                "num_hc_samples_per_step": self.num_hc_samples_per_step,
                "polyak_ema_log_alpha": self.polyak_ema_log_alpha,
                "hc_beta_anneal": self.hc_beta_anneal,
                "hc_beta_start": self.hc_beta_start,
                "hc_beta_end": self.hc_beta_end,
                "hc_beta_anneal_end_frac": self.hc_beta_anneal_end_frac,
                "lr_schedule": self.lr_schedule,
                "lr_min_ratio": self.lr_min_ratio,
            },
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Score readout
    # ------------------------------------------------------------------

    def _score_readout(self, readout_source, readout_fn, granularity, num_heads):
        """Convert trained log_alpha (or EMA) into serializable scores.

        Override in subclasses to change the readout shape (e.g. column
        granularity saves a 1D list instead of 2D).
        """
        with torch.no_grad():
            if granularity == "pair":
                r = readout_fn(readout_source)
                return r[0].tolist()
            elif granularity == "layer":
                scores = {}
                for l in self.layers:
                    r = readout_fn(readout_source[l])
                    scores[l] = r[0].tolist()
                return scores
            else:  # head
                scores = {}
                for l in self.layers:
                    r = readout_fn(readout_source[l])
                    scores[l] = {h: r[h].tolist() for h in range(num_heads)}
                return scores

    # ------------------------------------------------------------------
    # Logging / plotting
    # ------------------------------------------------------------------

    def _expected_sparsity(
        self, log_alpha, granularity, combined_filter=None, beta: float = _HC_BETA,
    ) -> float:
        """1 − (Σ P(z > 0) over learnable cells) / n_learnable at temperature β.

        The budget-side sparsity: the quantity ``target_size_relu``
        constrains, as opposed to ``_current_sparsity``'s clamped-to-zero
        count. Falls back to the whole tensor when no filter is given.
        """
        with torch.no_grad():
            if isinstance(log_alpha, torch.Tensor):
                probs_parts = [_hard_concrete_l0_probs(log_alpha, beta=beta)]
            else:
                probs_parts = [
                    _hard_concrete_l0_probs(log_alpha[l], beta=beta)
                    for l in self.layers
                ]
            n_active = 0.0
            n_valid = 0.0
            for probs in probs_parts:
                if combined_filter is not None:
                    valid = (~combined_filter.bool()).to(probs.dtype).unsqueeze(0)
                    n_active += float((probs * valid).sum().item())
                    n_valid += float(valid.sum().item()) * probs.shape[0]
                else:
                    n_active += float(probs.sum().item())
                    n_valid += float(probs.numel())
            if n_valid == 0:
                return 0.0
            return 1.0 - n_active / n_valid

    def _current_sparsity(self, log_alpha, granularity, combined_filter=None,
                          beta: float = _HC_BETA) -> float:
        """Fraction of *valid* (gap/mode/causal-passing) entries whose
        Hard-Concrete mean has hit the clamp at 0.

        When ``combined_filter`` is supplied, the fraction is taken only over
        learnable cells (``~combined_filter``). Otherwise it falls back to
        the fraction over the full parameter tensor (legacy behaviour).

        Matches Cao et al. 2021's "fraction of non-zero weights" metric: an
        entry counts as pruned when the deterministic HC readout
        ``m = clamp(σ(log_alpha/β)·(ζ−γ) + γ, 0, 1)`` is exactly 0
        (i.e. ``log_alpha ≤ β·log(−γ/ζ) ≈ −1.60`` for the default HC
        constants). Such entries are exactly the ones that get −∞ bias
        (log(0) → log(ε) floor) at forward time and are thus genuinely
        killed; continuous ``0 < m < 1`` are still active and attenuated.

        The previous ``m < 0.5`` cutoff over-counted partially-attenuated
        edges as "off" and diverged from the paper's notion of sparsity.
        """
        with torch.no_grad():
            if isinstance(log_alpha, torch.Tensor):
                m_flat = _hard_concrete_mean(log_alpha, beta=beta).flatten()
                if combined_filter is not None:
                    valid_flat = (~combined_filter.bool()).flatten().to(m_flat.device)
                    n_repeat = m_flat.numel() // valid_flat.numel() if valid_flat.numel() > 0 else 1
                    valid_flat = valid_flat.repeat(max(1, n_repeat))[:m_flat.numel()]
                else:
                    valid_flat = torch.ones_like(m_flat, dtype=torch.bool)
            else:
                m_parts = [
                    _hard_concrete_mean(log_alpha[l], beta=beta) for l in self.layers
                ]
                m_flat = torch.cat([m.flatten() for m in m_parts])
                if combined_filter is not None:
                    valid_2d = (~combined_filter.bool()).to(m_flat.device)
                    # Each layer tensor is (H, S, S) for "head" or (1, S, S) for "layer";
                    # broadcast valid_2d across the leading dim.
                    valid_parts = []
                    for m in m_parts:
                        v = valid_2d.unsqueeze(0).expand_as(m).flatten()
                        valid_parts.append(v)
                    valid_flat = torch.cat(valid_parts)
                else:
                    valid_flat = torch.ones_like(m_flat, dtype=torch.bool)
            n_valid = int(valid_flat.sum().item())
            if n_valid == 0:
                return 0.0
            pruned_in_valid = int(((m_flat == 0.0) & valid_flat).sum().item())
            return pruned_in_valid / n_valid

    # ------------------------------------------------------------------
    # Per-step loss routines (each builds a scalar and calls backward)
    # ------------------------------------------------------------------

    def _step_local(
        self,
        input_ids, continuations, clean_logits_list,
        prefix_len, device, branch_rewards, position_mask_overrides,
    ):
        """Sum local objective across branches; backward once per branch.

        Multiple ``.backward()`` calls without an intervening ``step()``
        accumulate gradients, which is what we want (same as summing
        losses and calling backward once, but memory-cheaper).

        Returns summed task-loss value for logging.
        """
        task_loss_total = 0.0
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

            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits

            loss = self.objective_fn(
                clean_logits, logits.float(), position_mask, token_ids=full_input,
            )
            if branch_rewards is not None:
                loss = loss * branch_rewards[cont_idx]
            task_loss_total += float(loss.detach().item())
            # Mask is an installed leaf (see training loop in `discover`),
            # so each branch's graph is independent; retain_graph=False frees
            # saved tensors during backward and bounds peak memory.
            loss.backward()
            del logits, loss
        return task_loss_total

    def _step_global(
        self,
        input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
    ):
        """Global-objective step via the two-pass per-chain-weight trick.

        Identical to ``NodewiseAttributionSDPA._ig_step_global`` but the
        gradient flows into Hard-Concrete log_alpha rather than the
        interpolated mask.
        """
        # Fused linear+CE (Liger): avoids the (seq_len, vocab) fp32 logits
        # tensor that previously OOMed for long prefixes / large vocabs.
        lm_head_weight = self.model.lm_head.weight
        lm_head_bias = getattr(self.model.lm_head, "bias", None)

        chain_lps_detached = []
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                hidden = self.model.model(full_input).last_hidden_state
            lp = chain_log_prob_chunked(
                hidden, lm_head_weight, full_input, prefix_len,
                temperature=self.temperature, lm_head_bias=lm_head_bias,
            )
            chain_lps_detached.append(lp.detach())
        chain_lps_detached = torch.stack(chain_lps_detached)

        # NaN diagnostic — log first call and any time non-finite is seen.
        _diag_log = (not getattr(self, "_diag_first_done", False)) or (
            not torch.isfinite(chain_lps_detached).all()
            or not torch.isfinite(chain_logprobs_clean).all()
        )
        if _diag_log:
            self._diag_first_done = True
            cm = chain_lps_detached
            cc = chain_logprobs_clean
            print(
                f"[NAN-DIAG] step_global call: "
                f"lp_masked finite={torch.isfinite(cm).sum().item()}/{cm.numel()} "
                f"min={cm[torch.isfinite(cm)].min().item() if torch.isfinite(cm).any() else 'all-bad'} "
                f"max={cm[torch.isfinite(cm)].max().item() if torch.isfinite(cm).any() else 'all-bad'}",
                flush=True,
            )
            print(
                f"[NAN-DIAG]                  "
                f"lp_clean  finite={torch.isfinite(cc).sum().item()}/{cc.numel()} "
                f"min={cc[torch.isfinite(cc)].min().item() if torch.isfinite(cc).any() else 'all-bad'} "
                f"max={cc[torch.isfinite(cc)].max().item() if torch.isfinite(cc).any() else 'all-bad'}",
                flush=True,
            )
            if not torch.isfinite(cm).all():
                bad = (~torch.isfinite(cm)).nonzero(as_tuple=True)[0].tolist()
                print(f"[NAN-DIAG]   non-finite lp_masked branches: {bad} values={cm[bad].tolist()}", flush=True)

        chain_lps_param = chain_lps_detached.clone().requires_grad_(True)
        global_loss = self.objective_fn(
            chain_lps_param, chain_logprobs_clean,
            answer_ids.to(device), num_answers,
            chain_lengths=chain_lengths,
            is_method=self.importance_sampling_method,
            is_temperature=self.importance_sampling_temperature,
        )
        task_loss_val = float(global_loss.detach().item())
        if _diag_log:
            print(f"[NAN-DIAG] global_loss={task_loss_val}", flush=True)
        global_loss.backward()
        per_chain_weights = chain_lps_param.grad.detach()
        # ∂loss/∂(chain log-prob) is the entire signal the model-side
        # backward will see; a near-one-hot |weight| distribution means the
        # objective has degenerated to pushing a single candidate.
        with torch.no_grad():
            absw = per_chain_weights.abs()
            absw_sum = float(absw.sum().item())
            if absw_sum > 0:
                p_absw = absw / absw_sum
                w_entropy = float(-(p_absw * (p_absw + 1e-12).log()).sum().item())
                w_max_share = float(p_absw.max().item())
            else:
                w_entropy = 0.0
                w_max_share = 0.0
            self._last_global_diag = {
                "per_chain_weight_abs_sum": absw_sum,
                "per_chain_weight_entropy": w_entropy,
                "per_chain_weight_max_share": w_max_share,
            }
        if _diag_log:
            pcw = per_chain_weights
            print(
                f"[NAN-DIAG] per_chain_weights finite={torch.isfinite(pcw).sum().item()}/{pcw.numel()} "
                f"min={pcw[torch.isfinite(pcw)].min().item() if torch.isfinite(pcw).any() else 'all-bad'} "
                f"max={pcw[torch.isfinite(pcw)].max().item() if torch.isfinite(pcw).any() else 'all-bad'}",
                flush=True,
            )

        # Mask is now an installed leaf (see training loop in `discover`),
        # so each branch's model-side autograd graph is fully self-contained;
        # retain_graph=False lets backward free saved tensors as it visits
        # them, capping per-branch peak memory.
        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.amp.autocast("cuda"):
                hidden = self.model.model(full_input).last_hidden_state
            lp = chain_log_prob_chunked(
                hidden, lm_head_weight, full_input, prefix_len,
                temperature=self.temperature, lm_head_bias=lm_head_bias,
            )
            (lp * per_chain_weights[cont_idx]).backward()
            del hidden, lp
            clear_cuda()
        return task_loss_val
