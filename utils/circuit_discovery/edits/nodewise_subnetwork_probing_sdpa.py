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

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from utils.circuit_discovery.edits.nodewise_attribution_sdpa import (
    _expand_mask_to_log_additive,
)
from utils.circuit_discovery.edits.nodewise_patching_flash import (
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
    """Reparameterised Hard-Concrete sample in [0, 1], differentiable in log_alpha."""
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


def _hard_concrete_l0_mean(log_alpha: torch.Tensor) -> torch.Tensor:
    """Average per-entry active probability — Cao et al. 2021's R(θ) = (1/d) Σ_i ..."""
    return _hard_concrete_l0_probs(log_alpha).mean()


def _hard_concrete_l0_count(log_alpha: torch.Tensor) -> torch.Tensor:
    """Expected number of active entries (sum of per-entry probs); for diagnostics."""
    return _hard_concrete_l0_probs(log_alpha).sum()


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
        log_dir: Optional[str] = None,
        log_every: int = 5,
        plot_every: int = 20,
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
        self.log_dir = log_dir
        self.log_every = log_every
        self.plot_every = plot_every

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

    def _init_log_alpha(self, granularity, num_heads, num_sents, device):
        init = self.log_alpha_init
        if granularity == "head":
            return {
                l: torch.full(
                    (num_heads, num_sents, num_sents),
                    init, device=device, dtype=torch.float32, requires_grad=True,
                )
                for l in self.layers
            }
        if granularity == "layer":
            return {
                l: torch.full(
                    (1, num_sents, num_sents),
                    init, device=device, dtype=torch.float32, requires_grad=True,
                )
                for l in self.layers
            }
        # "pair": one shared tensor
        return torch.full(
            (1, num_sents, num_sents),
            init, device=device, dtype=torch.float32, requires_grad=True,
        )

    def _params_as_list(self, log_alpha, granularity):
        if granularity == "pair":
            return [log_alpha]
        return list(log_alpha.values())

    def _sample_masks(self, log_alpha, granularity):
        if granularity == "pair":
            sampled = _hard_concrete_sample(log_alpha)
            return {l: sampled for l in self.layers}
        return {l: _hard_concrete_sample(log_alpha[l]) for l in self.layers}

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

    def _l0(self, log_alpha, granularity):
        """Mean per-entry active probability, matching Cao et al.'s (1/d) Σ ...

        Using a mean (rather than a sum that scales with #edges × #layers) makes
        ``l0_lambda`` roughly invariant to granularity and to the number of
        target layers, so the same λ has comparable pressure across configs.
        """
        if granularity == "pair":
            # One shared parameter tensor — it is applied across ``self.layers``
            # at forward time, but the *parameter count* is one tensor, so the
            # regularizer is computed once (matches paper's parameter-wise R(θ)).
            return _hard_concrete_l0_mean(log_alpha)
        # head / layer: average the per-tensor means (all layer tensors same shape).
        return sum(
            _hard_concrete_l0_mean(log_alpha[l]) for l in self.layers
        ) / len(self.layers)

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

    def _expected_active_count(self, log_alpha, granularity) -> float:
        """Interpretable diagnostic: expected number of active edges."""
        with torch.no_grad():
            if granularity == "pair":
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
        combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)

        forward_fn = self._sdpa_forward()
        granularity = self.mask_granularity

        # ----- Init learnable parameters -----
        log_alpha = self._init_log_alpha(granularity, num_heads, num_sents, device)
        params = self._params_as_list(log_alpha, granularity)
        optim = torch.optim.Adam(params, lr=self.learning_rate)

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
        clean_logits_list = self._get_clean_logits(input_ids, continuations)

        # Swap in the initial stochastic HC sample for training.
        init_masks = self._sample_masks(log_alpha, granularity)
        self._install_masks(init_masks)

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
        log_steps: list[int] = []
        log_task: list[float] = []
        log_l0: list[float] = []
        log_sparsity: list[float] = []
        plot_dir = self._prepare_log_dir()

        for step in tqdm(range(self.num_training_steps), desc="SNP steps"):
            sampled = self._sample_masks(log_alpha, granularity)
            self._install_masks(sampled)
            optim.zero_grad(set_to_none=True)

            if use_global:
                task_loss_val = self._step_global(
                    input_ids, continuations, prefix_len, device,
                    chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
                )
            else:
                task_loss_val = self._step_local(
                    input_ids, continuations, clean_logits_list,
                    prefix_len, device, branch_rewards, position_mask_overrides,
                )

            # Sparsity penalty (independent of the task forward), scheduled λ.
            lam = self._lambda_at_step(step)
            if lam > 0.0:
                l0 = lam * self._l0(log_alpha, granularity)
                l0_val = float(l0.detach().item())
                l0.backward()
            else:
                # During warmup, skip the L0 forward + backward entirely rather
                # than propagate zeros through the graph.
                l0_val = 0.0

            optim.step()

            if plot_dir is not None and (step % self.log_every == 0):
                sparsity_val = self._current_sparsity(log_alpha, granularity)
                log_steps.append(step)
                log_task.append(task_loss_val)
                log_l0.append(l0_val)
                log_sparsity.append(sparsity_val)

            if (
                plot_dir is not None
                and log_steps
                and ((step + 1) % self.plot_every == 0
                     or step == self.num_training_steps - 1)
            ):
                self._flush_plots(plot_dir, log_steps, log_task, log_l0, log_sparsity)

        self._unpatch_model(handles)
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # ----- Readout: deterministic mask mean in [0, 1] -----
        with torch.no_grad():
            if granularity == "pair":
                readout = _hard_concrete_mean(log_alpha).detach().cpu()
                scores = readout[0].tolist()
            elif granularity == "layer":
                scores = {}
                for l in self.layers:
                    r = _hard_concrete_mean(log_alpha[l]).detach().cpu()
                    scores[l] = r[0].tolist()
            else:  # head
                scores = {}
                for l in self.layers:
                    r = _hard_concrete_mean(log_alpha[l]).detach().cpu()
                    scores[l] = {h: r[h].tolist() for h in range(num_heads)}

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
                "log_alpha_init": self.log_alpha_init,
                "hard_concrete_beta": _HC_BETA,
                "hard_concrete_gamma": _HC_GAMMA,
                "hard_concrete_zeta": _HC_ZETA,
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "pair_aggregation": self.pair_aggregation,
                "mask_granularity": granularity,
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
                "attention_backend": "sdpa",
                "score_readout": "hard_concrete_mean",
            },
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Logging / plotting
    # ------------------------------------------------------------------

    def _prepare_log_dir(self) -> Optional[str]:
        if self.log_dir is None:
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        return self.log_dir

    def _current_sparsity(self, log_alpha, granularity) -> float:
        """Fraction of entries whose Hard-Concrete mean has hit the clamp at 0.

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
            if granularity == "pair":
                m = _hard_concrete_mean(log_alpha)
            else:
                m = torch.cat([_hard_concrete_mean(log_alpha[l]).flatten() for l in self.layers])
            # Exact equality: ``torch.clamp(x, 0, 1)`` produces *bit-exact*
            # 0.0 for any input ≤ 0, so no tolerance is needed or correct.
            return float((m == 0.0).float().mean().item())

    def _flush_plots(self, plot_dir, steps, task, l0, sparsity):
        """Overwrite three PDFs with the training curves so far."""
        for name, ys, ylabel in [
            ("task_loss", task, "task loss (per-step)"),
            ("l0_loss", l0, "L0 penalty loss"),
            ("sparsity", sparsity, "fraction of edges off (HC mean < 0.5)"),
        ]:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(steps, ys, marker="o", markersize=3)
            ax.set_xlabel("training step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{name}.pdf"))
            plt.close(fig)

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
            # retain_graph: sampled mask `m` is a non-leaf derived from
            # log_alpha; the m->log_alpha subgraph is shared across all
            # branches within a step, so backward on branch i must not
            # free it before branch i+1.
            loss.backward(retain_graph=True)
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

        chain_lps_param = chain_lps_detached.clone().requires_grad_(True)
        global_loss = self.objective_fn(
            chain_lps_param, chain_logprobs_clean,
            answer_ids.to(device), num_answers,
            chain_lengths=chain_lengths,
            is_method=self.importance_sampling_method,
            is_temperature=self.importance_sampling_temperature,
        )
        task_loss_val = float(global_loss.detach().item())
        global_loss.backward()
        per_chain_weights = chain_lps_param.grad.detach()

        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            with torch.amp.autocast("cuda"):
                logits = self.model(full_input).logits
            lp = chain_log_prob(
                logits.float(), full_input, prefix_len, temperature=self.temperature,
            )
            # See _step_local: shared m->log_alpha subgraph across branches.
            (lp * per_chain_weights[cont_idx]).backward(retain_graph=True)
            del logits, lp
        return task_loss_val
