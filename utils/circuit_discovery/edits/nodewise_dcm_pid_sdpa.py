"""Deterministic continuous mask (DCM) + PID sparsity controller (SDPA).

Port of the "A1" variant from the belief_dynamics DCM study
(``causal-experiments/exps/patching/dcm/run_variants.py``) to
sentence-pair attention masks. Differences from subnetwork probing
(``NodewiseSubnetworkProbingSDPA``):

- **Deterministic mask.** Each learnable sentence-pair entry is a plain
  parameter in [0, 1] (init 1 - ε), used directly as the attention mask
  every step. No Hard-Concrete sampling, no gradient variance from
  stochastic gates.
- **PID-controlled sparsity.** The sparsity term is
  ``mult · active_frac`` where ``active_frac`` is the mean mask value
  over learnable entries. A log-space PID controller actuates ``mult``
  so that the *hard zero count* (learnable entries with mask < 0.5)
  tracks a linear ramp from 0 to ``pid_max_target_sparsity`` of the
  learnable pool over the first ``pid_ramp_end_frac`` of training. There
  is no λ warmup and no hinge: sparsity pressure exists from step 0 and
  its magnitude is feedback-controlled rather than scheduled, so the
  budget can never be enforced in one burst (the L0-hinge spike failure
  mode documented in ``notes/reports_diagnostics/l0_hinge_spike_scan.md``).
- **One run, many sparsities.** As the ramp passes each entry of
  ``snapshot_sparsities``, the continuous mask is snapshotted. The
  caller (``expts/direct_answer_circuit_discovery/learn.py``) writes one
  NodeMask JSON per snapshot with ``target_sparsity`` set to the crossed
  value, so the standard matched-target evaluation applies unchanged
  (top-k over the snapshot's continuous scores).

To limit the early-training overshoot seen in the belief_dynamics study
(the controller reacting late to the initial ramp), ``pid_mult_init``
defaults to a very small value (1e-3), so the first steps are almost
pure task-loss optimization and the controller raises the pressure
gradually from there.

Only sentence-pair granularity is supported.
"""

import json
import math
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
from utils.utils import Sentence
from utils.objectives import is_global_objective
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
)


class _LogSpacePID:
    """PID on log(mult): same gains give the same *relative* change in mult.

    Matches the belief_dynamics ``PIDController``:
    ``u = kp·rate_error + ki·count_error + kd·clip(Δrate_error)``, where
    count_error is the (bounded) integral of rate_error, so windup cannot
    occur.
    """

    def __init__(self, kp, ki, kd, init_mult,
                 mult_min=1e-8, mult_max=1e8, d_clip=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.log_mult = math.log(max(init_mult, mult_min))
        self.log_mult_min = math.log(mult_min)
        self.log_mult_max = math.log(mult_max)
        self.d_clip = d_clip
        self._prev_rate_error = 0.0
        self.prev_n_zero: Optional[int] = None

    @property
    def mult(self) -> float:
        return math.exp(self.log_mult)

    def step(self, actual_rate, target_rate, count_error) -> float:
        rate_error = target_rate - actual_rate
        derivative = max(-self.d_clip, min(self.d_clip,
                                           rate_error - self._prev_rate_error))
        self._prev_rate_error = rate_error
        u = self.kp * rate_error + self.ki * count_error + self.kd * derivative
        self.log_mult = max(self.log_mult_min,
                            min(self.log_mult_max, self.log_mult + u))
        return self.mult


class NodewiseDCMPIDSDPA(NodewiseSubnetworkProbingSDPA):
    """DCM + PID circuit discovery over sentence-pair attention masks.

    Reuses the SDPA patching machinery and the per-step loss routines
    (``_step_local`` / ``_step_global``) from the SNP class; only the mask
    parameterization, the sparsity control, and the training loop differ.
    """

    def __init__(
        self,
        pid_kp: float = 0.1,
        pid_ki: float = 0.001,
        pid_kd: float = 0.0,
        pid_mult_init: float = 1e-3,
        pid_ramp_end_frac: float = 0.9,
        pid_max_target_sparsity: float = 0.95,
        snapshot_sparsities: Optional[List[float]] = None,
        dcm_mask_init: float = 0.99,
        dcm_l0_optimizer: str = "sgd",
        dcm_polarization: float = 0.0,
        pid_snapshot_hold_steps: int = 0,
        dcm_lr_init: Optional[float] = None,
        dcm_lr_warmup_frac: float = 0.0,
        dcm_task_optimizer: str = "adam",
        dcm_max_flips_per_step: Optional[int] = None,
        dcm_flip_cap_ramp_mult: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            pid_kp / pid_ki / pid_kd: PID gains on log(mult). Defaults are
                the belief_dynamics values (0.1 / 0.001 / 0).
            pid_mult_init: initial sparsity multiplier. Deliberately small
                so early steps are near-pure task optimization (limits the
                initial-ramp overshoot seen in the belief_dynamics study).
            pid_ramp_end_frac: fraction of training over which the target
                zero-count ramps linearly from 0 to the maximum; held at
                the maximum afterwards.
            pid_max_target_sparsity: ramp endpoint as a fraction of the
                learnable pool. Should be ≥ max(snapshot_sparsities).
            snapshot_sparsities: sparsity levels at which to snapshot the
                continuous mask as it densifies past them. Default
                [0.10, 0.25, 0.50, 0.75, 0.90].
            dcm_mask_init: initial mask value (1 − ε in the belief_dynamics
                study; ε = 0.01).
            dcm_l0_optimizer: how the sparsity gradient is applied.
                ``"sgd"`` (default): Adam sees only the task gradient; the
                sparsity push is a manual non-adaptive step of
                ``lr · mult / n_learnable`` per learnable cell, so ``mult``
                scales the pressure linearly and ``pid_mult_init`` being
                small genuinely means a small initial push. ``"adam"``:
                single combined loss through Adam — exact parity with the
                belief_dynamics A1 code, but Adam's per-parameter
                normalization makes the push magnitude nearly independent
                of ``mult`` (any nonzero mult sinks weak-task cells at
                ~lr per step; confirmed in the smoke run), reproducing the
                early-ramp overshoot the study reported.
            dcm_polarization: coefficient c of a per-cell push toward the
                nearer extreme, Δm = lr · c · (2m − 1) per step on
                learnable cells (the gradient-descent direction of the
                penalty c · m(1 − m)). Motivated by the pilot finding that
                DCM converges to genuinely intermediate mask values (a
                large fraction of cells strictly between 0.1 and 0.9), so
                the binary top-k readout is a big perturbation of the mask
                the task loss actually tuned; polarization drives cells to
                {0, 1} during training so the readout is faithful. 0 = off.
            pid_snapshot_hold_steps: number of steps to freeze the ramp
                target at each snapshot sparsity before taking the
                snapshot, so the membership at that operating point can
                settle under a constant budget (the pilot showed mid-ramp
                snapshots capture unsettled membership). 0 = snapshot at
                the crossing step (original behavior).
            dcm_task_optimizer: optimizer for the task gradient.
                ``"adam"`` (default, original behavior): Adam's
                per-parameter normalization moves every cell with a
                consistent gradient sign at ~lr per step regardless of
                gradient magnitude — from the homogeneous init the whole
                population descends as a coherent front and crosses 0.5
                nearly together (the avalanche documented in the
                early-analysis-point pilot). ``"sgd"``: plain SGD, no
                momentum — per-cell step sizes stay proportional to task
                gradient magnitude, so crossing times spread out; the raw
                gradient scale is prompt-dependent, so lr needs per-prompt
                tuning. ``"sgd_norm"``: SGD with the task gradient divided
                each step by its RMS over learnable cells — relative
                magnitudes (the ranking) are preserved while the
                mean-magnitude cell moves at exactly lr per step,
                removing the per-prompt lr dependence.
            dcm_max_flips_per_step: hard cap on the number of learnable
                cells allowed to newly cross below 0.5 in a single step.
                After each update, if more than this many cells crossed,
                the crossings with the lowest post-update values (the
                strongest pushes) are kept and the excess cells are
                reverted to their pre-step values. Flips back *on*
                (reopening) are never capped. None = off.
            dcm_flip_cap_ramp_mult: alternative/additional cap expressed
                as a multiple of the PID ramp rate (zero-cells per step
                needed to track the linear ramp). When both cap options
                are set the effective cap is their maximum, so an
                absolute cap of e.g. 10 cannot make the ramp infeasible
                on prompts with large learnable pools. None = off.
        """
        snapshot_sparsities = sorted(
            snapshot_sparsities if snapshot_sparsities is not None
            else [0.10, 0.25, 0.50, 0.75, 0.90]
        )
        if pid_max_target_sparsity < snapshot_sparsities[-1]:
            raise ValueError(
                f"pid_max_target_sparsity={pid_max_target_sparsity} is below "
                f"the largest snapshot sparsity "
                f"{snapshot_sparsities[-1]}; the ramp would never reach it."
            )
        if dcm_task_optimizer not in ("adam", "sgd", "sgd_norm"):
            raise ValueError(
                f"dcm_task_optimizer must be 'adam', 'sgd', or 'sgd_norm', "
                f"got {dcm_task_optimizer!r}"
            )
        if dcm_max_flips_per_step is not None and dcm_max_flips_per_step < 1:
            raise ValueError(
                f"dcm_max_flips_per_step must be >= 1, got "
                f"{dcm_max_flips_per_step}"
            )
        if dcm_flip_cap_ramp_mult is not None and dcm_flip_cap_ramp_mult <= 0:
            raise ValueError(
                f"dcm_flip_cap_ramp_mult must be > 0, got "
                f"{dcm_flip_cap_ramp_mult}"
            )
        super().__init__(**kwargs)
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pid_mult_init = pid_mult_init
        self.pid_ramp_end_frac = pid_ramp_end_frac
        self.pid_max_target_sparsity = pid_max_target_sparsity
        self.snapshot_sparsities = snapshot_sparsities
        self.dcm_mask_init = dcm_mask_init
        if dcm_l0_optimizer not in ("sgd", "adam"):
            raise ValueError(
                f"dcm_l0_optimizer must be 'sgd' or 'adam', got "
                f"{dcm_l0_optimizer!r}"
            )
        self.dcm_l0_optimizer = dcm_l0_optimizer
        if dcm_polarization < 0:
            raise ValueError(
                f"dcm_polarization must be >= 0, got {dcm_polarization}"
            )
        self.dcm_polarization = dcm_polarization
        if pid_snapshot_hold_steps < 0:
            raise ValueError(
                f"pid_snapshot_hold_steps must be >= 0, got "
                f"{pid_snapshot_hold_steps}"
            )
        self.pid_snapshot_hold_steps = pid_snapshot_hold_steps
        self.dcm_lr_init = dcm_lr_init
        if not (0.0 <= dcm_lr_warmup_frac < 1.0):
            raise ValueError(
                f"dcm_lr_warmup_frac must be in [0, 1), got "
                f"{dcm_lr_warmup_frac}"
            )
        self.dcm_lr_warmup_frac = dcm_lr_warmup_frac
        self.dcm_task_optimizer = dcm_task_optimizer
        self.dcm_max_flips_per_step = dcm_max_flips_per_step
        self.dcm_flip_cap_ramp_mult = dcm_flip_cap_ramp_mult
        # Filled by discover(); read by learn.py to write per-sparsity masks.
        self.snapshot_scores = {}
        self.snapshot_steps = {}

    @staticmethod
    def _apply_flip_cap(mask_param, learnable, pre_vals, flip_cap) -> int:
        """Cap the number of learnable cells newly crossing below 0.5.

        Keeps the ``flip_cap`` crossings with the lowest post-update
        values (the strongest pushes) and restores the excess cells to
        their pre-step values. Crossings back *above* 0.5 (reopenings)
        are never touched. Returns the number of reverted cells.
        """
        with torch.no_grad():
            vals = mask_param[0][learnable]
            newly_off = (vals < 0.5) & (pre_vals >= 0.5)
            n_new = int(newly_off.sum().item())
            if n_new <= flip_cap:
                return 0
            new_idx = newly_off.nonzero(as_tuple=True)[0]
            order = torch.argsort(vals[new_idx])
            revert_idx = new_idx[order[flip_cap:]]
            vals[revert_idx] = pre_vals[revert_idx]
            mask_param[0][learnable] = vals
            return int(revert_idx.numel())

    def _target_zero_frac(self, step: int) -> float:
        total = max(1, self.num_training_steps - 1)
        ramp_steps = max(1.0, self.pid_ramp_end_frac * total)
        return self.pid_max_target_sparsity * min(1.0, step / ramp_steps)

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
        if self.mask_granularity != "pair":
            raise ValueError(
                "nodewise_dcm_pid_sdpa supports mask_granularity='pair' only, "
                f"got {self.mask_granularity!r}"
            )
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
        learnable = (~combined_filter.bool())  # (S, S), True = learnable
        n_learnable = int(learnable.sum().item())

        forward_fn = self._sdpa_forward()

        # ----- Mask parameter (deterministic, pair granularity) -----
        mask_param = torch.full(
            (1, num_sents, num_sents), self.dcm_mask_init,
            device=device, dtype=torch.float32, requires_grad=True,
        )
        if self.dcm_task_optimizer == "adam":
            optim = torch.optim.Adam([mask_param], lr=self.learning_rate)
        else:
            # Plain SGD (no momentum): per-cell steps stay proportional to
            # the task gradient magnitude, so crossing times spread out
            # instead of the Adam herd. "sgd_norm" additionally divides
            # the gradient by its RMS over learnable cells each step.
            optim = torch.optim.SGD([mask_param], lr=self.learning_rate)
        pid = _LogSpacePID(
            self.pid_kp, self.pid_ki, self.pid_kd, self.pid_mult_init,
        )

        # ----- Patch target layers; deterministic all-ones for clean logits -----
        ones_masks = self._ones_masks("pair", num_heads, num_sents, device)
        handles = self._patch_model(ones_masks, token_to_sent, combined_filter, forward_fn)
        non_target_handles = []
        if self.ablate_non_target_layers:
            non_target_handles = self._patch_non_target_layers_sdpa(
                num_sents=num_sents,
                token_to_sent=token_to_sent,
                gap_filter=combined_filter,
            )

        # ----- Objective setup (same as SNP) -----
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
        if use_global:
            from utils.importance_sampling import chain_log_prob_chunked
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

        chain_lengths = torch.tensor(
            [c.shape[-1] for c in continuations], dtype=torch.long, device=device,
        )

        print(
            f"Running DCM+PID ({self.num_training_steps} steps, "
            f"lr={self.learning_rate}, kp={self.pid_kp}, ki={self.pid_ki}, "
            f"kd={self.pid_kd}, mult_init={self.pid_mult_init}, "
            f"ramp_end_frac={self.pid_ramp_end_frac}, "
            f"max_tsp={self.pid_max_target_sparsity}, "
            f"snapshots={self.snapshot_sparsities}, "
            f"task_optimizer={self.dcm_task_optimizer}, "
            f"max_flips={self.dcm_max_flips_per_step}, "
            f"flip_cap_ramp_mult={self.dcm_flip_cap_ramp_mult}, "
            f"{len(continuations)} continuations)..."
        )

        run_name = self.wandb_run_name
        if run_name is None and self.log_dir is not None:
            run_name = os.path.basename(os.path.normpath(self.log_dir))
        wandb_run = init_wandb_run(
            project=self.wandb_project,
            run_name=run_name,
            config={
                "algorithm": "nodewise_dcm_pid_sdpa",
                "num_training_steps": self.num_training_steps,
                "learning_rate": self.learning_rate,
                "pid_kp": self.pid_kp,
                "pid_ki": self.pid_ki,
                "pid_kd": self.pid_kd,
                "pid_mult_init": self.pid_mult_init,
                "pid_ramp_end_frac": self.pid_ramp_end_frac,
                "pid_max_target_sparsity": self.pid_max_target_sparsity,
                "snapshot_sparsities": self.snapshot_sparsities,
                "dcm_task_optimizer": self.dcm_task_optimizer,
                "dcm_max_flips_per_step": self.dcm_max_flips_per_step,
                "dcm_flip_cap_ramp_mult": self.dcm_flip_cap_ramp_mult,
                "num_continuations": len(continuations),
                "num_sentences": num_sents,
                "objective": objective_name,
                "mask_mode": mask_mode,
            },
        )

        if getattr(self.model, "is_gradient_checkpointing", False):
            self.model.train()

        pending_snapshots = list(self.snapshot_sparsities)
        n_zero_series_prev = None
        # Snapshot-hold state: while > 0, the ramp target is frozen at the
        # pending snapshot level and decremented each step; the snapshot is
        # taken when it reaches 0. `ramp_step` advances only outside holds,
        # so the ramp completes later by (num holds x hold length) steps.
        _hold_remaining = 0
        _ramp_step = 0
        # Gross node-flip tracking (mask crossing the 0.5 threshold),
        # accumulated between logged steps; net change alone is also
        # recoverable from the `sparsity` series.
        _prev_below = None
        _flips_off_accum = 0
        _flips_on_accum = 0
        total = max(1, self.num_training_steps - 1)
        # PID rate target: zero-cells per step while the ramp is active.
        ramp_steps = max(1.0, self.pid_ramp_end_frac * total)
        ramp_rate = n_learnable * self.pid_max_target_sparsity / ramp_steps
        # Effective flip cap: max over the provided cap options, so an
        # absolute cap cannot make the ramp infeasible on large pools.
        flip_cap = None
        if (self.dcm_max_flips_per_step is not None
                or self.dcm_flip_cap_ramp_mult is not None):
            cap_candidates = []
            if self.dcm_max_flips_per_step is not None:
                cap_candidates.append(self.dcm_max_flips_per_step)
            if self.dcm_flip_cap_ramp_mult is not None:
                cap_candidates.append(
                    math.ceil(self.dcm_flip_cap_ramp_mult * ramp_rate)
                )
            flip_cap = max(cap_candidates)
            print(
                f"  flip cap: at most {flip_cap} learnable cells may newly "
                f"cross below 0.5 per step (ramp rate {ramp_rate:.2f}/step)"
            )
        _flips_reverted_accum = 0
        _task_grad_rms_last = 0.0

        def _lr_at(step):
            if self.dcm_lr_warmup_frac <= 0.0 or self.dcm_lr_init is None:
                return self.learning_rate
            total_ = max(1, self.num_training_steps - 1)
            w = max(1.0, self.dcm_lr_warmup_frac * total_)
            f = min(1.0, step / w)
            return self.dcm_lr_init + (self.learning_rate - self.dcm_lr_init) * f

        for step in tqdm(range(self.num_training_steps), desc="DCM+PID steps"):
            current_lr = _lr_at(step)
            for pg in optim.param_groups:
                pg["lr"] = current_lr
            optim.zero_grad(set_to_none=True)
            # The mask parameter itself is installed; task gradients
            # accumulate directly on it (no sampling → no leaf trick).
            self._install_masks({l: mask_param for l in self.layers})

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

            # Task-gradient diagnostics + optional RMS normalization.
            # ("sgd_norm": divide by the RMS over learnable cells so the
            # mean-magnitude cell moves at exactly lr per step while the
            # per-cell ranking — which Adam erases — is preserved.)
            if mask_param.grad is not None:
                with torch.no_grad():
                    _g = mask_param.grad[0][learnable].float()
                    _task_grad_rms_last = float(_g.pow(2).mean().sqrt().item())
                    if (self.dcm_task_optimizer == "sgd_norm"
                            and _task_grad_rms_last > 0):
                        mask_param.grad /= (_task_grad_rms_last + 1e-12)
            # Pre-step values, for the flip cap's revert.
            if flip_cap is not None:
                with torch.no_grad():
                    pre_vals = mask_param[0][learnable].detach().clone()

            # Sparsity term: mult · mean mask value over learnable entries.
            # Its gradient is uniform (mult / n_learnable on every learnable
            # cell), so routing it through Adam would erase the mult
            # magnitude via per-parameter normalization; the default "sgd"
            # mode applies it as a manual non-adaptive step instead (same
            # rationale as the SNP hybrid optimizer).
            mult = pid.mult
            active_frac = (
                mask_param[0][learnable].to(torch.float32).mean()
            )
            sparsity_loss_val = mult * float(active_frac.detach().item())
            if self.dcm_l0_optimizer == "adam":
                (mult * active_frac).backward()
                optim.step()
            else:
                optim.step()
                with torch.no_grad():
                    push = current_lr * mult / max(1, n_learnable)
                    mask_param[0][learnable] -= push
            if self.dcm_polarization > 0.0:
                # Per-cell push toward the nearer extreme (see __init__).
                with torch.no_grad():
                    m = mask_param[0][learnable]
                    mask_param[0][learnable] = m + (
                        current_lr * self.dcm_polarization
                        * (2.0 * m - 1.0)
                    )
            with torch.no_grad():
                mask_param.clamp_(0.0, 1.0)

            # ----- Flip cap: revert excess new crossings below 0.5 -----
            if flip_cap is not None:
                _flips_reverted_accum += self._apply_flip_cap(
                    mask_param, learnable, pre_vals, flip_cap,
                )

            # ----- PID update on the hard zero count -----
            with torch.no_grad():
                n_zero = int(
                    (mask_param[0][learnable] < 0.5).sum().item()
                )
            actual_rate = (
                0.0 if n_zero_series_prev is None
                else float(n_zero - n_zero_series_prev)
            )
            n_zero_series_prev = n_zero
            if _hold_remaining > 0:
                # Frozen budget: hold the target at the pending snapshot
                # level and let membership settle.
                target_zero_frac = pending_snapshots[0]
                target_rate = 0.0
                _hold_remaining -= 1
            else:
                target_zero_frac = self._target_zero_frac(_ramp_step)
                target_rate = ramp_rate if _ramp_step < ramp_steps else 0.0
                _ramp_step += 1
            target_n_zero = n_learnable * target_zero_frac
            pid.step(actual_rate, target_rate, target_n_zero - n_zero)

            achieved_sparsity = n_zero / max(1, n_learnable)
            with torch.no_grad():
                below_now = mask_param[0][learnable] < 0.5
                if _prev_below is not None:
                    _flips_off_accum += int((below_now & ~_prev_below).sum().item())
                    _flips_on_accum += int((~below_now & _prev_below).sum().item())
                _prev_below = below_now.clone()

            # ----- Snapshots at target-sparsity crossings -----
            # With pid_snapshot_hold_steps > 0, the first crossing starts a
            # hold (budget frozen at the snapshot level); the snapshot is
            # taken when the hold expires. Without holds, snapshot at the
            # crossing step (original behavior).
            if (self.pid_snapshot_hold_steps > 0 and pending_snapshots
                    and _hold_remaining == 0
                    and achieved_sparsity >= pending_snapshots[0]
                    and pending_snapshots[0] not in self.snapshot_steps
                    and not getattr(self, "_hold_done_for", set()) & {pending_snapshots[0]}):
                _hold_remaining = self.pid_snapshot_hold_steps
                if not hasattr(self, "_hold_done_for"):
                    self._hold_done_for = set()
                self._hold_done_for.add(pending_snapshots[0])
                print(
                    f"  [hold] target sparsity {pending_snapshots[0]} crossed "
                    f"at step {step}; holding budget for "
                    f"{self.pid_snapshot_hold_steps} steps before snapshot"
                )
            take_now = (
                pending_snapshots and achieved_sparsity >= pending_snapshots[0]
                and (self.pid_snapshot_hold_steps == 0 or _hold_remaining == 0)
                and (self.pid_snapshot_hold_steps == 0
                     or pending_snapshots[0] in getattr(self, "_hold_done_for", set()))
            )
            while take_now:
                tsp = pending_snapshots.pop(0)
                with torch.no_grad():
                    self.snapshot_scores[tsp] = (
                        mask_param[0].detach().float().cpu().tolist()
                    )
                self.snapshot_steps[tsp] = step
                print(
                    f"  [snapshot] target sparsity {tsp} taken at step {step} "
                    f"(achieved {achieved_sparsity:.3f})"
                )
                take_now = (
                    pending_snapshots
                    and achieved_sparsity >= pending_snapshots[0]
                    and self.pid_snapshot_hold_steps == 0
                )

            if (step % self.log_every == 0) or (step + 1 == self.num_training_steps):
                metrics = {
                    "step": step,
                    "task_loss": task_loss_val,
                    "l0_loss": sparsity_loss_val,
                    "sparsity": achieved_sparsity,
                    "target_sparsity_ramp": target_zero_frac,
                    "pid_mult": mult,
                    "lr_current": current_lr,
                    "flips_off_interval": _flips_off_accum,
                    "flips_on_interval": _flips_on_accum,
                    "flips_reverted_interval": _flips_reverted_accum,
                    "task_grad_rms": _task_grad_rms_last,
                    "lr": self.learning_rate,
                }
                _flips_off_accum = 0
                _flips_on_accum = 0
                _flips_reverted_accum = 0
                if wandb_run is not None:
                    log_step(
                        wandb_run, step=step,
                        metrics={k: v for k, v in metrics.items() if k != "step"},
                    )
                if self.log_dir is not None:
                    os.makedirs(self.log_dir, exist_ok=True)
                    jl_path = os.path.join(self.log_dir, "training_metrics.jsonl")
                    with open(jl_path, "a") as _jlf:
                        _jlf.write(json.dumps(metrics) + "\n")

        finish_wandb_run(wandb_run)
        self._unpatch_model(handles)
        if non_target_handles:
            self._unpatch_model(non_target_handles)

        # Any snapshot targets the ramp never crossed fall back to the final
        # mask (logged loudly — the eval will show the shortfall via the
        # achieved-sparsity column).
        for tsp in pending_snapshots:
            print(
                f"  WARNING: ramp never crossed target sparsity {tsp} "
                f"(final achieved {achieved_sparsity:.3f}); snapshotting the "
                f"final mask instead."
            )
            with torch.no_grad():
                self.snapshot_scores[tsp] = (
                    mask_param[0].detach().float().cpu().tolist()
                )
            self.snapshot_steps[tsp] = self.num_training_steps - 1

        with torch.no_grad():
            # Frozen cells got no gradient; pin them to 1.0 in the saved
            # scores for readability (eval excludes them from the pool).
            final_scores = mask_param[0].detach().float().cpu()
            final_scores[~learnable.cpu()] = 1.0

        return NodeMask(
            model_name=self.model.config._name_or_path,
            algorithm="nodewise_dcm_pid_sdpa",
            layers=self.layers,
            sentences=[{"start": s.start, "end": s.end} for s in sentences],
            objective_name=objective_name,
            metadata={
                "num_training_steps": self.num_training_steps,
                "learning_rate": self.learning_rate,
                "pid_kp": self.pid_kp,
                "pid_ki": self.pid_ki,
                "pid_kd": self.pid_kd,
                "pid_mult_init": self.pid_mult_init,
                "pid_ramp_end_frac": self.pid_ramp_end_frac,
                "pid_max_target_sparsity": self.pid_max_target_sparsity,
                "snapshot_sparsities": self.snapshot_sparsities,
                "snapshot_steps": {
                    str(k): v for k, v in self.snapshot_steps.items()
                },
                "dcm_mask_init": self.dcm_mask_init,
                "dcm_l0_optimizer": self.dcm_l0_optimizer,
                "dcm_polarization": self.dcm_polarization,
                "pid_snapshot_hold_steps": self.pid_snapshot_hold_steps,
                "dcm_lr_init": self.dcm_lr_init,
                "dcm_lr_warmup_frac": self.dcm_lr_warmup_frac,
                "dcm_task_optimizer": self.dcm_task_optimizer,
                "dcm_max_flips_per_step": self.dcm_max_flips_per_step,
                "dcm_flip_cap_ramp_mult": self.dcm_flip_cap_ramp_mult,
                # The final mask sits at the ramp endpoint; matched-target
                # eval of the per-snapshot files uses their own values.
                "target_sparsity": self.pid_max_target_sparsity,
                "num_continuations": len(continuations),
                "sentence_gap": self.sentence_gap,
                "num_heads": num_heads,
                "ablate_non_target_layers": self.ablate_non_target_layers,
                "mask_mode": mask_mode,
                "num_prefix_sentences": num_prefix_sents,
                "num_frozen_prompt_sentences": num_frozen_prompt_sentences,
                "pair_aggregation": self.pair_aggregation,
                "mask_granularity": "pair",
                "branch_rewards": branch_rewards,
                "importance_sampling_method": self.importance_sampling_method,
                "importance_sampling_temperature": self.importance_sampling_temperature,
                "attention_backend": "sdpa",
                "score_readout": "raw_score",
            },
            scores=final_scores.tolist(),
        )
