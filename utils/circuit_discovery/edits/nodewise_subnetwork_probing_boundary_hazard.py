"""Subnetwork probing against boundary-hazard objectives.

Same Hard-Concrete gate training as ``NodewiseSubnetworkProbingSDPA``, but
the per-step readout is not the summed sequence log-probability of each
bank candidate — it is the per-sentence-boundary log-probability of the
dedicated ``</think>`` token ("hazard" in the survival-analysis sense)
along each teacher-forced candidate:

    log h_b = log p_m(</think> | prefix, candidate tokens up to boundary b)

The objectives over these readouts live in
``utils.objectives.HAZARD_OBJECTIVES`` (probability of stopping at a
probe-correct boundary within the horizon, hazard lift over the clean
model, expected remaining length).  No sequence log-probability and no
importance weight appears anywhere; each readout is a single next-token
log-probability, so nothing is aggregated over hundreds of tokens.

Boundary metadata (positions, probe-correctness flags, clean-model
hazards, token gaps) is precomputed once per bank by
``expts/cot_termination_circuit_discovery/build_boundary_data.py`` and
passed in via the ``boundary_data`` constructor kwarg.

The per-step structure mirrors the parent's two-pass global step:
pass 1 computes detached readouts, the objective is evaluated on leaf
tensors and backpropagated to get per-boundary weights, pass 2 re-runs
each candidate's forward with gradient and pushes the weighted sum of
readouts through the model into the Hard-Concrete ``log_alpha``.
"""

from typing import List

import torch

from utils.utils import clear_cuda
from utils.objectives import HAZARD_OBJECTIVES
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
)


class NodewiseSubnetworkProbingBoundaryHazard(NodewiseSubnetworkProbingSDPA):
    """SNP whose global step reads per-boundary ``</think>`` log-probs."""

    def __init__(self, boundary_data: dict = None, **kwargs):
        if boundary_data is None:
            raise ValueError(
                "nodewise_subnetwork_probing_boundary_hazard requires "
                "boundary_data (see build_boundary_data.py)."
            )
        self.boundary_data = boundary_data
        self._hazard_prepared = False
        super().__init__(**kwargs)

    # ------------------------------------------------------------------

    def _resolve_hazard_fn(self):
        name = getattr(self.objective_fn, "__name__", "")
        for key, fn in HAZARD_OBJECTIVES.items():
            if name in (key, fn.__name__):
                return fn
        raise ValueError(
            f"Objective {name!r} is not a boundary-hazard objective; "
            f"use nodewise_subnetwork_probing_sdpa for chain-level objectives."
        )

    def _prepare_hazard_tensors(self, device, num_continuations: int):
        cands = self.boundary_data["candidates"]
        if len(cands) != num_continuations:
            raise ValueError(
                f"boundary_data has {len(cands)} candidates but the run has "
                f"{num_continuations} continuations — rebuild boundary data "
                f"for this bank."
            )
        self._bd_positions: List[torch.Tensor] = []
        self._bd_eligible: List[torch.Tensor] = []
        self._bd_clean_log_h: List[torch.Tensor] = []
        self._bd_gaps: List[torch.Tensor] = []
        for c in cands:
            self._bd_positions.append(
                torch.tensor(c["boundaries"], dtype=torch.long, device=device)
            )
            self._bd_eligible.append(
                torch.tensor(c["eligible"], dtype=torch.bool, device=device)
            )
            self._bd_clean_log_h.append(
                torch.tensor(
                    c["clean_log_h"], dtype=torch.float32, device=device
                )
            )
            self._bd_gaps.append(
                torch.tensor(c["gaps"], dtype=torch.float32, device=device)
            )
        event_ids = self.boundary_data.get(
            "event_token_ids", [int(self.boundary_data["think_end_id"])]
        )
        self._event_token_ids = torch.tensor(
            event_ids, dtype=torch.long, device=device,
        )
        self._horizon = int(self.boundary_data["horizon"])
        self._hazard_fn = self._resolve_hazard_fn()
        self._hazard_prepared = True

    def _boundary_log_h(self, full_input, prefix_len, positions, with_grad):
        """log p(wrap-up event token set) at each boundary of one candidate.

        ``positions`` are continuation-relative token indices of
        paragraph-break tokens; the hidden state at absolute position
        ``prefix_len + j`` predicts the token after continuation token j.
        The hazard is the total probability of the event token set (the
        wrap-up head tokens plus ``</think>``; see build_boundary_data.py).
        The LM-head matmul and log-softmax run in fp32 over only the
        boundary rows.
        """
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx, torch.amp.autocast("cuda"):
            hidden = self.model.model(full_input).last_hidden_state
        rows = hidden[0, prefix_len + positions]              # (B, d)
        lm_w = self.model.lm_head.weight
        logits = (rows @ lm_w.T).float()                      # (B, V) fp32
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.logsumexp(
            log_probs[:, self._event_token_ids], dim=-1,
        )                                                     # (B,)

    # ------------------------------------------------------------------

    def _step_global(
        self,
        input_ids, continuations, prefix_len, device,
        chain_logprobs_clean, answer_ids, num_answers, chain_lengths,
    ):
        if not self._hazard_prepared:
            self._prepare_hazard_tensors(device, len(continuations))

        # ----- Pass 1: detached readouts -> objective -> boundary weights.
        leaves: List[torch.Tensor] = []
        for cont, pos in zip(continuations, self._bd_positions):
            full_input = torch.cat([input_ids, cont], dim=-1)
            log_h = self._boundary_log_h(
                full_input, prefix_len, pos, with_grad=False,
            )
            leaves.append(log_h.detach().float().requires_grad_(True))

        loss = self._hazard_fn(
            log_h=leaves,
            eligible=self._bd_eligible,
            clean_log_h=self._bd_clean_log_h,
            gaps=self._bd_gaps,
            horizon=self._horizon,
            positions=self._bd_positions,
        )
        task_loss_val = float(loss.detach().item())
        loss.backward()
        weights = [leaf.grad for leaf in leaves]

        with torch.no_grad():
            absw = torch.cat([
                w.abs().flatten() for w in weights if w is not None
            ])
            absw_sum = float(absw.sum().item())
            if absw_sum > 0:
                p_absw = absw / absw_sum
                w_entropy = float(
                    -(p_absw * (p_absw + 1e-12).log()).sum().item()
                )
                w_max_share = float(p_absw.max().item())
            else:
                w_entropy, w_max_share = 0.0, 0.0
            self._last_global_diag = {
                "per_boundary_weight_abs_sum": absw_sum,
                "per_boundary_weight_entropy": w_entropy,
                "per_boundary_weight_max_share": w_max_share,
            }

        # ----- Pass 2: recompute with gradient, push weighted readouts.
        for cont, pos, w in zip(continuations, self._bd_positions, weights):
            if w is None or not torch.any(w != 0):
                continue
            full_input = torch.cat([input_ids, cont], dim=-1)
            log_h = self._boundary_log_h(
                full_input, prefix_len, pos, with_grad=True,
            )
            (log_h * w.detach()).sum().backward()
            del log_h
            clear_cuda()
        return task_loss_val
