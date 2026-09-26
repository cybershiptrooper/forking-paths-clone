"""Subnetwork probing for the answer-distribution task on a rollout bank with
open-ended answers (MATH).

The loss of one training step is, as for ``answer_probe_kl`` on a rollout
bank, the sum over the step's continuations of the mean over the K
Hard-Concrete samples of KL(q_clean(. | x, y) || q_M(. | x, y)). Here
q(. | x, y) is the distribution over the answer clusters of a candidate bank:
the softmax over the teacher-forced log-probabilities of every candidate path
after ``prefix + rollout y + probe suffix``, summed per cluster (paper
Appendix "Open-ended answers"). q_clean comes from the rollout-bank file
(``build_candidate_rollout_bank.py``); only the answer tokens after the
probe suffix are scored, since the context is shared by every candidate.

One continuation needs K x N forward passes (N candidate paths), so the
rows (k, path) are batched with right padding in micro-batches of
``rows_per_forward`` and the gradient uses the two-pass per-path weight
trick of ``_step_global``: a no-grad pass gives every path log-probability,
the KL of each sample is differentiated with respect to them, and a second
pass with gradients backpropagates ``sum(weight * log-prob)``.

The continuations handed to ``discover`` only set the count and the length
of the token-to-sentence map; the rows are rebuilt from the bank by index
(``_get_clean_logits`` returns the indices, which ``_continuations_for_step``
subsamples like any per-continuation list).

Registered as ``nodewise_subnetwork_probing_candidate_rollouts``.
"""
from __future__ import annotations

from typing import Optional

import torch

from expts.direct_answer_circuit_discovery.candidate_rollouts import (
    padded_path_batch, path_logprobs_from_hidden, cluster_probs, kl,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA, _HC_BETA, _hard_concrete_sample,
)


def candidate_rollout_kl(*args, **kwargs):
    """Name tag for the objective; the loss is computed inside the trainer."""
    raise RuntimeError("candidate_rollout_kl is evaluated by the candidate-rollouts trainer")


class NodewiseSubnetworkProbingCandidateRollouts(NodewiseSubnetworkProbingHCBatched):

    def __init__(self, candidate_rollout_bank: Optional[dict] = None, rollout_bank_set: str = "B",
                 rows_per_forward: int = 4, **kwargs):
        super().__init__(**kwargs)
        if candidate_rollout_bank is None:
            raise ValueError("nodewise_subnetwork_probing_candidate_rollouts needs candidate_rollout_bank")
        bank = candidate_rollout_bank
        self._paths = [list(p) for p in bank["candidate_paths"]]
        self._cluster_ids = torch.tensor(bank["cluster_ids"], dtype=torch.long)
        self._num_clusters = int(bank["num_clusters"])
        rolls = bank["sets"][rollout_bank_set]["rollouts"]
        self._rollout_tokens = [list(r["tokens"]) for r in rolls]
        self._clean_q = [torch.tensor(r["clean_cluster_probs"], dtype=torch.float32) for r in rolls]
        self._suffix = list(bank["probe_suffix_token_ids"])
        self.rows_per_forward = int(rows_per_forward)

    def _sample_masks(self, log_alpha, granularity, beta: float = _HC_BETA):
        """Always (K, S, S), also for K = 1, so rows can index samples."""
        if granularity != "pair" or not isinstance(log_alpha, torch.Tensor):
            raise NotImplementedError("candidate-rollouts trainer supports pair granularity only")
        K = self._hc_total_samples
        sampled = self._apply_dropout(_hard_concrete_sample(log_alpha.expand(K, -1, -1), beta=beta))
        return {l: sampled for l in self.layers}

    def _get_clean_logits(self, input_ids, continuations):
        return list(range(len(continuations)))

    def _rows_logprobs(self, context, rows, masks_full, grad_weights=None):
        """Log-probs of (k, path) rows; with ``grad_weights`` also backprop
        sum(weight * log-prob) chunk by chunk."""
        out = []
        L = context.shape[-1]
        for lo in range(0, len(rows), self.rows_per_forward):
            chunk = rows[lo: lo + self.rows_per_forward]
            ks = torch.tensor([k for k, _ in chunk], device=context.device)
            NodewiseSubnetworkProbingSDPA._install_masks(
                self, {l: m.index_select(0, ks) for l, m in masks_full.items()})
            inp, tgt, valid = padded_path_batch(context, [self._paths[j] for _, j in chunk])
            if grad_weights is None:
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    hidden = self.model.model(inp).last_hidden_state
                    out.append(path_logprobs_from_hidden(self.model, hidden, L, tgt, valid).float())
            else:
                with torch.amp.autocast("cuda"):
                    hidden = self.model.model(inp).last_hidden_state
                lp = path_logprobs_from_hidden(self.model, hidden, L, tgt, valid)
                (lp * grad_weights[lo: lo + len(chunk)].to(lp.device)).sum().backward()
            del hidden
        return torch.cat(out) if out else None

    def _step_local(self, input_ids, continuations, clean_logits_list,
                    prefix_len, device, branch_rewards, position_mask_overrides):
        K = self._hc_total_samples
        N = len(self._paths)
        masks_full = self._installed_mask_dict
        assert masks_full is not None, "_install_masks must run before _step_local"
        suffix = torch.tensor([self._suffix], dtype=torch.long, device=device)
        rows = [(k, j) for k in range(K) for j in range(N)]
        total = 0.0
        try:
            for ridx in clean_logits_list:
                roll = torch.tensor([self._rollout_tokens[ridx]], dtype=torch.long, device=device)
                context = torch.cat([input_ids, roll, suffix], dim=-1)
                lp = self._rows_logprobs(context, rows, masks_full).view(K, N)
                lp_leaf = lp.detach().clone().requires_grad_(True)
                qc = self._clean_q[ridx].to(device)
                loss = torch.stack([kl(qc, cluster_probs(lp_leaf[k], self._cluster_ids, self._num_clusters))
                                    for k in range(K)]).mean()
                loss.backward()
                total += float(loss.detach().item())
                self._rows_logprobs(context, rows, masks_full, grad_weights=lp_leaf.grad.detach().view(-1))
        finally:
            NodewiseSubnetworkProbingSDPA._install_masks(self, masks_full)
        return total

    def discover(self, *args, **kwargs):
        node_mask = super().discover(*args, **kwargs)
        node_mask.algorithm = "nodewise_subnetwork_probing_candidate_rollouts"
        node_mask.metadata["rows_per_forward"] = self.rows_per_forward
        return node_mask
