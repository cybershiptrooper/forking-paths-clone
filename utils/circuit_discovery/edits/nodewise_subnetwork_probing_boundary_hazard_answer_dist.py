"""Boundary-hazard subnetwork probing against the answer-distribution loss.

Identical training to ``NodewiseSubnetworkProbingBoundaryHazardBatched``
except that the objective also receives, per bank continuation, the clean
model's full forced-probe distribution at every paragraph break
(``probe_probs``) and at the end of the continuation (``final_probs``),
both stored by build_answer_dist_boundary_data.py, plus the weight of the
KL term.  This is what ``boundary_answer_dist_kl_length`` needs (fix 2 of
the 2026-09-21 meeting notes): every break is a stopping point, and the
answer distribution the stops imply is matched to the bank's final-answer
distribution.

Implemented as a separate subclass (new file, registered under its own
algorithm name) so the existing trainer classes are not modified.
"""

from typing import List

import torch

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard import (
    NodewiseSubnetworkProbingBoundaryHazard,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard_batched import (
    NodewiseSubnetworkProbingBoundaryHazardBatched,
)


class NodewiseSubnetworkProbingBoundaryHazardAnswerDist(
    NodewiseSubnetworkProbingBoundaryHazard
):
    """Passes per-break and final probe distributions to the objective."""

    def __init__(self, answer_dist_kl_weight: float = 1.0, **kwargs):
        self.answer_dist_kl_weight = float(answer_dist_kl_weight)
        super().__init__(**kwargs)

    def _prepare_hazard_tensors(self, device, num_continuations: int):
        super()._prepare_hazard_tensors(device, num_continuations)
        letters = self.boundary_data.get("answer_letters")
        cands = self.boundary_data["candidates"]
        missing = [
            i for i, c in enumerate(cands)
            if "probe_probs" not in c or "final_probs" not in c
        ]
        if letters is None or missing:
            raise ValueError(
                "boundary_data lacks answer_letters / probe_probs / "
                f"final_probs (candidates {missing}); build it with "
                "build_answer_dist_boundary_data.py."
            )
        self._bd_probe_probs: List[torch.Tensor] = [
            torch.tensor(
                [[p[L] for L in letters] for p in c["probe_probs"]],
                dtype=torch.float32, device=device,
            )
            for c in cands
        ]
        self._bd_final_probs: List[torch.Tensor] = [
            torch.tensor(
                [c["final_probs"][L] for L in letters],
                dtype=torch.float32, device=device,
            )
            for c in cands
        ]

    def _resolve_hazard_fn(self):
        fn = super()._resolve_hazard_fn()

        def with_answer_dist(**kwargs):
            return fn(
                probe_probs=self._bd_probe_probs,
                final_probs=self._bd_final_probs,
                answer_kl_weight=self.answer_dist_kl_weight,
                **kwargs,
            )

        return with_answer_dist


class NodewiseSubnetworkProbingBoundaryHazardAnswerDistBatched(
    NodewiseSubnetworkProbingBoundaryHazardAnswerDist,
    NodewiseSubnetworkProbingBoundaryHazardBatched,
):
    """Batched step + answer-distribution objective inputs.

    MRO: the step comes from the batched class, the hazard-tensor
    preparation and objective wrapping from the answer-distribution class.
    """
