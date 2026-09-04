"""Boundary-hazard subnetwork probing with the answer guard made continuous.

Identical training to ``NodewiseSubnetworkProbingBoundaryHazard`` except
that the objective also receives, per boundary, the clean model's
forced-answer-probe probability of the trace's own answer
(``probe_p_trace``, stored by build_boundary_data.py).  This is what the
``boundary_stop_prob_soft`` objective needs: instead of the binary
eligible/ineligible filter, each boundary's stopping term is weighted by
that probability.

Implemented as a separate subclass (new file, registered under its own
algorithm name) so the existing trainer classes are not modified.
"""

from typing import List

import torch

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard import (
    NodewiseSubnetworkProbingBoundaryHazard,
)


class NodewiseSubnetworkProbingBoundaryHazardProbeWeighted(
    NodewiseSubnetworkProbingBoundaryHazard
):
    """Passes per-boundary probe_p_trace through to the hazard objective."""

    def _prepare_hazard_tensors(self, device, num_continuations: int):
        super()._prepare_hazard_tensors(device, num_continuations)
        cands = self.boundary_data["candidates"]
        missing = [i for i, c in enumerate(cands) if "probe_p_trace" not in c]
        if missing:
            raise ValueError(
                "boundary_data candidates missing probe_p_trace "
                f"(indices {missing}); rebuild with build_boundary_data.py."
            )
        self._bd_probe_p_trace: List[torch.Tensor] = [
            torch.tensor(c["probe_p_trace"], dtype=torch.float32, device=device)
            for c in cands
        ]

    def _resolve_hazard_fn(self):
        fn = super()._resolve_hazard_fn()

        def with_probe_p_trace(**kwargs):
            return fn(probe_p_trace=self._bd_probe_p_trace, **kwargs)

        return with_probe_p_trace
