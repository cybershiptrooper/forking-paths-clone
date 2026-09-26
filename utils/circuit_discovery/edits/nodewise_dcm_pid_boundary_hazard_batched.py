"""DCM + PID training with the batched boundary-hazard step.

Combines, by inheritance only (no existing class is modified):

- ``NodewiseDCMPIDSDPA.discover``: a deterministic mask parameter in
  [0, 1] with a PID controller ramping the hard zero count linearly
  across training, snapshotting the mask whenever the achieved sparsity
  crosses a requested target — one training run yields masks at several
  sparsities;
- ``NodewiseSubnetworkProbingBoundaryHazardBatched._step_global``: the
  single-pass boundary-hazard step that encodes the shared prefix once
  per step and forwards candidates in right-padded batches against the
  cached prefix K/V.

Method resolution order does all the work: ``discover`` comes from the
DCM+PID class (which does not define ``_step_global``), the hazard step
and its tensor preparation come from the batched hazard chain, and the
``__init__`` chain is cooperative (each class pops its own kwargs and
calls ``super().__init__``).

Registered as ``nodewise_dcm_pid_boundary_hazard_batched`` and (with the
continuous answer guard) ``nodewise_dcm_pid_boundary_hazard_probe_weighted_batched``.
"""

from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from utils.circuit_discovery.edits.nodewise_dcm_pid_sdpa import (
    NodewiseDCMPIDSDPA,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard_batched import (
    NodewiseSubnetworkProbingBoundaryHazardBatched,
    NodewiseSubnetworkProbingBoundaryHazardProbeWeightedBatched,
)


class _AlwaysCheckpointLayers:
    """Per-layer activation checkpointing WITHOUT HuggingFace-level
    gradient checkpointing.

    Run these classes with ``gradient_checkpointing: true`` (the
    empirically working configuration; verified across the 500- and
    600-step pilot runs: task loss live, no cache-guard warnings).  With
    checkpointing disabled the model stays in eval mode and the
    memory-efficient SDPA backward through the circuit-mask bias fails
    with "LSE is not correctly aligned (strideH)"; without this mixin's
    unconditional per-layer checkpointing, early configurations showed
    ``GradientCheckpointingLayer.__call__`` nulling the
    ``past_key_values`` kwarg ("Caching is incompatible with gradient
    checkpointing"), which discards the shared prefix K/V the batched
    hazard step passes to every candidate forward and silently
    disconnects the mask from the loss (bit-identical task loss for
    hundreds of steps).  Always verify a new configuration by checking
    that the task loss moves over the first ~20 steps.
    """

    def _maybe_checkpoint(self, fn, *args):
        return _torch_checkpoint(fn, *args, use_reentrant=False)


class NodewiseDCMPIDBoundaryHazardBatched(
    _AlwaysCheckpointLayers,
    NodewiseDCMPIDSDPA,
    NodewiseSubnetworkProbingBoundaryHazardBatched,
):
    """Deterministic PID-ramped mask trained on a boundary-hazard objective."""


class NodewiseDCMPIDBoundaryHazardProbeWeightedBatched(
    _AlwaysCheckpointLayers,
    NodewiseDCMPIDSDPA,
    NodewiseSubnetworkProbingBoundaryHazardProbeWeightedBatched,
):
    """Same, with the continuous (probe-probability-weighted) answer guard."""
