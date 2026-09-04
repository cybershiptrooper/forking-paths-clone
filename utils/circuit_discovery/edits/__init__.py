"""Auto-register patching methods from the edits package."""

from utils.circuit_discovery.factory import register_patching_method

from utils.circuit_discovery.edits.nodewise_patching_kv_cache import (
    NodewiseActivationPatchingKVCache,
)
from utils.circuit_discovery.edits.nodewise_patching_batch import (
    NodewiseActivationPatchingBatch,
)
from utils.circuit_discovery.edits.nodewise_attribution_memory import (
    NodewiseAttribution as NodewiseAttributionMemory,
)
from utils.circuit_discovery.edits.nodewise_patching_batched_probes import (
    NodewiseActivationPatchingBatchedProbes,
)
from utils.circuit_discovery.edits.nodewise_patching_flash import (
    NodewiseActivationPatchingFlash,
)
from utils.circuit_discovery.edits.nodewise_attribution_sdpa import (
    NodewiseAttributionSDPA,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
)
from utils.circuit_discovery.edits.nodewise_dcm_pid_sdpa import (
    NodewiseDCMPIDSDPA,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard import (
    NodewiseSubnetworkProbingBoundaryHazard,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard_probe_weighted import (
    NodewiseSubnetworkProbingBoundaryHazardProbeWeighted,
)
from utils.circuit_discovery.edits.nodewise_dcm_pid_boundary_hazard_batched import (
    NodewiseDCMPIDBoundaryHazardBatched,
    NodewiseDCMPIDBoundaryHazardProbeWeightedBatched,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_boundary_hazard_batched import (
    NodewiseSubnetworkProbingBoundaryHazardBatched,
    NodewiseSubnetworkProbingBoundaryHazardProbeWeightedBatched,
)
from utils.circuit_discovery.edits.column_subnetwork_probing import (
    ColumnSubnetworkProbing,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
)

register_patching_method(
    NodewiseActivationPatchingKVCache, "nodewise_activation_patching_kv_cache"
)
register_patching_method(
    NodewiseActivationPatchingBatch, "nodewise_activation_patching_batch"
)
register_patching_method(
    NodewiseAttributionMemory, "nodewise_attribution_memory"
)
register_patching_method(
    NodewiseActivationPatchingBatchedProbes,
    "nodewise_activation_patching_batched_probes",
)
register_patching_method(
    NodewiseActivationPatchingFlash, "nodewise_activation_patching_flash"
)
register_patching_method(
    NodewiseAttributionSDPA, "nodewise_attribution_sdpa"
)
register_patching_method(
    NodewiseSubnetworkProbingSDPA, "nodewise_subnetwork_probing_sdpa"
)
register_patching_method(
    NodewiseDCMPIDSDPA, "nodewise_dcm_pid_sdpa"
)
register_patching_method(
    NodewiseSubnetworkProbingBoundaryHazard,
    "nodewise_subnetwork_probing_boundary_hazard",
)
register_patching_method(
    NodewiseSubnetworkProbingBoundaryHazardProbeWeighted,
    "nodewise_subnetwork_probing_boundary_hazard_probe_weighted",
)
register_patching_method(
    NodewiseSubnetworkProbingBoundaryHazardBatched,
    "nodewise_subnetwork_probing_boundary_hazard_batched",
)
register_patching_method(
    NodewiseSubnetworkProbingBoundaryHazardProbeWeightedBatched,
    "nodewise_subnetwork_probing_boundary_hazard_probe_weighted_batched",
)
register_patching_method(
    NodewiseDCMPIDBoundaryHazardBatched,
    "nodewise_dcm_pid_boundary_hazard_batched",
)
register_patching_method(
    NodewiseDCMPIDBoundaryHazardProbeWeightedBatched,
    "nodewise_dcm_pid_boundary_hazard_probe_weighted_batched",
)
register_patching_method(
    ColumnSubnetworkProbing, "column_subnetwork_probing"
)
register_patching_method(
    NodewiseSubnetworkProbingHCBatched,
    "nodewise_subnetwork_probing_hc_batched",
)
