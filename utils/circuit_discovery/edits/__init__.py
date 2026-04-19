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
