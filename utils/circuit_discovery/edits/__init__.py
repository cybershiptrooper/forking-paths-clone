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

register_patching_method(
    NodewiseActivationPatchingKVCache, "nodewise_activation_patching_kv_cache"
)
register_patching_method(
    NodewiseActivationPatchingBatch, "nodewise_activation_patching_batch"
)
register_patching_method(
    NodewiseAttributionMemory, "nodewise_attribution_memory"
)
