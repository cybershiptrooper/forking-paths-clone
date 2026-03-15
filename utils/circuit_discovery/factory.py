"""Factory for creating circuit discovery algorithm instances from config."""

from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.nodewise_attribution import (
    NodewiseAttribution as MaskIGNodewiseAttribution,
)
from utils.circuit_discovery.nodewise_attribution_attention import (
    NodewiseAttribution as AttentionAPIGNodewiseAttribution,
)
from utils.circuit_discovery.nodewise_activation_patching import (
    NodewiseActivationPatching,
)
from utils.circuit_discovery.edits.nodewise_patching_kv_cache import (
    NodewiseActivationPatchingKVCache,
)
from utils.circuit_discovery.edits.nodewise_patching_batch import (
    NodewiseActivationPatchingBatch,
)


ALGORITHMS = {
    "nodewise_attribution": MaskIGNodewiseAttribution,
    "nodewise_attribution_attention": AttentionAPIGNodewiseAttribution,
    "nodewise_activation_patching": NodewiseActivationPatching,
    "nodewise_activation_patching_kv_cache": NodewiseActivationPatchingKVCache,
    "nodewise_activation_patching_batch": NodewiseActivationPatchingBatch,
    # Future:
    # "subnetwork_probing": SubnetworkProbing,
    # "EAP": EdgeAttributionPatching,
}


def create_circuit_discovery(algorithm_name: str, **kwargs) -> CircuitDiscovery:
    """Create a circuit discovery instance by algorithm name.

    Args:
        algorithm_name: One of the keys in ALGORITHMS
        **kwargs: Passed to the algorithm constructor
            (model, tokenizer, layers, objective_fn, sentence_gap, etc.)

    Returns:
        CircuitDiscovery instance
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[algorithm_name](**kwargs)
