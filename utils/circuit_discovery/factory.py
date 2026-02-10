"""Factory to instantiate circuit discovery algorithms by name."""

from utils.circuit_discovery.base import CircuitDiscoveryAlgorithm
from utils.circuit_discovery.nodewise_attribution import NodewiseAttributionDiscovery

ALGORITHMS = {
    "nodewise_attribution": NodewiseAttributionDiscovery,
}


def get_algorithm(name: str) -> CircuitDiscoveryAlgorithm:
    """Get algorithm instance by name.

    Args:
        name: Algorithm identifier (e.g., "nodewise_attribution").

    Returns:
        An instance of the requested algorithm.

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[name]()
