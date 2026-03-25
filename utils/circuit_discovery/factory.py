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


ALGORITHMS = {
    "nodewise_attribution": MaskIGNodewiseAttribution,
    "nodewise_attribution_attention": AttentionAPIGNodewiseAttribution,
    "nodewise_activation_patching": NodewiseActivationPatching,
}


def register_patching_method(cls, arg_name: str, *, aliases: list[str] | None = None):
    """Dynamically register a circuit discovery algorithm.

    Args:
        cls: The algorithm class (must be a subclass of CircuitDiscovery).
        arg_name: Primary name used as the --masking_algorithm CLI value.
        aliases: Optional alternative names that also map to this class.
    """
    if not (isinstance(cls, type) and issubclass(cls, CircuitDiscovery)):
        raise TypeError(
            f"{cls!r} is not a subclass of CircuitDiscovery"
        )
    ALGORITHMS[arg_name] = cls
    for alias in aliases or []:
        ALGORITHMS[alias] = cls


def get_available_algorithms() -> list[str]:
    """Return sorted list of registered algorithm names."""
    return sorted(ALGORITHMS.keys())


# Auto-register patching methods from the edits package
import utils.circuit_discovery.edits  # noqa: F401, E402


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
            f"Available: {get_available_algorithms()}"
        )
    return ALGORITHMS[algorithm_name](**kwargs)
