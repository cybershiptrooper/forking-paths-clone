from __future__ import annotations

from typing import Any

from utils.circuit_discovery.eap import EAPDiscovery


def get_circuit_discovery_algorithm(name: str, **kwargs: Any):
    name = name.lower()
    if name == "eap":
        return EAPDiscovery(**kwargs)
    raise ValueError(f"Unknown circuit discovery algorithm: {name}")
