from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class EdgewiseMask:
    model_name: str
    layers: List[int]
    num_heads: int
    sentence_chunks: List[Dict[str, int]]
    gap: int
    analysis_timestep: int
    mask_values: List  # nested list: [L][H][S][S]
    prompt: str
    prompt_len: int
    metadata: Optional[Dict[str, Any]] = None

    def to_json(self, path: str) -> None:
        payload = asdict(self)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "EdgewiseMask":
        with open(path, "r") as f:
            data = json.load(f)
        return EdgewiseMask(**data)


@dataclass
class NodewiseMask:
    model_name: str
    layers: List[int]
    nodes: List[Dict[str, Any]]
    mask_values: List
    metadata: Optional[Dict[str, Any]] = None

    def to_json(self, path: str) -> None:
        payload = asdict(self)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "NodewiseMask":
        with open(path, "r") as f:
            data = json.load(f)
        return NodewiseMask(**data)
