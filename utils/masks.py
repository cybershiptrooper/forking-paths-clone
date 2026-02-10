"""Mask data structures for circuit discovery results.

NodeMask: per-head, per-layer sentence-to-sentence attributions (within-layer).
EdgeMask: inter-layer edges connecting sentence representations across layers (placeholder for EAP).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from utils.utils import Sentence


@dataclass
class NodeMask:
    """Per-head, per-layer sentence-to-sentence attributions.

    Produced by nodewise attribution patching. Stores the full
    (num_heads, S, S) resolution per layer. Collapsing to (S, S)
    happens only during visualization.

    Attributes:
        scores: layer_idx -> (num_heads, S, S) attribution tensor.
            scores[layer][head][query_sent][key_sent] = attribution score.
        sentences: Sentence boundaries used for the analysis.
        sentence_texts: Decoded text for each sentence.
        metadata: Algorithm name, model, hyperparameters, prompt, etc.
    """

    scores: Dict[int, torch.Tensor]
    sentences: List[Sentence]
    sentence_texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        data = {
            "scores": {str(k): v.tolist() for k, v in self.scores.items()},
            "sentences": [{"start": s.start, "end": s.end} for s in self.sentences],
            "sentence_texts": self.sentence_texts,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NodeMask":
        with open(path) as f:
            data = json.load(f)
        scores = {int(k): torch.tensor(v) for k, v in data["scores"].items()}
        sentences = [
            Sentence(start=s["start"], end=s["end"]) for s in data["sentences"]
        ]
        return cls(
            scores=scores,
            sentences=sentences,
            sentence_texts=data.get("sentence_texts", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EdgeMask:
    """Inter-layer edges connecting sentence representations across layers.

    Placeholder for future EAP implementation. Stores attributions for edges
    of the form (src_layer, dst_layer) -> (S, S).

    Attributes:
        scores: (src_layer, dst_layer) -> (S, S) attribution tensor.
            scores[(l, l')][src_sent][dst_sent] = attribution score.
        sentences: Sentence boundaries used for the analysis.
        sentence_texts: Decoded text for each sentence.
        metadata: Algorithm name, model, hyperparameters, prompt, etc.
    """

    scores: Dict[Tuple[int, int], torch.Tensor]
    sentences: List[Sentence]
    sentence_texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        data = {
            "scores": {
                f"{k[0]},{k[1]}": v.tolist() for k, v in self.scores.items()
            },
            "sentences": [{"start": s.start, "end": s.end} for s in self.sentences],
            "sentence_texts": self.sentence_texts,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "EdgeMask":
        with open(path) as f:
            data = json.load(f)
        scores = {}
        for k, v in data["scores"].items():
            parts = k.split(",")
            key = (int(parts[0]), int(parts[1]))
            scores[key] = torch.tensor(v)
        sentences = [
            Sentence(start=s["start"], end=s["end"]) for s in data["sentences"]
        ]
        return cls(
            scores=scores,
            sentences=sentences,
            sentence_texts=data.get("sentence_texts", []),
            metadata=data.get("metadata", {}),
        )
