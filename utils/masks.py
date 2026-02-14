"""Mask dataclasses for storing circuit discovery results with JSON serialization."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import torch


def build_gap_filter(
    num_sents: int, sentence_gap: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Build boolean mask: True where |i - j| < gap (always-on, not learnable)."""
    if sentence_gap is None or sentence_gap <= 0:
        return torch.zeros(num_sents, num_sents, dtype=torch.bool, device=device)
    i = torch.arange(num_sents, device=device)
    return (i[:, None] - i[None, :]).abs() < sentence_gap


def apply_gap_filter(
    mask: torch.Tensor, gap_filter: Optional[torch.Tensor], fill_value: float = 1.0
) -> torch.Tensor:
    """Force gap-filtered entries to a fixed value (default 1.0)."""
    if gap_filter is None:
        return mask
    gap = gap_filter.to(device=mask.device, dtype=mask.dtype)
    while gap.dim() < mask.dim():
        gap = gap.unsqueeze(0)
    return gap * fill_value + (1.0 - gap) * mask


@dataclass
class MaskResult(ABC):
    """Base fields shared by all mask types."""

    model_name: str
    algorithm: str  # "nodewise_attribution", "EAP", etc.
    layers: List[int]
    sentences: List[dict]  # [{"start": int, "end": int, "text": str}, ...]
    objective_name: str  # "kl_divergence"
    metadata: dict = field(default_factory=dict)

    @abstractmethod
    def sparsity(self, threshold: float) -> float:
        """Fraction of entries with score < threshold.

        Note: This uses the *signed* score, not |score|. With the default
        negated scores (positive = helpful / reduces KL), this measures how
        many entries fall below an importance cutoff.
        """
        ...

    def to_json(self, path: str):
        """Serialize to JSON file."""
        data = asdict(self)
        data["mask_type"] = type(self).__name__
        # Convert int keys to strings for JSON compatibility
        if hasattr(self, "scores") and isinstance(self.scores, dict):
            data["scores"] = {
                str(k): v for k, v in data["scores"].items()
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    @abstractmethod
    def from_json(cls, path: str):
        ...


@dataclass
class NodeMask(MaskResult):
    """Per-head, sentence-to-sentence attribution scores.

    Each attention head is a separate node in the circuit.
    scores[layer_idx][head_idx] = 2D list (num_sents x num_sents)
    """

    scores: Dict[int, Dict[int, List[List[float]]]] = field(default_factory=dict)

    def sparsity(
        self,
        threshold: float,
        gap_filter: Optional[torch.Tensor] = None,
        sentence_gap: Optional[int] = None,
    ) -> float:
        """Fraction of entries with score < threshold.

        Note: Signed thresholding (not absolute value). This matches the
        default convention where positive scores reduce KL.
        """
        if gap_filter is None and sentence_gap is not None:
            gap_filter = build_gap_filter(len(self.sentences), sentence_gap)

        total = 0
        below = 0
        for layer_scores in self.scores.values():
            for head_scores in layer_scores.values():
                for i, row in enumerate(head_scores):
                    for j, val in enumerate(row):
                        if gap_filter is not None and bool(gap_filter[i, j]):
                            continue
                        total += 1
                        if val < threshold:
                            below += 1
        return below / total if total > 0 else 0.0

    def to_json(self, path: str):
        """Serialize to JSON file with string keys."""
        data = {
            "mask_type": "NodeMask",
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "layers": self.layers,
            "sentences": self.sentences,
            "objective_name": self.objective_name,
            "metadata": self.metadata,
            "scores": {
                str(layer): {
                    str(head): scores
                    for head, scores in heads.items()
                }
                for layer, heads in self.scores.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NodeMask":
        """Deserialize from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        scores = {
            int(layer): {
                int(head): scores
                for head, scores in heads.items()
            }
            for layer, heads in data["scores"].items()
        }
        return cls(
            model_name=data["model_name"],
            algorithm=data["algorithm"],
            layers=data["layers"],
            sentences=data["sentences"],
            objective_name=data["objective_name"],
            metadata=data.get("metadata", {}),
            scores=scores,
        )

    def get_layer_aggregated(self, layer: int, aggregation: str = "mean") -> List[List[float]]:
        """Aggregate across heads for a single layer. Used by visualization."""
        import numpy as np

        if layer not in self.scores:
            raise ValueError(f"Layer {layer} not in mask. Available: {list(self.scores.keys())}")
        heads = self.scores[layer]
        arrays = [np.array(scores) for scores in heads.values()]
        stacked = np.stack(arrays, axis=0)  # (num_heads, num_sents, num_sents)
        if aggregation == "mean":
            return stacked.mean(axis=0).tolist()
        elif aggregation == "max":
            return stacked.max(axis=0).tolist()
        elif aggregation == "sum":
            return stacked.sum(axis=0).tolist()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_all_layers_aggregated(self, aggregation: str = "mean") -> List[List[float]]:
        """Aggregate across all layers and heads. Used by visualization."""
        import numpy as np

        all_arrays = []
        for layer in self.scores:
            for head_scores in self.scores[layer].values():
                all_arrays.append(np.array(head_scores))
        stacked = np.stack(all_arrays, axis=0)
        if aggregation == "mean":
            return stacked.mean(axis=0).tolist()
        elif aggregation == "max":
            return stacked.max(axis=0).tolist()
        elif aggregation == "sum":
            return stacked.sum(axis=0).tolist()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_head_importance(self, layer: int, threshold: float = 0.0) -> Dict[int, float]:
        """Rank heads by total attribution at a layer (after thresholding).

        Only scores >= threshold contribute to importance.
        """
        import numpy as np

        result = {}
        for head, scores in self.scores[layer].items():
            arr = np.array(scores)
            arr = np.where(arr >= threshold, arr, 0.0)
            result[head] = float(arr.sum())
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


@dataclass
class EdgeMask(MaskResult):
    """For EAP: inter-layer edge masks (unimplemented).

    Would contain cross-layer connection scores in addition to
    within-layer attention edges.
    """

    scores: Dict = field(default_factory=dict)

    def sparsity(self, threshold: float) -> float:
        raise NotImplementedError("EdgeMask is for EAP, not yet implemented")

    def to_json(self, path: str):
        raise NotImplementedError("EdgeMask is for EAP, not yet implemented")

    @classmethod
    def from_json(cls, path: str) -> "EdgeMask":
        raise NotImplementedError("EdgeMask is for EAP, not yet implemented")


def load_mask(path: str) -> MaskResult:
    """Load any mask type from JSON, auto-detecting the type."""
    with open(path, "r") as f:
        data = json.load(f)
    mask_type = data.get("mask_type", "NodeMask")
    if mask_type == "NodeMask":
        return NodeMask.from_json(path)
    elif mask_type == "EdgeMask":
        return EdgeMask.from_json(path)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
