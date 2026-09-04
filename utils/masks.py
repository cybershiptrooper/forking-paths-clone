"""Mask dataclasses for storing circuit discovery results with JSON serialization."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
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


def build_mode_filter(
    num_prefix_sents: int,
    num_total_sents: int,
    mask_mode: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build boolean filter for mask_mode: True = frozen at 1.0, False = learnable.

    Modes:
        "prefix"     – learn only prefix-query → prefix-key (top-left block)
        "generation" – learn only generation-query → prefix-key (bottom-left block)
        "both"       – learn all-query → prefix-key (full left block)
    """
    frozen = torch.ones(
        num_total_sents, num_total_sents, dtype=torch.bool, device=device
    )
    if mask_mode == "prefix":
        frozen[:num_prefix_sents, :num_prefix_sents] = False
    elif mask_mode == "generation":
        frozen[num_prefix_sents:, :num_prefix_sents] = False
    elif mask_mode == "both":
        frozen[:, :num_prefix_sents] = False
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")
    return frozen


def build_prompt_filter(
    num_prompt_sents: int,
    num_total_sents: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build boolean filter freezing the first *num_prompt_sents* sentences.

    True = frozen at 1.0. Freezes every entry whose query OR key sentence
    is one of the first ``num_prompt_sents`` sentences (the question
    prompt, including the multiple-choice options). With this filter the
    mask can only ablate attention between reasoning sentences — it cannot
    learn the degenerate solution of removing attention to the wrong
    answer options in the prompt.
    """
    frozen = torch.zeros(
        num_total_sents, num_total_sents, dtype=torch.bool, device=device
    )
    if num_prompt_sents > 0:
        frozen[:num_prompt_sents, :] = True
        frozen[:, :num_prompt_sents] = True
    return frozen


def build_causal_filter(
    num_sents: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Build boolean mask: True where j > i (causally invalid: key after query).

    In autoregressive models, sentence i cannot attend to sentence j when j > i.
    The causal attention mask zeros these weights pre-softmax, so IG scores for
    these positions are structurally zero.  Excluding them from the permutable
    pool prevents random baselines from having a different *effective* sparsity
    than the learned mask.
    """
    i = torch.arange(num_sents, device=device)
    return i[None, :] > i[:, None]


def build_combined_filter(
    gap_filter: torch.Tensor,
    mode_filter: torch.Tensor,
    causal_filter: Optional[torch.Tensor] = None,
    prompt_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine gap, mode, causal, and prompt filters into one frozen-mask. True = frozen at 1.0."""
    combined = gap_filter | mode_filter
    if causal_filter is not None:
        combined = combined | causal_filter.to(combined.device)
    if prompt_filter is not None:
        combined = combined | prompt_filter.to(combined.device)
    return combined


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


_VALID_GRANULARITIES = {"head", "layer", "pair"}


def _count_below(scores_2d: List[List[float]], threshold: float,
                 gap_filter: Optional[torch.Tensor]) -> tuple[int, int]:
    """Count (total, below_threshold) for a single 2D score matrix."""
    total = 0
    below = 0
    for i, row in enumerate(scores_2d):
        for j, val in enumerate(row):
            if gap_filter is not None and bool(gap_filter[i, j]):
                continue
            total += 1
            if val < threshold:
                below += 1
    return total, below


@dataclass
class NodeMask(MaskResult):
    """Sentence-to-sentence attribution scores at configurable granularity.

    The *granularity* (stored in ``metadata["mask_granularity"]``) controls
    the shape of ``scores``:

    - ``"head"`` (default): ``scores[layer][head] = [[float]]``
      One score per (layer, head, src_sent, tgt_sent).
    - ``"layer"``: ``scores[layer] = [[float]]``
      One score per (layer, src_sent, tgt_sent), shared across heads.
    - ``"pair"``: ``scores = [[float]]``
      One score per (src_sent, tgt_sent), shared across layers and heads.
    - ``"column"``: ``scores = [float]``
      One score per target sentence (column), shared across all rows,
      layers, and heads. Used by column subnetwork probing.
    """

    scores: Any = field(default_factory=dict)

    @property
    def granularity(self) -> str:
        """Mask granularity: ``"head"``, ``"layer"``, ``"pair"``, or ``"column"``."""
        return self.metadata.get("mask_granularity", "head")

    # ------------------------------------------------------------------
    # Sparsity
    # ------------------------------------------------------------------

    def sparsity(
        self,
        threshold: float,
        gap_filter: Optional[torch.Tensor] = None,
        sentence_gap: Optional[int] = None,
    ) -> float:
        """Fraction of unique score entries below *threshold*.

        Counts only unique learnable parameters (not broadcasted copies).
        """
        if gap_filter is None and sentence_gap is not None:
            gap_filter = build_gap_filter(len(self.sentences), sentence_gap)

        g = self.granularity
        total = 0
        below = 0

        if g == "head":
            for layer_scores in self.scores.values():
                for head_scores in layer_scores.values():
                    t, b = _count_below(head_scores, threshold, gap_filter)
                    total += t
                    below += b
        elif g == "layer":
            for layer_scores in self.scores.values():
                t, b = _count_below(layer_scores, threshold, gap_filter)
                total += t
                below += b
        elif g == "pair":
            t, b = _count_below(self.scores, threshold, gap_filter)
            total += t
            below += b

        return below / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_to_sparse(matrix: List[List[float]], combined_filter: torch.Tensor) -> List[List]:
        """Convert a 2D score matrix to sparse [i, j, score] triples.

        Only positions where ``combined_filter[i][j]`` is False (i.e. active /
        learnable) are stored.
        """
        triples: List[List] = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if not bool(combined_filter[i, j]):
                    triples.append([i, j, val])
        return triples

    @staticmethod
    def _sparse_to_matrix(triples: List[List], num_sents: int, fill_value: float = 1.0) -> List[List[float]]:
        """Reconstruct a full S x S matrix from sparse [i, j, score] triples.

        Positions not present in *triples* are filled with *fill_value* (1.0,
        matching the frozen-at-1.0 convention used by the combined filter).
        """
        matrix = [[fill_value] * num_sents for _ in range(num_sents)]
        for i, j, score in triples:
            matrix[i][j] = score
        return matrix

    def _build_combined_filter_from_metadata(self) -> Optional[torch.Tensor]:
        """Build the combined filter from metadata fields, or return None."""
        num_sents = len(self.sentences)
        sentence_gap = self.metadata.get("sentence_gap")
        mask_mode = self.metadata.get("mask_mode")
        num_prefix_sents = self.metadata.get("num_prefix_sentences")
        num_frozen_prompt_sents = self.metadata.get("num_frozen_prompt_sentences", 0)

        gap_filter = build_gap_filter(num_sents, sentence_gap)
        causal_filter = build_causal_filter(num_sents)

        if mask_mode is not None and num_prefix_sents is not None:
            mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode)
        else:
            mode_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool)

        prompt_filter = (
            build_prompt_filter(num_frozen_prompt_sents, num_sents)
            if num_frozen_prompt_sents
            else None
        )

        return build_combined_filter(
            gap_filter, mode_filter, causal_filter, prompt_filter
        )

    def to_json(self, path: str, sparse: bool = False):
        """Serialize to JSON file.

        Parameters
        ----------
        path : str
            Destination file path.
        sparse : bool, optional
            If True (default), store only active positions as ``[i, j, score]``
            triples instead of full S x S matrices. This is strictly better for
            storage and is backward-compatible (``from_json`` detects the format
            automatically).
        """
        g = self.granularity

        if sparse:
            combined_filter = self._build_combined_filter_from_metadata()

            if g == "head":
                serialized_scores = {
                    str(layer): {
                        str(head): self._matrix_to_sparse(scores, combined_filter)
                        for head, scores in heads.items()
                    }
                    for layer, heads in self.scores.items()
                }
            elif g == "layer":
                serialized_scores = {
                    str(layer): self._matrix_to_sparse(scores, combined_filter)
                    for layer, scores in self.scores.items()
                }
            elif g == "pair":
                serialized_scores = self._matrix_to_sparse(self.scores, combined_filter)
            else:  # "column" — 1D, no sparse encoding needed
                serialized_scores = self.scores
        else:
            if g == "head":
                serialized_scores = {
                    str(layer): {
                        str(head): scores
                        for head, scores in heads.items()
                    }
                    for layer, heads in self.scores.items()
                }
            elif g == "layer":
                serialized_scores = {
                    str(layer): scores
                    for layer, scores in self.scores.items()
                }
            elif g == "pair":
                serialized_scores = self.scores
            else:  # "column"
                serialized_scores = self.scores

        data = {
            "mask_type": "NodeMask",
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "layers": self.layers,
            "sentences": self.sentences,
            "objective_name": self.objective_name,
            "metadata": self.metadata,
            "scores": serialized_scores,
        }
        if sparse:
            data["scores_format"] = "sparse"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "NodeMask":
        """Deserialize from JSON file.

        Supports both dense (legacy) and sparse formats.  The format is
        auto-detected via the ``scores_format`` field in the JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        g = metadata.get("mask_granularity", "head")
        scores_format = data.get("scores_format", "dense")

        raw = data["scores"]

        if scores_format == "sparse":
            num_sents = len(data["sentences"])

            if g == "head":
                scores = {
                    int(layer): {
                        int(head): cls._sparse_to_matrix(triples, num_sents)
                        for head, triples in heads.items()
                    }
                    for layer, heads in raw.items()
                }
            elif g == "layer":
                scores = {
                    int(layer): cls._sparse_to_matrix(triples, num_sents)
                    for layer, triples in raw.items()
                }
            elif g == "pair":
                scores = cls._sparse_to_matrix(raw, num_sents)
            else:  # "column" — 1D list, no sparse decoding
                scores = raw
        else:
            # Legacy dense format
            if g == "head":
                scores = {
                    int(layer): {
                        int(head): scores
                        for head, scores in heads.items()
                    }
                    for layer, heads in raw.items()
                }
            elif g == "layer":
                scores = {int(layer): scores for layer, scores in raw.items()}
            elif g == "pair":
                scores = raw  # already a 2D list
            else:  # "column"
                scores = raw  # already a 1D list

        return cls(
            model_name=data["model_name"],
            algorithm=data["algorithm"],
            layers=data["layers"],
            sentences=data["sentences"],
            objective_name=data["objective_name"],
            metadata=metadata,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Threshold computation from target sparsities
    # ------------------------------------------------------------------

    def _collect_active_scores(
        self, combined_filter: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Collect all learnable (non-frozen) score values as a flat list."""
        g = self.granularity
        values: List[float] = []

        def _gather_2d(scores_2d: List[List[float]]) -> None:
            for i, row in enumerate(scores_2d):
                for j, val in enumerate(row):
                    if combined_filter is not None and bool(combined_filter[i, j]):
                        continue
                    values.append(val)

        if g == "head":
            for layer_heads in self.scores.values():
                for head_scores in layer_heads.values():
                    _gather_2d(head_scores)
        elif g == "layer":
            for layer_scores in self.scores.values():
                _gather_2d(layer_scores)
        else:  # "pair"
            _gather_2d(self.scores)

        return values

    def thresholds_for_sparsities(
        self,
        target_sparsities: Optional[List[float]] = None,
        combined_filter: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Compute score thresholds that achieve the given sparsity levels.

        Parameters
        ----------
        target_sparsities : list of float, optional
            Desired sparsity fractions in [0, 1]. Defaults to a standard set
            covering 0% through 100%.
        combined_filter : torch.Tensor, optional
            Boolean filter (True = frozen). If None, built from metadata.

        Returns
        -------
        list of float
            Sorted, deduplicated thresholds (one per achievable sparsity).
        """
        if target_sparsities is None:
            target_sparsities = [
                0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
            ]

        if combined_filter is None:
            combined_filter = self._build_combined_filter_from_metadata()

        values = self._collect_active_scores(combined_filter)
        if not values:
            return [0.0]

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        thresholds: List[float] = []

        for sp in target_sparsities:
            if sp <= 0.0:
                # 0% sparsity: threshold below the minimum score
                thresholds.append(sorted_vals[0] - 1.0)
            elif sp >= 1.0:
                # 100% sparsity: threshold above the maximum score
                thresholds.append(sorted_vals[-1] + 1.0)
            else:
                # Index such that (idx / n) ≈ sp
                idx = int(sp * n)
                idx = min(idx, n - 1)
                thresholds.append(sorted_vals[idx])

        # Deduplicate and sort
        return sorted(set(thresholds))

    # ------------------------------------------------------------------
    # Aggregation helpers (used by visualization)
    # ------------------------------------------------------------------

    def get_layer_aggregated(self, layer: int, aggregation: str = "mean") -> List[List[float]]:
        """Aggregate scores for a single layer into a 2D (S, S) matrix."""
        import numpy as np

        g = self.granularity
        if g == "pair":
            return list(self.scores)  # same for every layer
        if g == "layer":
            if layer not in self.scores:
                raise ValueError(f"Layer {layer} not in mask. Available: {list(self.scores.keys())}")
            return list(self.scores[layer])
        # g == "head"
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
        """Aggregate scores across all layers (and heads) into a 2D (S, S) matrix."""
        import numpy as np

        g = self.granularity
        if g == "pair":
            return list(self.scores)
        if g == "layer":
            arrays = [np.array(s) for s in self.scores.values()]
        else:  # "head"
            arrays = []
            for layer in self.scores:
                for head_scores in self.scores[layer].values():
                    arrays.append(np.array(head_scores))

        stacked = np.stack(arrays, axis=0)
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

        Only available for ``granularity == "head"``.
        """
        import numpy as np

        if self.granularity != "head":
            raise ValueError(
                f"get_head_importance requires granularity='head', got '{self.granularity}'"
            )
        result = {}
        for head, scores in self.scores[layer].items():
            arr = np.array(scores)
            arr = np.where(arr >= threshold, arr, 0.0)
            result[head] = float(arr.sum())
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Expansion to per-head format (for mask application at eval time)
    # ------------------------------------------------------------------

    def expand_to_per_head(
        self, layers: List[int], num_heads: int, device: torch.device = torch.device("cpu"),
    ) -> Dict[int, torch.Tensor]:
        """Expand scores to ``{layer: (num_heads, S, S)}`` tensors.

        Broadcasts from native granularity so that downstream code
        (``install_mask_hooks``, ``expand_sentence_mask_to_tokens``) can
        always receive ``(H, S, S)`` tensors.
        """
        import numpy as np

        g = self.granularity
        result: Dict[int, torch.Tensor] = {}

        if g == "pair":
            shared = torch.tensor(np.array(self.scores, dtype=float), device=device)
            expanded = shared.unsqueeze(0).expand(num_heads, -1, -1)
            for layer in layers:
                result[layer] = expanded
        elif g == "layer":
            for layer in layers:
                layer_scores = torch.tensor(
                    np.array(self.scores[layer], dtype=float), device=device,
                )
                result[layer] = layer_scores.unsqueeze(0).expand(num_heads, -1, -1)
        else:  # "head"
            for layer in layers:
                head_arrays = [
                    np.array(self.scores[layer][h], dtype=float)
                    for h in range(num_heads)
                ]
                result[layer] = torch.tensor(
                    np.stack(head_arrays, axis=0), device=device,
                )
        return result


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
