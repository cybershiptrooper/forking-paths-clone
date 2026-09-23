"""Subnetwork probing over a named learnable region of the sentence grid.

``NodewiseSubnetworkProbingRegion`` is the HC-batched subnetwork-probing
trainer (:class:`NodewiseSubnetworkProbingHCBatched`) with one addition:
``discover`` accepts ``learnable_region`` and ``num_prompt_sentences`` and
freezes every cell outside that region at 1.0, on top of the usual gap /
mode / causal filters. The parent classes are not modified; this variant
wraps the parent's ``discover`` and, for the duration of that call, makes
the combined-filter builder the parent uses OR in the region filter.

Regions (see :func:`utils.masks.build_region_filter`):

- ``None``: the parent's default pool (no change in behaviour).
- ``"prompt_to_trace"``: only the cells where a reasoning sentence (query)
  reads a prompt sentence (key) are learnable. Reasoning-to-reasoning
  and prompt-to-prompt cells stay at 1.0. This is the pool for the
  question "which reads of the prompt carry a cue in the prompt into the
  answer".
- ``"trace_to_trace"``: only reasoning-to-reasoning cells are learnable.

``frozen_key_sentences`` (a list of sentence indices) freezes every read of
those sentences on top of the region, e.g. the answer-option chunks.

Registered as ``nodewise_subnetwork_probing_region`` in ``edits/__init__.py``. Config:

    masking_algorithm: nodewise_subnetwork_probing_region
    learnable_region: prompt_to_trace
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

import utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa as _sdpa_module
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
)
from utils.masks import NodeMask, build_region_filter


@contextlib.contextmanager
def _region_filter_applied(region_filter: torch.Tensor):
    """Make the parent's combined-filter builder OR in ``region_filter``.

    The parent's ``discover`` resolves ``build_combined_filter`` from its
    own module namespace, so replacing that name for the duration of the
    call changes the frozen set the parent trains and reads out with, and
    nothing else. The original builder is restored on exit.
    """
    original = _sdpa_module.build_combined_filter

    def with_region(gap_filter, mode_filter, causal_filter=None, prompt_filter=None):
        combined = original(gap_filter, mode_filter, causal_filter, prompt_filter)
        return combined | region_filter.to(combined.device)

    _sdpa_module.build_combined_filter = with_region
    try:
        yield
    finally:
        _sdpa_module.build_combined_filter = original


class NodewiseSubnetworkProbingRegion(NodewiseSubnetworkProbingHCBatched):
    """HC-batched subnetwork probing whose learnable pool is a named region."""

    def discover(
        self,
        *args,
        learnable_region: Optional[str] = None,
        num_prompt_sentences: int = 0,
        frozen_key_sentences=None,
        **kwargs,
    ) -> NodeMask:
        sentences = kwargs.get("sentences", args[1] if len(args) > 1 else None)
        if sentences is None:
            raise ValueError("discover() needs `sentences`")
        num_sents = len(sentences)
        device = next(self.model.parameters()).device
        region_filter = build_region_filter(
            learnable_region, num_prompt_sentences, num_sents, device=device,
            frozen_key_sentences=frozen_key_sentences,
        )
        if region_filter is None:
            node_mask = super().discover(*args, **kwargs)
        else:
            print(
                f"  Learnable region {learnable_region!r}: "
                f"{int((~region_filter).sum().item())} candidate cells before the "
                f"gap / mode / causal filters ({num_prompt_sentences} prompt sentences)"
            )
            with _region_filter_applied(region_filter):
                node_mask = super().discover(*args, **kwargs)
        node_mask.algorithm = "nodewise_subnetwork_probing_region"
        node_mask.metadata["learnable_region"] = learnable_region
        node_mask.metadata["num_prompt_sentences"] = num_prompt_sentences
        node_mask.metadata["frozen_key_sentences"] = list(frozen_key_sentences or [])
        return node_mask

