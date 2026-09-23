"""Zero the output of chosen attention heads.

New file (2026-09-17). Nothing in the existing mask machinery is changed:
this registers forward pre-hooks on the output projection of each layer
that hold a chosen head, and multiplies the head's slice of the projection
input by zero. Every other head of the layer, the MLP and the residual
stream are untouched. The hooks stay active for every forward call until
``remove()`` is called, so they also cover every generated token in a
rollout.

Usage::

    with HeadOutputZeroing(model, [(18, 4), (16, 24)]):
        logits = model(input_ids).logits

or ``z = HeadOutputZeroing(model, heads); z.install(); ...; z.remove()``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch

from utils.utils import get_attention_module


def head_slice_size(model) -> int:
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return int(hd)


class HeadOutputZeroing:
    """Context manager that zeroes the output of the given (layer, head) pairs."""

    def __init__(self, model: torch.nn.Module, heads: Iterable[tuple[int, int]]):
        self.model = model
        self.heads = [(int(l), int(h)) for l, h in heads]
        self.handles = []
        self.head_dim = head_slice_size(model)
        self.num_heads = int(model.config.num_attention_heads)

    def _masks_per_layer(self) -> dict[int, torch.Tensor]:
        by_layer: dict[int, list[int]] = defaultdict(list)
        for l, h in self.heads:
            if not 0 <= h < self.num_heads:
                raise ValueError(f"head {h} out of range for {self.num_heads} heads")
            by_layer[l].append(h)
        out = {}
        for l, hs in by_layer.items():
            attn = get_attention_module(self.model, l)
            in_features = attn.o_proj.in_features
            if in_features != self.num_heads * self.head_dim:
                raise ValueError(f"o_proj in_features {in_features} != heads*head_dim "
                                 f"{self.num_heads * self.head_dim}")
            keep = torch.ones(in_features, dtype=attn.o_proj.weight.dtype,
                              device=attn.o_proj.weight.device)
            for h in hs:
                keep[h * self.head_dim:(h + 1) * self.head_dim] = 0
            out[l] = keep
        return out

    def install(self):
        if self.handles:
            return self
        for l, keep in self._masks_per_layer().items():
            attn = get_attention_module(self.model, l)

            def pre_hook(module, args, _keep=keep):
                x = args[0]
                return (x * _keep.to(x.dtype),) + tuple(args[1:])

            self.handles.append(attn.o_proj.register_forward_pre_hook(pre_hook))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False
