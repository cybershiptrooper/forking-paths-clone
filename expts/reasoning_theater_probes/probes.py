"""Probes from "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"
(Boppana et al., arXiv:2603.05488).

LinearProbe   — last-token pooling + a single linear classifier head.
AttentionProbe — single-query attention pooling over the prefix, optional MLP value head.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Optional[Sequence[int]] = None) -> torch.Tensor:
        if lengths is not None:
            last_hidden = torch.stack([x[i, lengths[i] - 1] for i in range(x.shape[0])])
        else:
            last_hidden = x[:, -1]
        return self.linear(last_hidden)


class AttentionProbe(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float32,
        mlp: bool = False,
        mlp_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.q = nn.Linear(in_features, 1, dtype=dtype)
        if self.mlp:
            self.v_up = nn.Linear(in_features, mlp_hidden_dim, dtype=dtype)
            self.v_relu = nn.ReLU()
            self.v_down = nn.Linear(mlp_hidden_dim, out_features, dtype=dtype)
        else:
            self.v = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Optional[Sequence[int]] = None) -> torch.Tensor:
        attn_logits = self.q(x).squeeze(-1)
        if lengths is not None:
            mask = torch.zeros_like(attn_logits)
            for i, length in enumerate(lengths):
                if length > 0:
                    mask[i, :length] = 1
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1)
        if self.mlp:
            v_up_out = self.v_up(x)
            aggregated = torch.sum(attn_weights.unsqueeze(-1) * v_up_out, dim=1)
            return self.v_down(self.v_relu(aggregated))
        values = self.v(x)
        return torch.sum(attn_weights.unsqueeze(-1) * values, dim=1)


def build_probe(probe_class: str, in_features: int, out_features: int, **kwargs) -> nn.Module:
    if probe_class == "linear":
        return LinearProbe(in_features, out_features)
    if probe_class == "attention":
        return AttentionProbe(
            in_features,
            out_features,
            mlp=kwargs.get("mlp", False),
            mlp_hidden_dim=kwargs.get("mlp_hidden_dim", 32),
        )
    raise ValueError(f"Unknown probe_class: {probe_class}")
