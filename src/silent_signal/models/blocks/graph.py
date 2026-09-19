"""Mask-aware spatial graph blocks for pose tensors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphConvolution(nn.Module):
    """Aggregate neighboring joints with a fixed normalized adjacency matrix."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, T, V, C].")
        if adjacency.ndim == 2:
            aggregated = torch.matmul(adjacency, features)
        elif adjacency.ndim == 3:
            aggregated = torch.einsum("bvw,btwc->btvc", adjacency, features)
        else:
            raise ValueError("adjacency must have shape [V, V] or [B, V, V].")
        return self.projection(aggregated)


class SpatialGraphBlock(nn.Module):
    """Pre-normalized residual graph convolution followed by an MLP."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.graph = GraphConvolution(hidden_dim, hidden_dim)
        self.graph_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor, adjacency: Tensor, joint_mask: Tensor) -> Tensor:
        mask = joint_mask.unsqueeze(-1).to(dtype=features.dtype)
        features = features + self.graph_dropout(self.graph(self.graph_norm(features), adjacency))
        features = features * mask
        features = features + self.mlp(self.mlp_norm(features))
        return features * mask
