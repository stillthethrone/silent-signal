"""Graph-spatial-temporal encoder for fixed pose graph caches."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from silent_signal.models.blocks.graph import SpatialGraphBlock
from silent_signal.models.blocks.temporal import TemporalConvBlock


class PoseGraphEncoder(nn.Module):
    """Encode `[B, T, V, C]` pose graphs into one vector per clip."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_nodes: int,
        max_frames: int,
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        temporal_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(input_dim, num_nodes, max_frames, hidden_dim, embedding_dim, num_blocks) < 1:
            raise ValueError("All graph encoder dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.max_frames = max_frames
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, num_nodes, hidden_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, max_frames, 1, hidden_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.spatial_blocks = nn.ModuleList(
            SpatialGraphBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalConvBlock(
                hidden_dim,
                kernel_size=temporal_kernel,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        self._validate_inputs(features, joint_mask, frame_mask, adjacency)
        frames = features.shape[1]
        mask = joint_mask.unsqueeze(-1).to(dtype=features.dtype)
        encoded = self.input_projection(features)
        encoded = encoded + self.joint_embedding + self.time_embedding[:, :frames]
        encoded = self.input_dropout(encoded) * mask
        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks, strict=True):
            encoded = spatial(encoded, adjacency, joint_mask)
            encoded = temporal(encoded, joint_mask)

        joint_weights = joint_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        per_frame = (encoded * joint_weights).sum(dim=2) / joint_weights.sum(dim=2).clamp_min(1.0)
        effective_frames = frame_mask & joint_mask.any(dim=2)
        frame_weights = effective_frames.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (per_frame * frame_weights).sum(dim=1) / frame_weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)

    def _validate_inputs(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, T, V, C].")
        batch, frames, nodes, channels = features.shape
        if nodes != self.num_nodes or channels != self.input_dim:
            raise ValueError("features do not match the configured graph dimensions.")
        if frames > self.max_frames:
            raise ValueError("Sequence is longer than max_frames.")
        if joint_mask.shape != (batch, frames, nodes):
            raise ValueError("joint_mask must have shape [B, T, V].")
        if frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must have shape [B, T].")
        if adjacency.shape not in {(nodes, nodes), (batch, nodes, nodes)}:
            raise ValueError("adjacency shape does not match graph nodes.")
