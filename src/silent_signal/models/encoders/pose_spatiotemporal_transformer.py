"""Graph, spatial-attention, and temporal-attention encoder for pose caches."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from silent_signal.models.blocks.graph import SpatialGraphBlock


def _safe_valid_mask(valid: Tensor) -> Tensor:
    """Ensure every Transformer sequence has at least one unmasked token."""

    safe = valid.to(dtype=torch.bool).clone()
    empty = ~safe.any(dim=1)
    if empty.any():
        safe[empty, 0] = True
    return safe


class GraphSpatialTemporalTransformerEncoder(nn.Module):
    """Encode pose as Graph Encoder -> Spatial Transformer -> Temporal Transformer."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_nodes: int,
        max_frames: int,
        embedding_dim: int,
        graph_layers: int = 1,
        spatial_layers: int = 1,
        temporal_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dimensions = (
            input_dim,
            num_nodes,
            max_frames,
            embedding_dim,
            graph_layers,
            spatial_layers,
            temporal_layers,
            heads,
        )
        if min(dimensions) < 1:
            raise ValueError("All pose encoder dimensions and layer counts must be positive.")
        if embedding_dim % heads:
            raise ValueError("pose embedding_dim must be divisible by heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.max_frames = max_frames
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, num_nodes, embedding_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, max_frames, embedding_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.graph_encoder = nn.ModuleList(
            SpatialGraphBlock(embedding_dim, dropout=dropout) for _ in range(graph_layers)
        )

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer,
            num_layers=spatial_layers,
            enable_nested_tensor=False,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(features, joint_mask, frame_mask, adjacency)
        batch, frames, joints, _ = features.shape
        joint_valid = joint_mask.to(dtype=torch.bool)
        joint_weights = joint_valid.unsqueeze(-1).to(dtype=features.dtype)

        encoded = self.input_projection(features) + self.joint_embedding
        encoded = self.input_dropout(encoded) * joint_weights
        for graph_block in self.graph_encoder:
            encoded = graph_block(encoded, adjacency, joint_valid)

        spatial = encoded.reshape(batch * frames, joints, -1)
        flat_joint_valid = joint_valid.reshape(batch * frames, joints)
        safe_joint_valid = _safe_valid_mask(flat_joint_valid)
        spatial = self.spatial_transformer(
            spatial,
            src_key_padding_mask=~safe_joint_valid,
        )
        spatial = self.spatial_norm(spatial).reshape(batch, frames, joints, -1)
        spatial = spatial * joint_weights
        per_frame = spatial.sum(dim=2) / joint_weights.sum(dim=2).clamp_min(1.0)

        effective_frames = frame_mask.to(dtype=torch.bool) & joint_valid.any(dim=2)
        temporal = per_frame + self.time_embedding[:, :frames]
        temporal = temporal * effective_frames.unsqueeze(-1).to(dtype=temporal.dtype)
        safe_frame_valid = _safe_valid_mask(effective_frames)
        temporal = self.temporal_transformer(
            temporal,
            src_key_padding_mask=~safe_frame_valid,
        )
        temporal = self.temporal_norm(temporal)
        temporal = temporal * effective_frames.unsqueeze(-1).to(dtype=temporal.dtype)
        return temporal, effective_frames

    def _validate_inputs(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, T, J, C].")
        batch, frames, nodes, channels = features.shape
        if nodes != self.num_nodes or channels != self.input_dim:
            raise ValueError("features do not match the configured pose dimensions.")
        if frames > self.max_frames:
            raise ValueError("Sequence is longer than max_frames.")
        if joint_mask.shape != (batch, frames, nodes):
            raise ValueError("joint_mask must have shape [B, T, J].")
        if frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must have shape [B, T].")
        if adjacency.shape not in {(nodes, nodes), (batch, nodes, nodes)}:
            raise ValueError("adjacency shape does not match the configured pose graph.")
