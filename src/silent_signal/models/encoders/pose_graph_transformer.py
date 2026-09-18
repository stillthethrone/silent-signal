"""Graph, spatial-attention, and temporal-attention pose encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from silent_signal.models.blocks.graph import SpatialGraphBlock


class PoseGraphSpatialTemporalTransformer(nn.Module):
    """Encode graph pose tensors with explicit spatial and temporal attention.

    The graph blocks first inject the fixed anatomical topology. Spatial
    Transformer layers then attend across joints independently for each frame.
    Masked joint pooling produces frame tokens consumed by a temporal
    Transformer and a learned clip token.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_nodes: int,
        max_frames: int,
        hidden_dim: int,
        embedding_dim: int,
        num_graph_blocks: int,
        spatial_layers: int,
        temporal_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = (
            input_dim,
            num_nodes,
            max_frames,
            hidden_dim,
            embedding_dim,
            num_graph_blocks,
            spatial_layers,
            temporal_layers,
            num_heads,
            ffn_dim,
        )
        if min(dimensions) < 1:
            raise ValueError("All graph Transformer dimensions must be positive.")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.max_frames = max_frames
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, num_nodes, hidden_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, max_frames, 1, hidden_dim))
        self.clip_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.graph_blocks = nn.ModuleList(
            SpatialGraphBlock(hidden_dim, dropout=dropout)
            for _ in range(num_graph_blocks)
        )

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
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
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
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
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.clip_token, std=0.02)

    def forward(
        self,
        features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        self._validate_inputs(features, joint_mask, frame_mask, adjacency)
        batch, frames, joints, _ = features.shape
        numeric_joint_mask = joint_mask.unsqueeze(-1).to(dtype=features.dtype)

        encoded = self.input_projection(features)
        encoded = encoded + self.joint_embedding + self.time_embedding[:, :frames]
        encoded = self.input_dropout(encoded) * numeric_joint_mask
        for block in self.graph_blocks:
            encoded = block(encoded, adjacency, joint_mask)

        spatial_tokens = encoded.reshape(batch * frames, joints, -1)
        spatial_valid = joint_mask.reshape(batch * frames, joints)
        safe_spatial_valid = spatial_valid.clone()
        empty_frames = ~safe_spatial_valid.any(dim=1)
        safe_spatial_valid[empty_frames, 0] = True
        spatial_tokens = self.spatial_transformer(
            spatial_tokens,
            src_key_padding_mask=~safe_spatial_valid,
        )
        spatial_tokens = spatial_tokens * spatial_valid.unsqueeze(-1).to(
            dtype=spatial_tokens.dtype
        )
        spatial_tokens = spatial_tokens.reshape(batch, frames, joints, -1)
        joint_weights = joint_mask.unsqueeze(-1).to(dtype=spatial_tokens.dtype)
        frame_tokens = (spatial_tokens * joint_weights).sum(dim=2) / joint_weights.sum(
            dim=2
        ).clamp_min(1.0)

        effective_frames = frame_mask & joint_mask.any(dim=2)
        frame_tokens = frame_tokens * effective_frames.unsqueeze(-1).to(
            dtype=frame_tokens.dtype
        )
        clip_token = self.clip_token.expand(batch, -1, -1)
        temporal_tokens = torch.cat((clip_token, frame_tokens), dim=1)
        temporal_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=features.device),
                ~effective_frames,
            ),
            dim=1,
        )
        temporal_tokens = self.temporal_transformer(
            temporal_tokens,
            src_key_padding_mask=temporal_padding,
        )
        return self.output(temporal_tokens[:, 0])

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
