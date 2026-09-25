"""Graph Encoder -> Spatial Transformer -> Joint Pooling -> Temporal Transformer (pose branch).

Input ``features [B, T, V, C]`` with ``joint_mask [B, T, V]`` and ``frame_mask [B, T]``.
Masked joints never influence the output: they are zeroed before the first layer, are
excluded as attention keys, and get zero pooling weight. ``encode`` also returns the
temporal tokens so a later RGB branch can attend to them (cross-attention).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from silent_signal.models.blocks.graph import SpatialGraphBlock
from silent_signal.models.heads.classification import ClassificationHead
from silent_signal.pose.cache import canonical_sha256


@dataclass(frozen=True, slots=True)
class PoseTransformerConfig:
    input_dim: int = 9
    num_nodes: int = 68
    max_frames: int = 64
    graph_dim: int = 128
    graph_blocks: int = 2
    spatial_layers: int = 2
    spatial_heads: int = 4
    temporal_dim: int = 256
    temporal_layers: int = 3
    temporal_heads: int = 4
    feedforward_ratio: int = 2
    dropout: float = 0.2
    num_classes: int = 70

    def __post_init__(self) -> None:
        sizes = (
            self.input_dim,
            self.num_nodes,
            self.max_frames,
            self.graph_dim,
            self.graph_blocks,
            self.spatial_layers,
            self.spatial_heads,
            self.temporal_dim,
            self.temporal_layers,
            self.temporal_heads,
            self.feedforward_ratio,
        )
        if min(sizes) < 1 or self.num_classes < 2:
            raise ValueError("Pose transformer sizes must be positive and num_classes >= 2.")
        if self.graph_dim % self.spatial_heads or self.temporal_dim % self.temporal_heads:
            raise ValueError("Model widths must be divisible by their attention head counts.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def fingerprint(self) -> str:
        return canonical_sha256(
            {"architecture": "pose_graph_transformer_v1", "config": asdict(self)}
        )


def _encoder(
    width: int, heads: int, layers: int, ratio: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * ratio,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class PoseGraphTransformer(nn.Module):
    """Pose branch of the RGB-pose design: ``[B, T, V, C]`` -> ``[B, temporal_dim]`` -> logits."""

    def __init__(self, config: PoseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.graph_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, config.num_nodes, config.graph_dim))
        self.graph_blocks = nn.ModuleList(
            SpatialGraphBlock(config.graph_dim, dropout=config.dropout)
            for _ in range(config.graph_blocks)
        )
        self.spatial = _encoder(
            config.graph_dim,
            config.spatial_heads,
            config.spatial_layers,
            config.feedforward_ratio,
            config.dropout,
        )
        self.joint_score = nn.Linear(config.graph_dim, 1)
        self.to_temporal = nn.Sequential(
            nn.LayerNorm(config.graph_dim), nn.Linear(config.graph_dim, config.temporal_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.temporal_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, config.max_frames, config.temporal_dim))
        self.temporal = _encoder(
            config.temporal_dim,
            config.temporal_heads,
            config.temporal_layers,
            config.feedforward_ratio,
            config.dropout,
        )
        self.output_norm = nn.LayerNorm(config.temporal_dim)
        self.classifier = ClassificationHead(config.temporal_dim, config.num_classes)
        for parameter in (self.joint_embedding, self.cls_token, self.time_embedding):
            nn.init.trunc_normal_(parameter, std=0.02)

    def encode(
        self, features: Tensor, joint_mask: Tensor, frame_mask: Tensor, adjacency: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(temporal tokens [B, T, D], valid frames [B, T], embedding [B, D])``."""

        self._validate(features, joint_mask, frame_mask, adjacency)
        batch, frames, joints, _ = features.shape
        weights = joint_mask.unsqueeze(-1).to(features.dtype)
        encoded = (self.input_projection(features * weights) + self.joint_embedding) * weights
        for block in self.graph_blocks:
            encoded = block(encoded, adjacency, joint_mask)

        width = encoded.shape[-1]
        padding = ~joint_mask.reshape(batch * frames, joints)
        # A frame without any joint would give an all-masked attention row (NaN); let it
        # attend over its zero tokens instead and drop it at the temporal stage.
        padding = padding & ~padding.all(dim=1, keepdim=True)
        spatial = self.spatial(
            encoded.reshape(batch * frames, joints, width), src_key_padding_mask=padding
        )
        spatial = spatial.reshape(batch, frames, joints, width) * weights

        valid_frames = frame_mask & joint_mask.any(dim=2)
        scores = self.joint_score(spatial).squeeze(-1).float()
        scores = scores.masked_fill(~joint_mask, float("-inf"))
        scores = torch.where(valid_frames.unsqueeze(-1), scores, torch.zeros_like(scores))
        pool = torch.softmax(scores, dim=-1) * joint_mask
        pooled = (pool.unsqueeze(-1).to(spatial.dtype) * spatial).sum(dim=2)

        sequence = self.to_temporal(pooled) + self.time_embedding[:, :frames]
        sequence = torch.cat(
            [self.cls_token.expand(batch, -1, -1).to(sequence.dtype), sequence], dim=1
        )
        temporal_padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=features.device), ~valid_frames], dim=1
        )
        sequence = self.temporal(sequence, src_key_padding_mask=temporal_padding)
        return sequence[:, 1:], valid_frames, self.output_norm(sequence[:, 0])

    def forward(
        self, features: Tensor, joint_mask: Tensor, frame_mask: Tensor, adjacency: Tensor
    ) -> Tensor:
        return self.classifier(self.encode(features, joint_mask, frame_mask, adjacency)[2])

    def _validate(
        self, features: Tensor, joint_mask: Tensor, frame_mask: Tensor, adjacency: Tensor
    ) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, T, V, C].")
        batch, frames, joints, channels = features.shape
        if joints != self.config.num_nodes or channels != self.config.input_dim:
            raise ValueError(f"Expected [B, T, {self.config.num_nodes}, {self.config.input_dim}].")
        if frames > self.config.max_frames:
            raise ValueError("Sequence is longer than max_frames.")
        if joint_mask.shape != (batch, frames, joints) or frame_mask.shape != (batch, frames):
            raise ValueError("Mask shapes do not match the features.")
        if adjacency.shape not in {(joints, joints), (batch, joints, joints)}:
            raise ValueError("adjacency shape does not match the graph.")
