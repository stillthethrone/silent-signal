"""Cross-attention fusion blocks for RGB and pose token sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _safe_valid_mask(valid: Tensor) -> Tensor:
    safe = valid.to(dtype=torch.bool).clone()
    empty = ~safe.any(dim=1)
    if empty.any():
        safe[empty, 0] = True
    return safe


def _masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
    weights = valid.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class BidirectionalCrossAttentionFusion(nn.Module):
    """Let RGB attend to pose and pose attend to RGB before feature fusion."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or heads < 1 or embedding_dim % heads:
            raise ValueError("embedding_dim must be positive and divisible by heads.")
        self.rgb_input_norm = nn.LayerNorm(embedding_dim)
        self.pose_input_norm = nn.LayerNorm(embedding_dim)
        self.rgb_to_pose = nn.MultiheadAttention(
            embedding_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pose_to_rgb = nn.MultiheadAttention(
            embedding_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.rgb_output_norm = nn.LayerNorm(embedding_dim)
        self.pose_output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        rgb_tokens: Tensor,
        pose_tokens: Tensor,
        rgb_valid: Tensor,
        pose_valid: Tensor,
    ) -> Tensor:
        if rgb_tokens.ndim != 3 or pose_tokens.ndim != 3:
            raise ValueError("RGB and pose tokens must have shape [B, T, C].")
        if rgb_tokens.shape[0] != pose_tokens.shape[0]:
            raise ValueError("RGB and pose batches must have the same size.")
        if rgb_tokens.shape[2] != pose_tokens.shape[2]:
            raise ValueError("RGB and pose embeddings must have the same dimension.")
        if rgb_valid.shape != rgb_tokens.shape[:2]:
            raise ValueError("rgb_valid must match the RGB token sequence.")
        if pose_valid.shape != pose_tokens.shape[:2]:
            raise ValueError("pose_valid must match the pose token sequence.")

        safe_rgb_valid = _safe_valid_mask(rgb_valid)
        safe_pose_valid = _safe_valid_mask(pose_valid)
        rgb_normalized = self.rgb_input_norm(rgb_tokens)
        pose_normalized = self.pose_input_norm(pose_tokens)
        rgb_context, _ = self.rgb_to_pose(
            query=rgb_normalized,
            key=pose_normalized,
            value=pose_normalized,
            key_padding_mask=~safe_pose_valid,
            need_weights=False,
        )
        pose_context, _ = self.pose_to_rgb(
            query=pose_normalized,
            key=rgb_normalized,
            value=rgb_normalized,
            key_padding_mask=~safe_rgb_valid,
            need_weights=False,
        )
        rgb_cross = self.rgb_output_norm(rgb_tokens + self.dropout(rgb_context))
        pose_cross = self.pose_output_norm(pose_tokens + self.dropout(pose_context))
        rgb_feature = _masked_mean(rgb_cross, rgb_valid)
        pose_feature = _masked_mean(pose_cross, pose_valid)
        return self.fusion(torch.cat((rgb_feature, pose_feature), dim=-1))
