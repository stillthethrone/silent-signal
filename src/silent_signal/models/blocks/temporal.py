"""Mask-aware temporal blocks for graph pose sequences."""

from __future__ import annotations

from torch import Tensor, nn


class TemporalConvBlock(nn.Module):
    """Residual depthwise temporal convolution applied independently per joint."""

    def __init__(self, hidden_dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, joint_mask: Tensor) -> Tensor:
        batch, frames, joints, channels = features.shape
        residual = features
        normalized = self.norm(features)
        temporal = normalized.permute(0, 2, 3, 1).reshape(
            batch * joints, channels, frames
        )
        temporal = self.pointwise(self.activation(self.depthwise(temporal)))
        temporal = temporal.reshape(batch, joints, channels, frames).permute(0, 3, 1, 2)
        mask = joint_mask.unsqueeze(-1).to(dtype=features.dtype)
        return (residual + self.dropout(temporal)) * mask
