"""Dual-stream VideoMAE and graph-spatial-temporal Transformer recognizer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from silent_signal.models.encoders.pose_spatiotemporal_transformer import (
    GraphSpatialTemporalTransformerEncoder,
)
from silent_signal.models.fusion.cross_attention import BidirectionalCrossAttentionFusion


class FrozenVideoMAETemporalEncoder(nn.Module):
    """Frozen VideoMAE V2 feature extractor plus a trainable RGB Transformer."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embedding_dim: int = 128,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or layers < 1 or heads < 1:
            raise ValueError("RGB dimensions, layers, and heads must be positive.")
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads.")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        visual = self.backbone.model
        source_dim = int(visual.embed_dim)
        spatial_patches = int(visual.patch_embed.img_size[0] // visual.patch_embed.patch_size[0])
        spatial_patches *= int(
            visual.patch_embed.img_size[1] // visual.patch_embed.patch_size[1]
        )
        self.temporal_tokens = int(visual.patch_embed.num_patches) // spatial_patches
        self.projection = nn.Linear(source_dim, embedding_dim)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.temporal_tokens, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rgb_transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

    def train(self, mode: bool = True) -> FrozenVideoMAETemporalEncoder:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self._frozen_temporal_tokens(pixel_values)
        tokens = self.projection(tokens)
        if tokens.shape[1] != self.temporal_tokens:
            raise RuntimeError(
                f"Expected {self.temporal_tokens} temporal tokens, got {tokens.shape[1]}."
            )
        tokens = self.output_norm(
            self.rgb_transformer(tokens + self.temporal_position[:, : tokens.shape[1]])
        )
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, valid

    def _frozen_temporal_tokens(self, pixel_values: Tensor) -> Tensor:
        visual = self.backbone.model
        with torch.no_grad():
            tokens = visual.patch_embed(pixel_values)
            if visual.pos_embed is not None:
                position = visual.pos_embed.expand(tokens.size(0), -1, -1)
                tokens = tokens + position.type_as(tokens).to(tokens.device).detach()
            tokens = visual.pos_drop(tokens)
            for block in visual.blocks:
                tokens = block(tokens)
            temporal = pixel_values.shape[2] // int(visual.tubelet_size)
            if tokens.shape[1] % temporal:
                raise RuntimeError("VideoMAE token count is not divisible by temporal tubes.")
            spatial = tokens.shape[1] // temporal
            tokens = tokens.reshape(tokens.shape[0], temporal, spatial, tokens.shape[2])
            tokens = tokens.mean(dim=2)
            tokens = visual.fc_norm(tokens) if visual.fc_norm is not None else visual.norm(tokens)
        return tokens


class DualStreamRecognizer(nn.Module):
    """Recognize signs with the document-specified dual-stream architecture."""

    def __init__(
        self,
        rgb_encoder: FrozenVideoMAETemporalEncoder,
        pose_encoder: GraphSpatialTemporalTransformerEncoder,
        *,
        embedding_dim: int,
        num_classes: int,
        fusion_heads: int = 4,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.pose_encoder = pose_encoder
        self.fusion = BidirectionalCrossAttentionFusion(
            embedding_dim,
            heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        *,
        pixel_values: Tensor,
        pose_features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        rgb_tokens, rgb_valid = self.rgb_encoder(pixel_values)
        pose_tokens, pose_valid = self.pose_encoder(
            pose_features,
            joint_mask,
            frame_mask,
            adjacency,
        )
        fused = self.fusion(rgb_tokens, pose_tokens, rgb_valid, pose_valid)
        return self.classifier(fused)

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("rgb_encoder.backbone.")
        }

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        current = self.state_dict()
        expected = {name for name in current if not name.startswith("rgb_encoder.backbone.")}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise RuntimeError(
                f"Invalid dual-stream checkpoint: missing={missing[:5]}, extra={extra[:5]}."
            )
        current.update(state)
        self.load_state_dict(current)
