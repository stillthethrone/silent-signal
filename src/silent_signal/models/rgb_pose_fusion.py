"""Small late-gated fusion model for cached VideoMAE and MediaPipe pose features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from silent_signal.models.encoders.pose_transformer import (
    PoseGraphTransformer,
    PoseTransformerConfig,
)
from silent_signal.models.heads.classification import ClassificationHead
from silent_signal.pose.cache import canonical_sha256


@dataclass(frozen=True, slots=True)
class RGBPoseFusionConfig:
    """Trainable RGB adapter and fusion-head settings.

    The VideoMAE backbone is deliberately absent: notebook 13 caches its temporal
    tokens once.  This model only sees those frozen tokens and the pose sequence.
    """

    rgb_source_dim: int = 768
    max_rgb_tokens: int = 8
    fusion_dim: int = 128
    rgb_layers: int = 1
    rgb_heads: int = 4
    feedforward_ratio: int = 2
    dropout: float = 0.30
    modality_dropout: float = 0.15
    # The trainer always replaces this with the contiguous class count read from
    # the pinned manifest.  Keep the standalone default neutral instead of
    # implying that every release necessarily contains 472 usable classes.
    num_classes: int = 2

    def __post_init__(self) -> None:
        sizes = (
            self.rgb_source_dim,
            self.max_rgb_tokens,
            self.fusion_dim,
            self.rgb_layers,
            self.rgb_heads,
            self.feedforward_ratio,
        )
        if min(sizes) < 1 or self.num_classes < 2:
            raise ValueError("Fusion sizes must be positive and num_classes >= 2.")
        if self.fusion_dim % self.rgb_heads:
            raise ValueError("fusion_dim must be divisible by rgb_heads.")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("dropout and modality_dropout must be in [0, 1).")

    def fingerprint(self, pose: PoseTransformerConfig) -> str:
        return canonical_sha256(
            {
                "architecture": "cached_videomaev2_pose_late_gated_fusion_v1",
                "pose": asdict(pose),
                "fusion": asdict(self),
            }
        )


def _encoder(config: RGBPoseFusionConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=config.fusion_dim,
        nhead=config.rgb_heads,
        dim_feedforward=config.fusion_dim * config.feedforward_ratio,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, config.rgb_layers, enable_nested_tensor=False)


class RGBPoseFusionClassifier(nn.Module):
    """Pose encoder + cached-RGB adapter + gated late fusion.

    Late fusion is intentional.  Pose augmentation may choose a different 64-frame
    temporal window each epoch, whereas cached VideoMAE uses a fixed 16-frame view.
    Pooling both branches before fusion avoids pretending those token grids align.
    """

    def __init__(
        self,
        pose_config: PoseTransformerConfig,
        fusion_config: RGBPoseFusionConfig,
    ) -> None:
        super().__init__()
        if pose_config.num_classes != fusion_config.num_classes:
            raise ValueError("Pose and fusion class counts must match.")
        self.pose_config = pose_config
        self.fusion_config = fusion_config
        self.pose = PoseGraphTransformer(pose_config)
        # ``encode`` is the public pose-branch contract.  Remove its standalone head
        # so the fusion model has no unused trainable classifier parameters.
        self.pose.classifier = nn.Identity()

        self.rgb_input = nn.Sequential(
            nn.LayerNorm(fusion_config.rgb_source_dim),
            nn.Linear(fusion_config.rgb_source_dim, fusion_config.fusion_dim),
        )
        self.rgb_cls = nn.Parameter(torch.zeros(1, 1, fusion_config.fusion_dim))
        self.rgb_position = nn.Parameter(
            torch.zeros(1, fusion_config.max_rgb_tokens, fusion_config.fusion_dim)
        )
        self.rgb_temporal = _encoder(fusion_config)
        self.rgb_output_norm = nn.LayerNorm(fusion_config.fusion_dim)
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(pose_config.temporal_dim),
            nn.Linear(pose_config.temporal_dim, fusion_config.fusion_dim),
            nn.LayerNorm(fusion_config.fusion_dim),
        )
        self.gate = nn.Linear(fusion_config.fusion_dim * 2, fusion_config.fusion_dim)
        self.fusion_block = nn.Sequential(
            nn.LayerNorm(fusion_config.fusion_dim),
            nn.Linear(fusion_config.fusion_dim, fusion_config.fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_config.dropout),
        )
        self.classifier = ClassificationHead(
            fusion_config.fusion_dim, fusion_config.num_classes
        )
        nn.init.trunc_normal_(self.rgb_cls, std=0.02)
        nn.init.trunc_normal_(self.rgb_position, std=0.02)

    def encode_rgb(self, rgb_features: Tensor, rgb_mask: Tensor) -> Tensor:
        if rgb_features.ndim != 3:
            raise ValueError("rgb_features must have shape [B, L, D].")
        batch, tokens, channels = rgb_features.shape
        config = self.fusion_config
        if channels != config.rgb_source_dim or tokens > config.max_rgb_tokens:
            raise ValueError(
                f"Expected RGB [B, L <= {config.max_rgb_tokens}, "
                f"{config.rgb_source_dim}]."
            )
        if rgb_mask.shape != (batch, tokens):
            raise ValueError("rgb_mask shape does not match rgb_features.")
        if not bool(rgb_mask.any(dim=1).all()):
            raise ValueError("Every sample needs at least one valid RGB token.")

        sequence = self.rgb_input(rgb_features)
        sequence = sequence + self.rgb_position[:, :tokens].to(sequence.dtype)
        sequence = torch.cat(
            [self.rgb_cls.expand(batch, -1, -1).to(sequence.dtype), sequence], dim=1
        )
        padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=rgb_mask.device), ~rgb_mask],
            dim=1,
        )
        sequence = self.rgb_temporal(sequence, src_key_padding_mask=padding)
        return self.rgb_output_norm(sequence[:, 0])

    def encode(
        self,
        pose_features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
        rgb_features: Tensor,
        rgb_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(pose embedding, RGB embedding, fused embedding)``."""

        pose_embedding = self.pose.encode(
            pose_features, joint_mask, frame_mask, adjacency
        )[2]
        pose_embedding = self.pose_projection(pose_embedding)
        rgb_embedding = self.encode_rgb(rgb_features, rgb_mask)

        keep_pose, keep_rgb = self._modality_masks(len(pose_embedding), pose_embedding.device)
        gate = torch.sigmoid(self.gate(torch.cat([pose_embedding, rgb_embedding], dim=-1)))
        pose_weight = gate * keep_pose
        rgb_weight = (1.0 - gate) * keep_rgb
        fused = (
            pose_embedding * pose_weight + rgb_embedding * rgb_weight
        ) / (pose_weight + rgb_weight).clamp_min(1e-6)
        fused = fused + self.fusion_block(fused)
        return pose_embedding, rgb_embedding, fused

    def forward(
        self,
        pose_features: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        adjacency: Tensor,
        rgb_features: Tensor,
        rgb_mask: Tensor,
    ) -> Tensor:
        fused = self.encode(
            pose_features,
            joint_mask,
            frame_mask,
            adjacency,
            rgb_features,
            rgb_mask,
        )[2]
        return self.classifier(fused)

    def _modality_masks(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        probability = self.fusion_config.modality_dropout
        if not self.training or probability == 0:
            ones = torch.ones(batch, 1, device=device)
            return ones, ones
        keep_pose = torch.rand(batch, 1, device=device) >= probability
        keep_rgb = torch.rand(batch, 1, device=device) >= probability
        # Never discard both modalities for one sample.  Restoring both avoids
        # introducing a systematic RGB preference in the rare double-drop case.
        both_dropped = ~(keep_pose | keep_rgb)
        keep_pose = keep_pose | both_dropped
        keep_rgb = keep_rgb | both_dropped
        return keep_pose.float(), keep_rgb.float()
