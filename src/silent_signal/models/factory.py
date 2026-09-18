"""Typed model configuration and construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from silent_signal.models.encoders.pose_encoder import PoseGraphEncoder
from silent_signal.models.encoders.pose_graph_transformer import (
    PoseGraphSpatialTemporalTransformer,
)
from silent_signal.models.recognizer import PoseGraphRecognizer
from silent_signal.pose.cache import canonical_sha256


@dataclass(frozen=True, slots=True)
class PoseGraphModelConfig:
    architecture: str = "graph_conv_temporal_conv_v1"
    input_dim: int = 7
    num_nodes: int = 75
    max_frames: int = 64
    hidden_dim: int = 128
    embedding_dim: int = 256
    num_blocks: int = 4
    temporal_kernel: int = 5
    spatial_layers: int = 2
    temporal_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    num_classes: int = 200

    def __post_init__(self) -> None:
        dimensions = (
            self.input_dim,
            self.num_nodes,
            self.max_frames,
            self.hidden_dim,
            self.embedding_dim,
            self.num_blocks,
            self.num_classes,
            self.spatial_layers,
            self.temporal_layers,
            self.num_heads,
            self.ffn_dim,
        )
        if min(dimensions) < 1 or self.num_classes < 2:
            raise ValueError("Model dimensions must be positive and num_classes at least 2.")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer.")
        supported = {
            "graph_conv_temporal_conv_v1",
            "graph_spatial_temporal_transformer_v1",
        }
        if self.architecture not in supported:
            raise ValueError(f"Unsupported pose graph architecture: {self.architecture}.")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class SmokeTrainingConfig:
    batch_size: int = 8
    train_steps: int = 5
    sample_limit: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.batch_size, self.train_steps, self.sample_limit) < 1:
            raise ValueError("Smoke batch_size, train_steps, and sample_limit must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")


@dataclass(frozen=True, slots=True)
class FullTrainingConfig:
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0
    early_stopping_min_epochs: int = 10
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.max_epochs,
            self.gradient_clip,
            self.early_stopping_min_epochs,
            self.early_stopping_patience,
        )
        if min(positive) < 1:
            raise ValueError(
                "Full-training batch, epoch, clipping, and patience values must be positive."
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid full-training optimizer settings.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.early_stopping_min_delta < 0 or self.num_workers < 0:
            raise ValueError("Early-stopping delta and num_workers must not be negative.")


@dataclass(frozen=True, slots=True)
class GraphEncoderExperimentConfig:
    model: PoseGraphModelConfig
    smoke: SmokeTrainingConfig
    preprocessing_fingerprint: str
    manifest_sha256: str | None
    training: FullTrainingConfig | None = None


def load_graph_encoder_config(path: str | Path) -> GraphEncoderExperimentConfig:
    """Load the pinned graph encoder and smoke-training contract."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Graph encoder config must be a schema_version 1 mapping.")
    model = payload.get("model")
    smoke = payload.get("smoke")
    training = payload.get("training")
    expected = payload.get("expected")
    if not isinstance(model, dict) or not isinstance(smoke, dict) or not isinstance(expected, dict):
        raise ValueError("Graph encoder config requires model, smoke, and expected mappings.")
    preprocessing_fingerprint = str(expected.get("preprocessing_fingerprint", ""))
    raw_manifest_sha256 = expected.get("manifest_sha256")
    manifest_sha256 = (
        None
        if raw_manifest_sha256 in (None, "", "auto")
        else str(raw_manifest_sha256)
    )
    if len(preprocessing_fingerprint) != 64:
        raise ValueError("Expected preprocessing SHA-256 value must be pinned.")
    if manifest_sha256 is not None and len(manifest_sha256) != 64:
        raise ValueError("Expected manifest SHA-256 must be 64 characters or 'auto'.")
    if training is not None and not isinstance(training, dict):
        raise ValueError("training must be a mapping when provided.")
    return GraphEncoderExperimentConfig(
        model=PoseGraphModelConfig(**model),
        smoke=SmokeTrainingConfig(**smoke),
        preprocessing_fingerprint=preprocessing_fingerprint,
        manifest_sha256=manifest_sha256,
        training=FullTrainingConfig(**training) if training is not None else None,
    )


def build_pose_graph_recognizer(config: PoseGraphModelConfig) -> PoseGraphRecognizer:
    """Construct the graph encoder and classification head."""

    if config.architecture == "graph_spatial_temporal_transformer_v1":
        encoder: nn.Module = PoseGraphSpatialTemporalTransformer(
            input_dim=config.input_dim,
            num_nodes=config.num_nodes,
            max_frames=config.max_frames,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_graph_blocks=config.num_blocks,
            spatial_layers=config.spatial_layers,
            temporal_layers=config.temporal_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
    else:
        encoder = PoseGraphEncoder(
            input_dim=config.input_dim,
            num_nodes=config.num_nodes,
            max_frames=config.max_frames,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            temporal_kernel=config.temporal_kernel,
            dropout=config.dropout,
        )
    return PoseGraphRecognizer(
        encoder,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
    )


def model_fingerprint(config: PoseGraphModelConfig) -> str:
    """Return a stable identity for architecture-changing settings."""

    if config.architecture == "graph_conv_temporal_conv_v1":
        legacy_config = {
            key: value
            for key, value in asdict(config).items()
            if key
            in {
                "input_dim",
                "num_nodes",
                "max_frames",
                "hidden_dim",
                "embedding_dim",
                "num_blocks",
                "temporal_kernel",
                "dropout",
                "num_classes",
            }
        }
        return canonical_sha256(
            {"architecture": "pose_graph_encoder_v1", "config": legacy_config}
        )
    return canonical_sha256({"architecture": config.architecture, "config": asdict(config)})


def graph_encoder_config_dict(config: GraphEncoderExperimentConfig) -> dict[str, Any]:
    """Return a serialization-friendly experiment configuration."""

    return {
        "model": asdict(config.model),
        "smoke": asdict(config.smoke),
        "training": asdict(config.training) if config.training is not None else None,
        "preprocessing_fingerprint": config.preprocessing_fingerprint,
        "manifest_sha256": config.manifest_sha256,
    }
