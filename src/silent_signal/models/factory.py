"""Typed model configuration and construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from silent_signal.models.encoders.pose_encoder import PoseGraphEncoder
from silent_signal.models.recognizer import PoseGraphRecognizer
from silent_signal.pose.cache import canonical_sha256


@dataclass(frozen=True, slots=True)
class PoseGraphModelConfig:
    input_dim: int = 7
    num_nodes: int = 75
    max_frames: int = 64
    hidden_dim: int = 128
    embedding_dim: int = 256
    num_blocks: int = 4
    temporal_kernel: int = 5
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
        )
        if min(dimensions) < 1 or self.num_classes < 2:
            raise ValueError("Model dimensions must be positive and num_classes at least 2.")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer.")
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
class GraphEncoderExperimentConfig:
    model: PoseGraphModelConfig
    smoke: SmokeTrainingConfig
    preprocessing_fingerprint: str
    manifest_sha256: str


def load_graph_encoder_config(path: str | Path) -> GraphEncoderExperimentConfig:
    """Load the pinned graph encoder and smoke-training contract."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Graph encoder config must be a schema_version 1 mapping.")
    model = payload.get("model")
    smoke = payload.get("smoke")
    expected = payload.get("expected")
    if not isinstance(model, dict) or not isinstance(smoke, dict) or not isinstance(expected, dict):
        raise ValueError("Graph encoder config requires model, smoke, and expected mappings.")
    preprocessing_fingerprint = str(expected.get("preprocessing_fingerprint", ""))
    manifest_sha256 = str(expected.get("manifest_sha256", ""))
    if len(preprocessing_fingerprint) != 64 or len(manifest_sha256) != 64:
        raise ValueError("Expected preprocessing and manifest SHA-256 values must be pinned.")
    return GraphEncoderExperimentConfig(
        model=PoseGraphModelConfig(**model),
        smoke=SmokeTrainingConfig(**smoke),
        preprocessing_fingerprint=preprocessing_fingerprint,
        manifest_sha256=manifest_sha256,
    )


def build_pose_graph_recognizer(config: PoseGraphModelConfig) -> PoseGraphRecognizer:
    """Construct the graph encoder and classification head."""

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

    return canonical_sha256({"architecture": "pose_graph_encoder_v1", "config": asdict(config)})


def graph_encoder_config_dict(config: GraphEncoderExperimentConfig) -> dict[str, Any]:
    """Return a serialization-friendly experiment configuration."""

    return {
        "model": asdict(config.model),
        "smoke": asdict(config.smoke),
        "preprocessing_fingerprint": config.preprocessing_fingerprint,
        "manifest_sha256": config.manifest_sha256,
    }
