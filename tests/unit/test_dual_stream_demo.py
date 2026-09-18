from __future__ import annotations

import pytest
import torch
from torch import nn

from silent_signal.cli.train_dual_stream_demo import (
    _validate_selected_manifest,
    build_parser,
)
from silent_signal.models.dual_stream import DualStreamRecognizer
from silent_signal.models.encoders.pose_spatiotemporal_transformer import (
    GraphSpatialTemporalTransformerEncoder,
)
from silent_signal.models.fusion.cross_attention import BidirectionalCrossAttentionFusion


def test_dual_stream_defaults_keep_both_branches_compact() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--selection-report",
            "selected.json",
            "--dataset-root",
            "videos",
            "--graph-root",
            "graphs",
            "--graph-report",
            "graph-report.json",
            "--output-root",
            "output",
        ]
    )

    assert args.classes == 50
    assert (args.rgb_embedding_dim, args.rgb_layers, args.rgb_heads) == (128, 1, 4)
    assert args.pose_embedding_dim == 128
    assert (
        args.pose_graph_layers,
        args.pose_spatial_layers,
        args.pose_temporal_layers,
        args.pose_heads,
    ) == (1, 1, 1, 4)
    assert args.rgb_dropout == 0.4
    assert args.pose_dropout == 0.3
    assert args.fusion_heads == 4
    assert args.fusion_dropout == 0.3
    assert args.run_test is False


def test_selected_manifest_preserves_exact_baseline_mapping() -> None:
    rows = [
        {
            "sample_id": f"{split}-{class_index}",
            "video_path": f"videos/{split}-{class_index}.mp4",
            "class_index": str(class_index),
            "source_class_index": str(class_index + 10),
            "split": split,
        }
        for split in ("train", "validation", "test")
        for class_index in range(2)
    ]
    selection = {
        "classes": [
            {"class_index": 0, "gloss_name": "FIRST"},
            {"class_index": 1, "gloss_name": "SECOND"},
        ]
    }

    assert _validate_selected_manifest(rows, selection, 2) == {0: "FIRST", 1: "SECOND"}


def test_selected_manifest_rejects_a_class_missing_from_validation() -> None:
    rows = [
        {
            "sample_id": f"{split}-{class_index}",
            "video_path": f"videos/{split}-{class_index}.mp4",
            "class_index": str(class_index),
            "source_class_index": str(class_index),
            "split": split,
        }
        for split in ("train", "validation", "test")
        for class_index in range(2)
        if not (split == "validation" and class_index == 1)
    ]
    selection = {
        "classes": [
            {"class_index": 0, "gloss_name": "FIRST"},
            {"class_index": 1, "gloss_name": "SECOND"},
        ]
    }

    with pytest.raises(RuntimeError, match="validation"):
        _validate_selected_manifest(rows, selection, 2)


def test_pose_encoder_uses_spatial_and_temporal_transformers() -> None:
    encoder = GraphSpatialTemporalTransformerEncoder(
        input_dim=7,
        num_nodes=4,
        max_frames=3,
        embedding_dim=8,
        graph_layers=1,
        spatial_layers=1,
        temporal_layers=1,
        heads=2,
        dropout=0.0,
    )
    features = torch.randn(2, 3, 4, 7)
    joint_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    joint_mask[1, 2] = False
    frame_mask = torch.ones(2, 3, dtype=torch.bool)
    adjacency = torch.eye(4).repeat(2, 1, 1)

    tokens, valid = encoder(features, joint_mask, frame_mask, adjacency)

    assert tokens.shape == (2, 3, 8)
    assert valid.tolist() == [[True, True, True], [True, True, False]]
    assert torch.isfinite(tokens).all()
    assert isinstance(encoder.spatial_transformer, nn.TransformerEncoder)
    assert isinstance(encoder.temporal_transformer, nn.TransformerEncoder)


def test_cross_attention_fusion_preserves_embedding_shape() -> None:
    fusion = BidirectionalCrossAttentionFusion(8, heads=2, dropout=0.0)
    fused = fusion(
        torch.randn(3, 4, 8),
        torch.randn(3, 5, 8),
        torch.ones(3, 4, dtype=torch.bool),
        torch.ones(3, 5, dtype=torch.bool),
    )

    assert fused.shape == (3, 8)
    assert torch.isfinite(fused).all()


class _FakeRGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.projection = nn.Linear(4, 8)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.projection(pixel_values)
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool)


class _FakePose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 8)

    def forward(
        self,
        features: torch.Tensor,
        joint_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_mask, frame_mask, adjacency
        tokens = self.projection(features)
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool)


def test_dual_checkpoint_excludes_frozen_videomae_backbone() -> None:
    model = DualStreamRecognizer(
        _FakeRGB(),
        _FakePose(),
        embedding_dim=8,
        num_classes=3,
        fusion_heads=2,
        fusion_dropout=0.0,
    )
    state = model.trainable_state_dict()

    assert state
    assert not any(name.startswith("rgb_encoder.backbone.") for name in state)
    assert any(name.startswith("rgb_encoder.projection.") for name in state)
    model.load_trainable_state_dict(state)
