from __future__ import annotations

import pytest
import torch
from torch import nn

from silent_signal.cli.analyze_videomaev2_demo import build_parser as build_analysis_parser
from silent_signal.cli.train_dual_stream_demo import (
    _is_overfit_epoch,
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
    assert (
        args.overfit_monitor_patience,
        args.overfit_min_epoch,
        args.overfit_loss_gap,
        args.overfit_top1_gap,
        args.overfit_validation_loss_regression,
    ) == (3, 6, 0.5, 0.2, 0.1)
    assert args.run_test is False


def test_analysis_cli_accepts_dual_stream_artifact_paths() -> None:
    args = build_analysis_parser().parse_args(
        [
            "--run-root",
            "dual-run",
            "--selection-report",
            "selected.json",
            "--training-report",
            "dual_stream_report.json",
            "--model-label",
            "dual-stream",
        ]
    )

    assert str(args.run_root) == "dual-run"
    assert str(args.selection_report) == "selected.json"
    assert str(args.training_report) == "dual_stream_report.json"
    assert args.model_label == "dual-stream"


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


def test_overfit_monitor_requires_validation_regression_from_best() -> None:
    record = {
        "epoch": 8,
        "train_loss": 1.0,
        "validation_loss": 2.1,
        "train_top1": 0.8,
        "validation_top1": 0.5,
    }
    thresholds = {
        "min_epoch": 6,
        "loss_gap": 0.5,
        "top1_gap": 0.2,
        "validation_loss_regression": 0.1,
    }

    assert not _is_overfit_epoch(record, best_validation_loss=2.05, **thresholds)
    assert _is_overfit_epoch(record, best_validation_loss=1.8, **thresholds)


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
