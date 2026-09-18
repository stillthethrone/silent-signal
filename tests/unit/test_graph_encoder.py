from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from silent_signal.models.factory import (  # noqa: E402
    PoseGraphModelConfig,
    build_pose_graph_recognizer,
    model_fingerprint,
)


def test_graph_encoder_forward_backward_and_mask_invariance() -> None:
    config = PoseGraphModelConfig(
        input_dim=7,
        num_nodes=5,
        max_frames=8,
        hidden_dim=16,
        embedding_dim=24,
        num_blocks=2,
        temporal_kernel=3,
        dropout=0.0,
        num_classes=4,
    )
    model = build_pose_graph_recognizer(config)
    features = torch.randn(2, 8, 5, 7)
    joint_mask = torch.ones(2, 8, 5, dtype=torch.bool)
    joint_mask[:, 2, 3] = False
    frame_mask = joint_mask.any(dim=2)
    adjacency = torch.eye(5)

    model.eval()
    baseline = model(features, joint_mask, frame_mask, adjacency)
    corrupted = features.clone()
    corrupted[~joint_mask] = 10000.0
    candidate = model(corrupted, joint_mask, frame_mask, adjacency)
    torch.testing.assert_close(candidate, baseline)

    model.train()
    logits = model(features, joint_mask, frame_mask, adjacency)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert len(model_fingerprint(config)) == 64


def test_graph_spatial_temporal_transformer_masks_joints_and_empty_frames() -> None:
    config = PoseGraphModelConfig(
        architecture="graph_spatial_temporal_transformer_v1",
        input_dim=7,
        num_nodes=5,
        max_frames=6,
        hidden_dim=16,
        embedding_dim=24,
        num_blocks=1,
        spatial_layers=1,
        temporal_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        num_classes=4,
    )
    model = build_pose_graph_recognizer(config)
    features = torch.randn(2, 6, 5, 7)
    joint_mask = torch.ones(2, 6, 5, dtype=torch.bool)
    joint_mask[:, 2, 3] = False
    joint_mask[:, 4] = False
    frame_mask = joint_mask.any(dim=2)
    adjacency = torch.eye(5)

    model.eval()
    baseline = model(features, joint_mask, frame_mask, adjacency)
    corrupted = features.clone()
    corrupted[~joint_mask] = 10000.0
    candidate = model(corrupted, joint_mask, frame_mask, adjacency)
    torch.testing.assert_close(candidate, baseline)

    model.train()
    logits = model(features, joint_mask, frame_mask, adjacency)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
