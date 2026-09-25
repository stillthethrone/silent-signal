from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from silent_signal.models.encoders.pose_transformer import (  # noqa: E402
    PoseGraphTransformer,
    PoseTransformerConfig,
)


def _config() -> PoseTransformerConfig:
    return PoseTransformerConfig(
        input_dim=9,
        num_nodes=6,
        max_frames=8,
        graph_dim=16,
        spatial_heads=4,
        temporal_dim=24,
        temporal_heads=4,
        dropout=0.0,
        num_classes=5,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    features = torch.randn(3, 8, 6, 9)
    joint_mask = torch.rand(3, 8, 6) > 0.3
    joint_mask[0, 1] = False  # a frame without any joint
    joint_mask[1, :, 4:] = False  # a hand never detected
    frame_mask = torch.ones(3, 8, dtype=torch.bool)
    frame_mask[2, 6:] = False
    return features, joint_mask, frame_mask, torch.eye(6)


def test_forward_shapes_and_finite_outputs() -> None:
    model = PoseGraphTransformer(_config()).eval()
    features, joint_mask, frame_mask, adjacency = _inputs()
    tokens, valid, embedding = model.encode(features, joint_mask, frame_mask, adjacency)
    logits = model(features, joint_mask, frame_mask, adjacency)

    assert tokens.shape == (3, 8, 24) and embedding.shape == (3, 24)
    assert logits.shape == (3, 5) and torch.isfinite(logits).all()
    assert not valid[0, 1] and not valid[2, 7]


def test_masked_joints_and_frames_cannot_change_the_output() -> None:
    model = PoseGraphTransformer(_config()).eval()
    features, joint_mask, frame_mask, adjacency = _inputs()
    baseline = model(features, joint_mask, frame_mask, adjacency)

    corrupted = features.clone()
    corrupted[~joint_mask] = 1e4
    corrupted[~frame_mask] = -1e4
    torch.testing.assert_close(model(corrupted, joint_mask, frame_mask, adjacency), baseline)


def test_backward_reaches_every_stage() -> None:
    model = PoseGraphTransformer(_config()).train()
    features, joint_mask, frame_mask, adjacency = _inputs()
    loss = torch.nn.functional.cross_entropy(
        model(features, joint_mask, frame_mask, adjacency), torch.tensor([0, 1, 2])
    )
    loss.backward()

    for name in (
        "input_projection",
        "graph_blocks",
        "spatial",
        "joint_score",
        "temporal",
        "classifier",
    ):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(name)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name


def test_config_rejects_bad_head_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PoseTransformerConfig(graph_dim=30, spatial_heads=4)
    assert len(_config().fingerprint()) == 64
