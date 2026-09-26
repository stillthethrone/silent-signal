from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from silent_signal.models.encoders.pose_transformer import (  # noqa: E402
    PoseTransformerConfig,
)
from silent_signal.models.rgb_pose_fusion import (  # noqa: E402
    RGBPoseFusionClassifier,
    RGBPoseFusionConfig,
)


def _model(*, modality_dropout: float = 0.0) -> RGBPoseFusionClassifier:
    pose = PoseTransformerConfig(
        input_dim=9,
        num_nodes=6,
        max_frames=8,
        graph_dim=16,
        graph_blocks=1,
        spatial_layers=1,
        spatial_heads=4,
        temporal_dim=24,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.0,
        num_classes=5,
    )
    fusion = RGBPoseFusionConfig(
        rgb_source_dim=10,
        max_rgb_tokens=4,
        fusion_dim=16,
        rgb_layers=1,
        rgb_heads=4,
        dropout=0.0,
        modality_dropout=modality_dropout,
        num_classes=5,
    )
    return RGBPoseFusionClassifier(pose, fusion)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(2)
    pose = torch.randn(3, 8, 6, 9)
    joint_mask = torch.ones(3, 8, 6, dtype=torch.bool)
    frame_mask = torch.ones(3, 8, dtype=torch.bool)
    rgb = torch.randn(3, 4, 10)
    rgb_mask = torch.ones(3, 4, dtype=torch.bool)
    rgb_mask[1, -2:] = False
    return pose, joint_mask, frame_mask, torch.eye(6), rgb, rgb_mask


def test_fusion_forward_shapes_and_masked_rgb_is_ignored() -> None:
    model = _model().eval()
    inputs = _inputs()
    logits = model(*inputs)
    pose_embedding, rgb_embedding, fused = model.encode(*inputs)

    assert logits.shape == (3, 5)
    assert pose_embedding.shape == rgb_embedding.shape == fused.shape == (3, 16)
    assert torch.isfinite(logits).all()

    corrupted = list(inputs)
    corrupted[4] = corrupted[4].clone()
    corrupted[4][~inputs[5]] = 1e5
    torch.testing.assert_close(model(*corrupted), logits)


def test_backward_reaches_pose_rgb_gate_and_classifier() -> None:
    model = _model().train()
    loss = torch.nn.functional.cross_entropy(model(*_inputs()), torch.tensor([0, 1, 2]))
    loss.backward()

    for prefix in ("pose", "rgb_input", "rgb_temporal", "gate", "classifier"):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        assert gradients and all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        ), prefix


def test_fusion_config_rejects_invalid_width_and_class_mismatch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RGBPoseFusionConfig(fusion_dim=30, rgb_heads=4)
    model = _model()
    wrong = RGBPoseFusionConfig(
        rgb_source_dim=10,
        max_rgb_tokens=4,
        fusion_dim=16,
        num_classes=6,
    )
    with pytest.raises(ValueError, match="class counts"):
        RGBPoseFusionClassifier(model.pose_config, wrong)


def test_modality_dropout_never_drops_both_branches() -> None:
    model = _model(modality_dropout=0.999).train()

    keep_pose, keep_rgb = model._modality_masks(512, torch.device("cpu"))

    assert torch.all((keep_pose + keep_rgb) >= 1)
    both = (keep_pose == 1) & (keep_rgb == 1)
    assert bool(both.any())
