from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from silent_signal.preprocessing.mediapipe_features import (
    MEDIAPIPE_FEATURE_NAMES,
    MEDIAPIPE_JOINT_NAMES,
    MediaPipeGraphPreprocessConfig,
    load_mediapipe_array,
    mediapipe_preprocessing_fingerprint,
    prepare_mediapipe_graph_pose,
)


def _sequence(frames: int = 5, joints: int = 76) -> np.ndarray:
    values = np.zeros((frames, joints, 3), dtype=np.float32)
    for frame in range(frames):
        values[frame, :75, 0] = np.linspace(0.1, 0.9, 75) + frame * 0.01
        values[frame, :75, 1] = np.linspace(0.2, 0.8, 75) + frame * 0.01
        values[frame, :75, 2] = np.linspace(-0.2, 0.2, 75)
    values[2, 33] = 0.0
    if joints == 76:
        values[:, 75] = 99.0
    return values


def test_mediapipe_graph_preserves_model_shape_and_ignores_extra_joint() -> None:
    config = MediaPipeGraphPreprocessConfig(target_frames=8, interpolation_max_gap=1)
    sample = prepare_mediapipe_graph_pose(
        _sequence(),
        sample_id="sample-1",
        class_index=3,
        split="train",
        config=config,
    )

    assert len(MEDIAPIPE_JOINT_NAMES) == 75
    assert len(MEDIAPIPE_FEATURE_NAMES) == 7
    assert sample.features.shape == (8, 75, 7)
    assert sample.joint_mask.shape == (8, 75)
    assert sample.adjacency.shape == (75, 75)
    assert np.isfinite(sample.features).all()
    np.testing.assert_allclose(sample.adjacency, sample.adjacency.T)
    assert sample.metadata["ignored_source_indices"] == [75]
    assert sample.metadata["preprocessing_fingerprint"] == (
        mediapipe_preprocessing_fingerprint(config)
    )


def test_load_mediapipe_array_accepts_75_or_76_and_rejects_other_shapes(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npy"
    np.save(valid, _sequence(joints=75))
    assert load_mediapipe_array(valid).shape == (5, 75, 3)

    invalid = tmp_path / "invalid.npy"
    np.save(invalid, np.zeros((4, 74, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="75, 3"):
        load_mediapipe_array(invalid)
