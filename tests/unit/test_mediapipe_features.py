from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from silent_signal.data.keypoint_pack import (
    read_packed_keypoints,
    select_sequences,
    write_packed_keypoints,
)
from silent_signal.pose.layouts import MEDIAPIPE_UPPER68_V1
from silent_signal.preprocessing.mediapipe_features import (
    AugmentationConfig,
    MediaPipeFeatureConfig,
    mediapipe_graph_features,
)

_LEFT_HAND = [j.source_index for j in MEDIAPIPE_UPPER68_V1.joints if j.body_part == "left_hand"]


def _raw(frames: int = 40, seed: int = 0) -> np.ndarray:
    raw = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(frames, 76, 3)).astype(np.float32)
    raw[:, 25:33] = 0.0  # legs outside the crop are not used by the layout anyway
    raw[10:20, _LEFT_HAND] = 0.0  # left hand lost for 10 frames
    raw[5, 15] = 0.0  # a one-frame gap in the left wrist (body joint 15)
    return raw


def test_features_shape_masks_and_channels() -> None:
    config = MediaPipeFeatureConfig(target_frames=64)
    features, joint_mask, frame_mask = mediapipe_graph_features(_raw(), config)

    assert features.shape == (64, 68, 9) and features.dtype == np.float32
    assert joint_mask.shape == (64, 68) and frame_mask.all()
    left_hand = [i for i, j in enumerate(MEDIAPIPE_UPPER68_V1.joints) if j.body_part == "left_hand"]
    lost = ~joint_mask[:, left_hand].any(axis=1)
    assert 10 < lost.sum() < 25  # the 10-frame gap is longer than the 3-frame fill limit
    assert np.all(features[~joint_mask] == 0.0)
    wrist = MEDIAPIPE_UPPER68_V1.index_by_name["left_wrist"]
    assert joint_mask[:, wrist].all()  # the single-frame gap was interpolated


def test_bones_stay_inside_each_hand_and_velocity_is_masked() -> None:
    config = MediaPipeFeatureConfig(target_frames=40)
    raw = _raw()
    features, joint_mask, _ = mediapipe_graph_features(raw, config)
    names = MEDIAPIPE_UPPER68_V1.index_by_name
    hand_wrist = names["left_hand_wrist"]
    assert np.all(features[:, hand_wrist, 6:9] == 0.0)  # no bone to the body wrist
    thumb = names["left_hand_thumb_1"]
    valid = joint_mask[:, thumb] & joint_mask[:, hand_wrist]
    expected = features[valid, thumb, 0:3] - features[valid, hand_wrist, 0:3]
    np.testing.assert_allclose(features[valid, thumb, 6:9], expected, atol=1e-6)
    assert np.all(features[0, :, 3:6] == 0.0)


def test_augmentation_is_seeded_and_keeps_missing_joints_zero() -> None:
    config = MediaPipeFeatureConfig(target_frames=32)
    augmentation = AugmentationConfig()
    first = mediapipe_graph_features(
        _raw(), config, rng=np.random.default_rng(1), augmentation=augmentation
    )
    again = mediapipe_graph_features(
        _raw(), config, rng=np.random.default_rng(1), augmentation=augmentation
    )
    other = mediapipe_graph_features(
        _raw(), config, rng=np.random.default_rng(2), augmentation=augmentation
    )

    np.testing.assert_array_equal(first[0], again[0])
    assert not np.array_equal(first[0], other[0])
    assert np.all(first[0][~first[1]] == 0.0)


def test_rejects_wrong_keypoint_shapes() -> None:
    with pytest.raises(ValueError, match=r"\[T, 76, 3\]"):
        mediapipe_graph_features(np.zeros((10, 75, 3)), MediaPipeFeatureConfig())


def test_rejects_all_zero_keypoints() -> None:
    with pytest.raises(ValueError, match="no observed joints"):
        mediapipe_graph_features(np.zeros((10, 76, 3)), MediaPipeFeatureConfig())


def test_rejects_keypoints_only_in_excluded_leg_joints() -> None:
    raw = np.zeros((10, 76, 3), dtype=np.float32)
    raw[:, 25:33] = 1.0
    with pytest.raises(ValueError, match="no observed joints in layout"):
        mediapipe_graph_features(raw, MediaPipeFeatureConfig())


def test_rejects_incompatible_pose_layout() -> None:
    with pytest.raises(ValueError, match="MediaPipe Holistic 76"):
        MediaPipeFeatureConfig(layout_name="coco_wholebody_75_v1")


def test_packed_keypoints_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    sequences = {
        "b": rng.normal(size=(7, 76, 3)).astype(np.float32),
        "a": rng.normal(size=(3, 76, 3)).astype(np.float32),
    }
    path = tmp_path / "pack.npz"
    write_packed_keypoints(path, sequences, {"format": "test"})
    packed = read_packed_keypoints(path)

    assert packed.sample_ids == ("a", "b") and packed.metadata == {"format": "test"}
    np.testing.assert_array_equal(select_sequences(packed, ["b"])[0], sequences["b"])
    with pytest.raises(KeyError):
        select_sequences(packed, ["missing"])
