from __future__ import annotations

from pathlib import Path

import numpy as np

from silent_signal.pose.cache import pose_cache_path
from silent_signal.pose.interface import RawPoseSequence
from silent_signal.preprocessing.cache import (
    graph_cache_is_current,
    read_graph_pose_cache,
    write_graph_pose_cache,
)
from silent_signal.preprocessing.pose_features import (
    GraphPreprocessConfig,
    prepare_graph_pose,
    preprocessing_fingerprint,
)


def _raw_sequence(sample_id: str = "sample-1") -> RawPoseSequence:
    frames = 5
    xy = np.zeros((frames, 133, 2), dtype=np.float32)
    scores = np.full((frames, 133), 0.9, dtype=np.float32)
    for frame in range(frames):
        xy[frame, :, 0] = np.arange(133, dtype=np.float32) + frame
        xy[frame, :, 1] = np.arange(133, dtype=np.float32) * 0.5 + frame
    scores[2, 91] = 0.0
    return RawPoseSequence(
        sample_id=sample_id,
        source_video="videos/sample.mp4",
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps_seconds=np.arange(frames, dtype=np.float64) / 25.0,
        frame_size_hw=(480, 640),
        keypoints_xy=xy,
        keypoint_scores=scores,
        bboxes_xyxy=np.tile(np.asarray([0, 0, 639, 479], dtype=np.float32), (frames, 1)),
        bbox_scores=np.ones((frames,), dtype=np.float32),
        person_detected=np.ones((frames,), dtype=np.bool_),
        metadata={
            "extractor_fingerprint": "extractor-v1",
            "video_sha256": "video-v1",
        },
    )


def test_prepare_graph_pose_has_stable_shape_masks_and_adjacency() -> None:
    config = GraphPreprocessConfig(target_frames=8, interpolation_max_gap=1)
    sample = prepare_graph_pose(
        _raw_sequence(),
        class_index=4,
        split="train",
        config=config,
    )

    assert sample.features.shape == (8, 75, 7)
    assert sample.joint_mask.shape == (8, 75)
    assert sample.observed_mask.shape == (8, 75)
    assert sample.adjacency.shape == (75, 75)
    assert np.isfinite(sample.features).all()
    np.testing.assert_allclose(sample.adjacency, sample.adjacency.T)
    assert np.all(np.diag(sample.adjacency) > 0)
    assert sample.metadata["preprocessing_fingerprint"] == preprocessing_fingerprint(config)


def test_graph_cache_round_trip_and_resume_identity(tmp_path: Path) -> None:
    config = GraphPreprocessConfig(target_frames=8)
    sample = prepare_graph_pose(
        _raw_sequence("unsafe/id"),
        class_index=7,
        split="validation",
        config=config,
    )
    destination = pose_cache_path(tmp_path, sample.sample_id)
    write_graph_pose_cache(destination, sample)
    restored = read_graph_pose_cache(
        destination,
        expected_sample_id=sample.sample_id,
        expected_fingerprint=preprocessing_fingerprint(config),
    )

    np.testing.assert_array_equal(restored.features, sample.features)
    assert restored.class_index == 7
    assert restored.split == "validation"
    assert graph_cache_is_current(
        destination,
        sample_id=sample.sample_id,
        preprocessing_fingerprint=preprocessing_fingerprint(config),
    )
    assert not graph_cache_is_current(
        destination,
        sample_id=sample.sample_id,
        preprocessing_fingerprint="stale",
    )
