from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from silent_signal.pose.cache import (
    PoseCacheError,
    cache_is_current,
    pose_cache_path,
    read_pose_cache,
    sha256_file,
    write_pose_cache,
)
from silent_signal.pose.interface import RawPoseSequence


def _sequence() -> RawPoseSequence:
    frame_count = 3
    return RawPoseSequence(
        sample_id="clip/unsafe-looking-id",
        source_video="C:/datasets/clip.mp4",
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_seconds=np.asarray([0.0, 0.04, 0.08], dtype=np.float64),
        frame_size_hw=(480, 640),
        keypoints_xy=np.ones((frame_count, 133, 2), dtype=np.float32),
        keypoint_scores=np.full((frame_count, 133), 0.9, dtype=np.float32),
        bboxes_xyxy=np.tile(
            np.asarray([10, 20, 300, 450], dtype=np.float32), (frame_count, 1)
        ),
        bbox_scores=np.full((frame_count,), 0.95, dtype=np.float32),
        person_detected=np.ones((frame_count,), dtype=np.bool_),
        metadata={
            "extractor_fingerprint": "extractor-v1",
            "video_sha256": "video-v1",
        },
    )


def test_pose_cache_round_trip_is_atomic_and_pickle_free(tmp_path: Path) -> None:
    sequence = _sequence()
    path = pose_cache_path(tmp_path, sequence.sample_id)

    write_pose_cache(path, sequence)
    restored = read_pose_cache(
        path,
        expected_sample_id=sequence.sample_id,
        expected_fingerprint="extractor-v1",
        expected_video_sha256="video-v1",
    )

    assert path.is_relative_to(tmp_path)
    assert restored.sample_id == sequence.sample_id
    np.testing.assert_array_equal(restored.keypoints_xy, sequence.keypoints_xy)
    assert not list(path.parent.glob("*.tmp"))


def test_pose_cache_detects_stale_provenance_and_refuses_overwrite(tmp_path: Path) -> None:
    sequence = _sequence()
    path = pose_cache_path(tmp_path, sequence.sample_id)
    write_pose_cache(path, sequence)

    assert cache_is_current(
        path,
        sample_id=sequence.sample_id,
        extractor_fingerprint="extractor-v1",
        video_sha256="video-v1",
    )
    assert not cache_is_current(
        path,
        sample_id=sequence.sample_id,
        extractor_fingerprint="extractor-v2",
        video_sha256="video-v1",
    )
    with pytest.raises(PoseCacheError, match="already exists"):
        write_pose_cache(path, sequence)


def test_sha256_file_returns_full_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pth"
    path.write_bytes(b"checkpoint")

    assert sha256_file(path) == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
