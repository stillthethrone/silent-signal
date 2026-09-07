"""Framework-independent contracts for offline whole-body pose extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

COCO_WHOLEBODY_KEYPOINTS = 133


class PoseExtractionError(RuntimeError):
    """Raised when a video cannot produce a valid raw pose sequence."""


@dataclass(frozen=True, slots=True)
class PoseCandidate:
    """One person's COCO-WholeBody prediction in one video frame."""

    keypoints_xy: NDArray[np.float32]
    keypoint_scores: NDArray[np.float32]
    bbox_xyxy: NDArray[np.float32]
    bbox_score: float

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_xy, dtype=np.float32)
        scores = np.asarray(self.keypoint_scores, dtype=np.float32)
        bbox = np.asarray(self.bbox_xyxy, dtype=np.float32)
        if keypoints.shape != (COCO_WHOLEBODY_KEYPOINTS, 2):
            raise ValueError(
                "keypoints_xy must have shape "
                f"({COCO_WHOLEBODY_KEYPOINTS}, 2), found {keypoints.shape}."
            )
        if scores.shape != (COCO_WHOLEBODY_KEYPOINTS,):
            raise ValueError(
                "keypoint_scores must have shape "
                f"({COCO_WHOLEBODY_KEYPOINTS},), found {scores.shape}."
            )
        if bbox.shape != (4,):
            raise ValueError(f"bbox_xyxy must have shape (4,), found {bbox.shape}.")
        if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
            raise ValueError("Pose predictions must not contain NaN or infinity.")
        if not np.isfinite(bbox).all() or not np.isfinite(self.bbox_score):
            raise ValueError("Bounding-box predictions must not contain NaN or infinity.")
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2.")
        object.__setattr__(self, "keypoints_xy", keypoints)
        object.__setattr__(self, "keypoint_scores", scores)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "bbox_score", float(self.bbox_score))


@dataclass(frozen=True, slots=True)
class RawPoseSequence:
    """Unnormalized 133-keypoint output for one physical video clip.

    Missing detections use zero-valued numeric rows and are distinguished by
    ``person_detected``. Coordinates remain in source-image pixels so changes
    to downstream normalization never require rerunning RTMPose.
    """

    sample_id: str
    source_video: str
    frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
    frame_size_hw: tuple[int, int]
    keypoints_xy: NDArray[np.float32]
    keypoint_scores: NDArray[np.float32]
    bboxes_xyxy: NDArray[np.float32]
    bbox_scores: NDArray[np.float32]
    person_detected: NDArray[np.bool_]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        timestamps = np.asarray(self.timestamps_seconds, dtype=np.float64)
        keypoints = np.asarray(self.keypoints_xy, dtype=np.float32)
        scores = np.asarray(self.keypoint_scores, dtype=np.float32)
        bboxes = np.asarray(self.bboxes_xyxy, dtype=np.float32)
        bbox_scores = np.asarray(self.bbox_scores, dtype=np.float32)
        detected = np.asarray(self.person_detected, dtype=np.bool_)
        frame_count = len(frame_indices)
        expected = {
            "timestamps_seconds": (frame_count,),
            "keypoints_xy": (frame_count, COCO_WHOLEBODY_KEYPOINTS, 2),
            "keypoint_scores": (frame_count, COCO_WHOLEBODY_KEYPOINTS),
            "bboxes_xyxy": (frame_count, 4),
            "bbox_scores": (frame_count,),
            "person_detected": (frame_count,),
        }
        actual = {
            "timestamps_seconds": timestamps.shape,
            "keypoints_xy": keypoints.shape,
            "keypoint_scores": scores.shape,
            "bboxes_xyxy": bboxes.shape,
            "bbox_scores": bbox_scores.shape,
            "person_detected": detected.shape,
        }
        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, found {actual[name]}.")
        height, width = self.frame_size_hw
        if frame_count == 0:
            raise ValueError("A pose sequence must contain at least one decoded frame.")
        if height <= 0 or width <= 0:
            raise ValueError("frame_size_hw must contain positive height and width.")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing.")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps_seconds must be monotonically increasing.")
        for name, value in (
            ("timestamps_seconds", timestamps),
            ("keypoints_xy", keypoints),
            ("keypoint_scores", scores),
            ("bboxes_xyxy", bboxes),
            ("bbox_scores", bbox_scores),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must not contain NaN or infinity.")
        if np.any(scores < 0):
            raise ValueError("keypoint_scores must not be negative.")
        if np.any(bbox_scores < 0):
            raise ValueError("bbox_scores must not be negative.")
        missing = ~detected
        if missing.any() and (
            np.any(keypoints[missing])
            or np.any(scores[missing])
            or np.any(bboxes[missing])
            or np.any(bbox_scores[missing])
        ):
            raise ValueError("Frames without a person must contain zero-valued predictions.")
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(self, "keypoints_xy", keypoints)
        object.__setattr__(self, "keypoint_scores", scores)
        object.__setattr__(self, "bboxes_xyxy", bboxes)
        object.__setattr__(self, "bbox_scores", bbox_scores)
        object.__setattr__(self, "person_detected", detected)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def frame_count(self) -> int:
        """Number of decoded frames represented by the sequence."""

        return int(self.frame_indices.shape[0])


@runtime_checkable
class WholeBodyExtractor(Protocol):
    """Backend contract used by the extraction CLI."""

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint of model files, configuration, and software."""

    def extract_video(self, video_path: Path, *, sample_id: str) -> RawPoseSequence:
        """Extract unnormalized COCO-WholeBody keypoints from one video."""
