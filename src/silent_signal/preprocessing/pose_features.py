"""Turn raw COCO-WholeBody pose into a fixed graph-ready representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from silent_signal.pose.cache import canonical_sha256
from silent_signal.pose.interface import RawPoseSequence
from silent_signal.pose.layouts import PoseLayout, get_pose_layout
from silent_signal.preprocessing.coordinates import (
    interpolate_short_gaps,
    normalize_signer_coordinates,
)
from silent_signal.preprocessing.sampling import temporal_sample_indices

GRAPH_POSE_SCHEMA_VERSION = 1
FEATURE_NAMES = ("x", "y", "confidence", "velocity_x", "velocity_y", "bone_x", "bone_y")


@dataclass(frozen=True, slots=True)
class GraphPreprocessConfig:
    """Versioned preprocessing choices that determine graph-cache identity."""

    layout_name: str = "coco_wholebody_75_v1"
    target_frames: int = 64
    confidence_threshold: float = 0.3
    interpolation_max_gap: int = 3
    include_velocity: bool = True
    include_bones: bool = True
    schema_version: int = GRAPH_POSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        get_pose_layout(self.layout_name)
        if self.schema_version != GRAPH_POSE_SCHEMA_VERSION:
            raise ValueError("Unsupported graph preprocessing schema_version.")
        if self.target_frames < 1:
            raise ValueError("target_frames must be at least 1.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if self.interpolation_max_gap < 0:
            raise ValueError("interpolation_max_gap must not be negative.")


@dataclass(frozen=True, slots=True)
class GraphPoseSample:
    """One fixed-length, graph-ready pose sample."""

    sample_id: str
    class_index: int
    split: str
    features: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    observed_mask: NDArray[np.bool_]
    frame_mask: NDArray[np.bool_]
    source_frame_indices: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float64]
    adjacency: NDArray[np.float32]
    metadata: dict[str, Any]


def preprocessing_fingerprint(config: GraphPreprocessConfig) -> str:
    """Hash config, layout semantics, and ordered feature channels."""

    layout = get_pose_layout(config.layout_name)
    return canonical_sha256(
        {
            "config": asdict(config),
            "feature_names": list(_feature_names(config)),
            "layout": {
                "name": layout.name,
                "source_layout": layout.source_layout,
                "joints": [
                    {
                        "name": joint.name,
                        "source_index": joint.source_index,
                        "body_part": joint.body_part,
                        "parent": joint.parent,
                    }
                    for joint in layout.joints
                ],
                "edges": [list(edge) for edge in layout.edges],
            },
        }
    )


def normalized_adjacency(layout: PoseLayout) -> NDArray[np.float32]:
    """Return symmetric degree-normalized adjacency with self loops."""

    adjacency = np.eye(layout.num_joints, dtype=np.float32)
    for left, right in layout.edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.power(degree, -0.5, dtype=np.float32)
    return (inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]).astype(np.float32)


def prepare_graph_pose(
    sequence: RawPoseSequence,
    *,
    class_index: int,
    split: str,
    config: GraphPreprocessConfig,
) -> GraphPoseSample:
    """Select, clean, normalize, sample, and featurize one raw pose sequence."""

    if class_index < 0:
        raise ValueError("class_index must not be negative.")
    if not split:
        raise ValueError("split must not be empty.")
    layout = get_pose_layout(config.layout_name)
    xy, scores = layout.select(sequence.keypoints_xy, sequence.keypoint_scores)
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(scores)
    observed = sequence.person_detected[:, None] & finite & (scores >= config.confidence_threshold)
    xy, scores, usable = interpolate_short_gaps(
        xy,
        scores,
        observed,
        max_gap=config.interpolation_max_gap,
    )
    normalized, _, _ = normalize_signer_coordinates(
        xy,
        usable,
        frame_size_hw=sequence.frame_size_hw,
    )
    temporal_indices = temporal_sample_indices(sequence.frame_count, config.target_frames)
    sampled_xy = normalized[temporal_indices]
    sampled_scores = scores[temporal_indices]
    sampled_usable = usable[temporal_indices]
    sampled_observed = observed[temporal_indices]

    channels: list[NDArray[np.float32]] = [sampled_xy]
    channels.append(np.where(sampled_usable, sampled_scores, 0.0)[..., None])
    if config.include_velocity:
        velocity = np.zeros_like(sampled_xy)
        velocity_valid = sampled_usable[1:] & sampled_usable[:-1]
        delta = sampled_xy[1:] - sampled_xy[:-1]
        velocity[1:] = np.where(velocity_valid[..., None], delta, 0.0)
        channels.append(velocity)
    if config.include_bones:
        channels.append(_bone_vectors(sampled_xy, sampled_usable, layout))
    features = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)
    features[~sampled_usable, :3] = 0.0

    metadata = {
        "schema_version": GRAPH_POSE_SCHEMA_VERSION,
        "layout_name": layout.name,
        "source_layout": layout.source_layout,
        "feature_names": list(_feature_names(config)),
        "preprocessing_fingerprint": preprocessing_fingerprint(config),
        "raw_extractor_fingerprint": sequence.metadata.get("extractor_fingerprint"),
        "video_sha256": sequence.metadata.get("video_sha256"),
        "raw_frame_count": sequence.frame_count,
        "observed_joint_ratio": float(observed.mean()),
        "usable_joint_ratio": float(usable.mean()),
    }
    return GraphPoseSample(
        sample_id=sequence.sample_id,
        class_index=class_index,
        split=split,
        features=features,
        joint_mask=sampled_usable,
        observed_mask=sampled_observed,
        frame_mask=sampled_usable.any(axis=1),
        source_frame_indices=sequence.frame_indices[temporal_indices],
        timestamps_seconds=sequence.timestamps_seconds[temporal_indices],
        adjacency=normalized_adjacency(layout),
        metadata=metadata,
    )


def _feature_names(config: GraphPreprocessConfig) -> tuple[str, ...]:
    names = ["x", "y", "confidence"]
    if config.include_velocity:
        names.extend(("velocity_x", "velocity_y"))
    if config.include_bones:
        names.extend(("bone_x", "bone_y"))
    return tuple(names)


def _bone_vectors(
    coordinates: NDArray[np.float32],
    usable: NDArray[np.bool_],
    layout: PoseLayout,
) -> NDArray[np.float32]:
    bones = np.zeros_like(coordinates)
    for index, parent in enumerate(layout.parents):
        if parent is None:
            continue
        valid = usable[:, index] & usable[:, parent]
        delta = coordinates[:, index] - coordinates[:, parent]
        bones[:, index] = np.where(valid[:, None], delta, 0.0)
    return bones
