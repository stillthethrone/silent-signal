"""Convert pre-extracted MediaPipe Holistic arrays into graph caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from silent_signal.pose.cache import canonical_sha256, sha256_file
from silent_signal.preprocessing.coordinates import (
    interpolate_short_gaps,
    normalize_signer_coordinates,
)
from silent_signal.preprocessing.pose_features import GraphPoseSample
from silent_signal.preprocessing.sampling import temporal_sample_indices

MEDIAPIPE_GRAPH_SCHEMA_VERSION: Final = 1
MEDIAPIPE_SOURCE_JOINTS: Final = 75
MEDIAPIPE_ACCEPTED_SOURCE_JOINTS: Final = (75, 76)
MEDIAPIPE_FEATURE_NAMES: Final = (
    "x",
    "y",
    "presence",
    "velocity_x",
    "velocity_y",
    "bone_x",
    "bone_y",
)

_POSE_NAMES: Final = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_pose",
    "right_pinky_pose",
    "left_index_pose",
    "right_index_pose",
    "left_thumb_pose",
    "right_thumb_pose",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

_HAND_NAMES: Final = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

MEDIAPIPE_JOINT_NAMES: Final = (
    *_POSE_NAMES,
    *(f"left_hand_{name}" for name in _HAND_NAMES),
    *(f"right_hand_{name}" for name in _HAND_NAMES),
)

_POSE_PARENTS: Final[tuple[int | None, ...]] = (
    None,
    0,
    1,
    2,
    0,
    4,
    5,
    3,
    6,
    0,
    0,
    0,
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    15,
    16,
    15,
    16,
    11,
    12,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
)


def _hand_parents(offset: int, body_wrist: int) -> tuple[int, ...]:
    parents: list[int] = [body_wrist]
    for local_index in range(1, len(_HAND_NAMES)):
        if local_index in {1, 5, 9, 13, 17}:
            parents.append(offset)
        else:
            parents.append(offset + local_index - 1)
    return tuple(parents)


MEDIAPIPE_PARENTS: Final = (
    *_POSE_PARENTS,
    *_hand_parents(33, 15),
    *_hand_parents(54, 16),
)


def _layout_edges() -> tuple[tuple[int, int], ...]:
    edges = {
        (min(index, parent), max(index, parent))
        for index, parent in enumerate(MEDIAPIPE_PARENTS)
        if parent is not None
    }
    edges.update(
        {
            (11, 12),
            (17, 19),
            (18, 20),
            (19, 21),
            (20, 22),
            (23, 24),
            (27, 31),
            (28, 32),
        }
    )
    return tuple(sorted(edges))


MEDIAPIPE_EDGES: Final = _layout_edges()


@dataclass(frozen=True, slots=True)
class MediaPipeGraphPreprocessConfig:
    """Versioned conversion contract for Kaggle MediaPipe arrays."""

    layout_name: str = "mediapipe_holistic_75_v1"
    target_frames: int = 64
    interpolation_max_gap: int = 3
    zero_epsilon: float = 1e-8
    schema_version: int = MEDIAPIPE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.layout_name != "mediapipe_holistic_75_v1":
            raise ValueError("Unsupported MediaPipe layout.")
        if self.target_frames < 1:
            raise ValueError("target_frames must be at least 1.")
        if self.interpolation_max_gap < 0:
            raise ValueError("interpolation_max_gap must not be negative.")
        if self.zero_epsilon < 0:
            raise ValueError("zero_epsilon must not be negative.")
        if self.schema_version != MEDIAPIPE_GRAPH_SCHEMA_VERSION:
            raise ValueError("Unsupported MediaPipe graph schema version.")


def mediapipe_preprocessing_fingerprint(config: MediaPipeGraphPreprocessConfig) -> str:
    """Return a stable identity for the MediaPipe graph representation."""

    return canonical_sha256(
        {
            "config": asdict(config),
            "source_layout": "mediapipe_holistic_33_pose_21_left_21_right",
            "ignored_source_indices": [75],
            "joint_names": list(MEDIAPIPE_JOINT_NAMES),
            "parents": list(MEDIAPIPE_PARENTS),
            "edges": [list(edge) for edge in MEDIAPIPE_EDGES],
            "feature_names": list(MEDIAPIPE_FEATURE_NAMES),
        }
    )


def load_mediapipe_array(path: str | Path) -> NDArray[np.float32]:
    """Load and validate one safe ``[T, 75|76, 3]`` NumPy array."""

    source = Path(path)
    loaded = np.load(source, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise ValueError(f"Expected a .npy array, found an .npz archive: {source}")
    array = np.asarray(loaded, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] not in MEDIAPIPE_ACCEPTED_SOURCE_JOINTS:
        raise ValueError(
            "MediaPipe keypoints must have shape [T, 75, 3] or [T, 76, 3]; "
            f"found {array.shape}."
        )
    if array.shape[0] < 1 or array.shape[2] != 3:
        raise ValueError(f"MediaPipe keypoints must have shape [T, V, 3]; found {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("MediaPipe keypoints contain NaN or infinity.")
    return array


def prepare_mediapipe_graph_pose(
    keypoints: NDArray[np.floating],
    *,
    sample_id: str,
    class_index: int,
    split: str,
    config: MediaPipeGraphPreprocessConfig,
    source_path: str | Path | None = None,
    fps: float | None = None,
) -> GraphPoseSample:
    """Convert one MediaPipe sequence into the existing seven-channel graph contract."""

    if class_index < 0:
        raise ValueError("class_index must not be negative.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    source = np.asarray(keypoints, dtype=np.float32)
    if source.ndim != 3 or source.shape[1] not in MEDIAPIPE_ACCEPTED_SOURCE_JOINTS:
        raise ValueError("MediaPipe keypoints must have 75 or 76 joints.")
    if source.shape[0] < 1 or source.shape[2] != 3 or not np.isfinite(source).all():
        raise ValueError("MediaPipe keypoints must be a finite [T, V, 3] array.")

    selected = source[:, :MEDIAPIPE_SOURCE_JOINTS]
    observed = np.any(np.abs(selected) > config.zero_epsilon, axis=-1)
    if not observed.any():
        raise ValueError("MediaPipe sequence contains no observed joints.")

    xy = selected[..., :2]
    presence = observed.astype(np.float32)
    xy, presence, usable = interpolate_short_gaps(
        xy,
        presence,
        observed,
        max_gap=config.interpolation_max_gap,
    )
    normalized, _, _ = normalize_signer_coordinates(
        xy,
        usable,
        frame_size_hw=(1, 1),
        left_shoulder=11,
        right_shoulder=12,
        left_hip=23,
        right_hip=24,
    )

    temporal_indices = temporal_sample_indices(source.shape[0], config.target_frames)
    sampled_xy = normalized[temporal_indices]
    sampled_presence = presence[temporal_indices]
    sampled_usable = usable[temporal_indices]
    sampled_observed = observed[temporal_indices]

    velocity = np.zeros_like(sampled_xy)
    valid_velocity = sampled_usable[1:] & sampled_usable[:-1]
    velocity[1:] = np.where(
        valid_velocity[..., None],
        sampled_xy[1:] - sampled_xy[:-1],
        0.0,
    )
    bones = _bone_vectors(sampled_xy, sampled_usable)
    features = np.concatenate(
        (
            sampled_xy,
            np.where(sampled_usable, sampled_presence, 0.0)[..., None],
            velocity,
            bones,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    features[~sampled_usable] = 0.0

    source_fps = float(fps) if fps is not None and fps > 0 else 1.0
    source_indices = temporal_indices.astype(np.int64, copy=False)
    metadata: dict[str, Any] = {
        "schema_version": MEDIAPIPE_GRAPH_SCHEMA_VERSION,
        "layout_name": config.layout_name,
        "source_layout": "mediapipe_holistic_33_pose_21_left_21_right",
        "source_joint_count": int(source.shape[1]),
        "ignored_source_indices": [75] if source.shape[1] == 76 else [],
        "feature_names": list(MEDIAPIPE_FEATURE_NAMES),
        "z_coordinate_policy": "preserved_in_source_but_not_used_by_v1_graph",
        "confidence_policy": "binary_presence_from_nonzero_xyz",
        "preprocessing_fingerprint": mediapipe_preprocessing_fingerprint(config),
        "raw_frame_count": int(source.shape[0]),
        "observed_joint_ratio": float(observed.mean()),
        "usable_joint_ratio": float(usable.mean()),
    }
    if source_path is not None:
        source_file = Path(source_path)
        metadata["source_path"] = str(source_file)
        if source_file.is_file():
            metadata["source_sha256"] = sha256_file(source_file)

    return GraphPoseSample(
        sample_id=sample_id,
        class_index=class_index,
        split=split,
        features=features,
        joint_mask=sampled_usable,
        observed_mask=sampled_observed,
        frame_mask=sampled_usable.any(axis=1),
        source_frame_indices=source_indices,
        timestamps_seconds=source_indices.astype(np.float64) / source_fps,
        adjacency=_normalized_adjacency(),
        metadata=metadata,
    )


def _normalized_adjacency() -> NDArray[np.float32]:
    adjacency = np.eye(MEDIAPIPE_SOURCE_JOINTS, dtype=np.float32)
    for left, right in MEDIAPIPE_EDGES:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.power(degree, -0.5, dtype=np.float32)
    return (inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]).astype(np.float32)


def _bone_vectors(
    coordinates: NDArray[np.float32],
    usable: NDArray[np.bool_],
) -> NDArray[np.float32]:
    bones = np.zeros_like(coordinates)
    for joint_index, parent in enumerate(MEDIAPIPE_PARENTS):
        if parent is None:
            continue
        valid = usable[:, joint_index] & usable[:, parent]
        bones[:, joint_index] = np.where(
            valid[:, None],
            coordinates[:, joint_index] - coordinates[:, parent],
            0.0,
        )
    return bones
