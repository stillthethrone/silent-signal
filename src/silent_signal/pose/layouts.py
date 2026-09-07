"""Versioned joint subsets and graph topology for sign-language models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from silent_signal.pose.interface import COCO_WHOLEBODY_KEYPOINTS


@dataclass(frozen=True, slots=True)
class JointDefinition:
    """One model joint and its location in the source pose estimator output."""

    name: str
    source_index: int
    body_part: str
    parent: int | None


@dataclass(frozen=True, slots=True)
class PoseLayout:
    """Immutable mapping from COCO-WholeBody output to model graph nodes."""

    name: str
    source_layout: str
    joints: tuple[JointDefinition, ...]
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.source_layout:
            raise ValueError("Pose layout names must not be empty.")
        if not self.joints:
            raise ValueError("A pose layout must contain at least one joint.")
        names = tuple(joint.name for joint in self.joints)
        source_indices = tuple(joint.source_index for joint in self.joints)
        if len(set(names)) != len(names):
            raise ValueError("Pose joint names must be unique.")
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("Pose source indices must be unique.")
        if any(index < 0 or index >= COCO_WHOLEBODY_KEYPOINTS for index in source_indices):
            raise ValueError("Pose source indices must refer to COCO-WholeBody's 133 joints.")
        joint_count = len(self.joints)
        if any(parent is not None and not 0 <= parent < joint_count for parent in self.parents):
            raise ValueError("Every parent must be a valid model-joint index.")
        normalized_edges = tuple((min(a, b), max(a, b)) for a, b in self.edges)
        if any(a == b or a < 0 or b >= joint_count for a, b in normalized_edges):
            raise ValueError("Every graph edge must connect two distinct model joints.")
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("Pose graph edges must be unique.")

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(joint.source_index for joint in self.joints)

    @property
    def parents(self) -> tuple[int | None, ...]:
        return tuple(joint.parent for joint in self.joints)

    @property
    def index_by_name(self) -> MappingProxyType[str, int]:
        return MappingProxyType({joint.name: index for index, joint in enumerate(self.joints)})

    def select(
        self,
        keypoints_xy: NDArray[np.floating],
        keypoint_scores: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Select layout joints from arrays ending in ``[133, 2]`` and ``[133]``."""

        xy = np.asarray(keypoints_xy)
        scores = np.asarray(keypoint_scores)
        if xy.shape[-2:] != (COCO_WHOLEBODY_KEYPOINTS, 2):
            raise ValueError("keypoints_xy must end with shape (133, 2).")
        if scores.shape[-1:] != (COCO_WHOLEBODY_KEYPOINTS,):
            raise ValueError("keypoint_scores must end with shape (133,).")
        if xy.shape[:-2] != scores.shape[:-1]:
            raise ValueError("Keypoint coordinates and scores must share leading dimensions.")
        indices = np.asarray(self.source_indices, dtype=np.int64)
        return (
            np.take(xy, indices, axis=-2).astype(np.float32, copy=False),
            np.take(scores, indices, axis=-1).astype(np.float32, copy=False),
        )


_BODY_NAMES: Final = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

_HAND_LANDMARK_NAMES: Final = (
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
)

_FACE_SOURCE_INDICES: Final = (
    40,
    44,
    45,
    49,
    59,
    62,
    65,
    68,
    71,
    74,
    77,
    80,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)

_FACE_NAMES: Final = (
    "right_brow_outer",
    "right_brow_inner",
    "left_brow_inner",
    "left_brow_outer",
    "right_eye_outer",
    "right_eye_inner",
    "left_eye_inner",
    "left_eye_outer",
    "mouth_left",
    "upper_lip",
    "mouth_right",
    "lower_lip",
    "inner_mouth_0",
    "inner_mouth_1",
    "inner_mouth_2",
    "inner_mouth_3",
    "inner_mouth_4",
    "inner_mouth_5",
    "inner_mouth_6",
    "inner_mouth_7",
)


def _hand_parent(local_index: int, hand_offset: int, body_wrist: int) -> int:
    if local_index == 0:
        return body_wrist
    if local_index in {1, 5, 9, 13, 17}:
        return hand_offset
    return hand_offset + local_index - 1


def _build_asl_layout() -> PoseLayout:
    joints: list[JointDefinition] = []
    body_parents: tuple[int | None, ...] = (
        None,
        0,
        0,
        1,
        2,
        None,
        5,
        5,
        6,
        7,
        8,
        5,
        6,
    )
    joints.extend(
        JointDefinition(name, source_index, "body", body_parents[source_index])
        for source_index, name in enumerate(_BODY_NAMES)
    )

    left_offset = len(joints)
    joints.extend(
        JointDefinition(
            f"left_hand_{name}",
            91 + local_index,
            "left_hand",
            _hand_parent(local_index, left_offset, 9),
        )
        for local_index, name in enumerate(_HAND_LANDMARK_NAMES)
    )

    right_offset = len(joints)
    joints.extend(
        JointDefinition(
            f"right_hand_{name}",
            112 + local_index,
            "right_hand",
            _hand_parent(local_index, right_offset, 10),
        )
        for local_index, name in enumerate(_HAND_LANDMARK_NAMES)
    )

    face_offset = len(joints)
    face_local_parents: tuple[int | None, ...] = (
        None,
        face_offset,
        None,
        face_offset + 2,
        None,
        face_offset + 4,
        None,
        face_offset + 6,
        None,
        face_offset + 8,
        face_offset + 9,
        face_offset + 10,
        None,
        face_offset + 12,
        face_offset + 13,
        face_offset + 14,
        face_offset + 15,
        face_offset + 16,
        face_offset + 17,
        face_offset + 18,
    )
    joints.extend(
        JointDefinition(name, source_index, "face", face_local_parents[local_index])
        for local_index, (name, source_index) in enumerate(
            zip(_FACE_NAMES, _FACE_SOURCE_INDICES, strict=True)
        )
    )

    parent_edges = {
        (min(index, joint.parent), max(index, joint.parent))
        for index, joint in enumerate(joints)
        if joint.parent is not None
    }
    parent_edges.update(
        {
            (face_offset + 8, face_offset + 11),
            (face_offset + 12, face_offset + 19),
        }
    )
    return PoseLayout(
        name="asl_citizen_coco_wholebody_v1",
        source_layout="coco_wholebody_133",
        joints=tuple(joints),
        edges=tuple(sorted(parent_edges)),
    )


ASL_CITIZEN_WHOLEBODY_V1: Final = _build_asl_layout()
POSE_LAYOUTS: Final = MappingProxyType(
    {ASL_CITIZEN_WHOLEBODY_V1.name: ASL_CITIZEN_WHOLEBODY_V1}
)


def get_pose_layout(name: str) -> PoseLayout:
    """Return a registered layout or raise a configuration-friendly error."""

    try:
        return POSE_LAYOUTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(POSE_LAYOUTS))
        raise ValueError(f"Unknown pose layout {name!r}; available layouts: {choices}.") from exc
