"""Graph features from MediaPipe Holistic keypoints ``[T, 76, 3]`` (Kaggle VSL release).

Coordinates arrive already normalized by the uploader: the body into [-0.5, 0.5] of a box
around nose, shoulders and hips, and each hand into its own box. Missing detections are
(0, 0, 0). This module keeps that geometry, recovers masks from the zero rows, fills
short gaps, resamples to a fixed length and adds velocity and bone channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent_signal.pose.layouts import MEDIAPIPE_76_JOINTS, get_pose_layout
from silent_signal.preprocessing.coordinates import interpolate_short_gaps
from silent_signal.preprocessing.sampling import temporal_sample_indices


@dataclass(frozen=True, slots=True)
class MediaPipeFeatureConfig:
    """Choices that define the model input ``[target_frames, joints, channels]``."""

    layout_name: str = "mediapipe_upper68_v1"
    target_frames: int = 64
    interpolation_max_gap: int = 3
    include_velocity: bool = True
    include_bones: bool = True

    def __post_init__(self) -> None:
        get_pose_layout(self.layout_name)
        if self.target_frames < 1 or self.interpolation_max_gap < 0:
            raise ValueError("target_frames must be positive and interpolation_max_gap >= 0.")

    @property
    def channels(self) -> int:
        return 3 * (1 + int(self.include_velocity) + int(self.include_bones))


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Training-only perturbations; evaluation always uses the deterministic view."""

    min_keep_ratio: float = 0.8
    rotation_degrees: float = 10.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    shift: float = 0.05
    noise_std: float = 0.005

    def __post_init__(self) -> None:
        if not 0 < self.min_keep_ratio <= 1:
            raise ValueError("min_keep_ratio must be in (0, 1].")
        if self.rotation_degrees < 0 or self.shift < 0 or self.noise_std < 0:
            raise ValueError("Augmentation magnitudes must not be negative.")
        low, high = self.scale_range
        if not 0 < low <= high:
            raise ValueError("scale_range must be positive and ordered.")


def validate_raw_keypoints(raw: ArrayLike) -> NDArray[np.float32]:
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (MEDIAPIPE_76_JOINTS, 3) or array.shape[0] < 1:
        raise ValueError(f"Expected MediaPipe keypoints [T, 76, 3], found {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("MediaPipe keypoints contain NaN or infinity.")
    return array


def mediapipe_graph_features(
    raw: ArrayLike,
    config: MediaPipeFeatureConfig,
    *,
    rng: np.random.Generator | None = None,
    augmentation: AugmentationConfig | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
    """Return ``features [F, V, C]``, ``joint_mask [F, V]`` and ``frame_mask [F]``.

    With ``rng`` and ``augmentation``, a random temporal window and a small random
    similarity transform of the x/y plane are applied; masked joints stay zero.
    """

    layout = get_pose_layout(config.layout_name)
    xyz = validate_raw_keypoints(raw)[:, list(layout.source_indices), :]
    observed = np.any(xyz != 0.0, axis=-1)
    xyz, _, usable = interpolate_short_gaps(
        xyz,
        np.ones(observed.shape, dtype=np.float32),
        observed,
        max_gap=config.interpolation_max_gap,
    )
    frames = xyz.shape[0]
    if rng is not None and augmentation is not None:
        indices = _random_window(frames, config.target_frames, augmentation.min_keep_ratio, rng)
        xyz = _random_similarity(xyz, usable, augmentation, rng)
    else:
        indices = temporal_sample_indices(frames, config.target_frames)
    sampled = xyz[indices]
    mask = usable[indices]

    channels = [sampled]
    if config.include_velocity:
        velocity = np.zeros_like(sampled)
        valid = mask[1:] & mask[:-1]
        velocity[1:] = np.where(valid[..., None], sampled[1:] - sampled[:-1], 0.0)
        channels.append(velocity)
    if config.include_bones:
        bones = np.zeros_like(sampled)
        for joint, parent in enumerate(layout.parents):
            if parent is None:
                continue
            valid = mask[:, joint] & mask[:, parent]
            bones[:, joint] = np.where(valid[:, None], sampled[:, joint] - sampled[:, parent], 0.0)
        channels.append(bones)
    features = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)
    features[~mask] = 0.0
    return features, mask, mask.any(axis=1)


def _random_window(
    frames: int, target: int, min_keep_ratio: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    keep = max(1, round(frames * rng.uniform(min_keep_ratio, 1.0)))
    start = int(rng.integers(0, frames - keep + 1))
    positions = np.linspace(start, start + keep - 1, target)
    if keep > 1:
        positions += rng.uniform(-0.5, 0.5, size=target)
    return np.clip(np.rint(np.sort(positions)), start, start + keep - 1).astype(np.int64)


def _random_similarity(
    xyz: NDArray[np.float32],
    usable: NDArray[np.bool_],
    augmentation: AugmentationConfig,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    angle = math.radians(rng.uniform(-augmentation.rotation_degrees, augmentation.rotation_degrees))
    scale = rng.uniform(*augmentation.scale_range)
    rotation = scale * np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32
    )
    shift = rng.uniform(-augmentation.shift, augmentation.shift, size=2).astype(np.float32)
    output = xyz.copy()
    output[..., :2] = xyz[..., :2] @ rotation.T + shift
    if augmentation.noise_std:
        output += rng.normal(0.0, augmentation.noise_std, size=output.shape).astype(np.float32)
    output[~usable] = 0.0
    return output
