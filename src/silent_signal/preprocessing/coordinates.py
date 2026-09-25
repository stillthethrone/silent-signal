"""Coordinate cleaning and signer-centric normalization for pose sequences."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


def interpolate_short_gaps(
    coordinates: NDArray[np.floating],
    scores: NDArray[np.floating],
    observed_mask: NDArray[np.bool_],
    *,
    max_gap: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Linearly fill only gaps bracketed by observations from the same joint.

    ``coordinates`` has shape ``[T, V, D]`` (2D or 3D points); scores and masks ``[T, V]``.
    """

    if max_gap < 0:
        raise ValueError("max_gap must not be negative.")
    xy = np.asarray(coordinates, dtype=np.float32).copy()
    confidence = np.asarray(scores, dtype=np.float32).copy()
    observed = np.asarray(observed_mask, dtype=np.bool_)
    if xy.ndim != 3 or xy.shape[-1] < 1:
        raise ValueError("coordinates must have shape [T, V, D].")
    if confidence.shape != xy.shape[:2] or observed.shape != xy.shape[:2]:
        raise ValueError("scores and observed_mask must have shape [T, V].")

    usable = observed.copy()
    if max_gap == 0:
        return xy, confidence, usable
    for joint_index in range(xy.shape[1]):
        valid_indices = np.flatnonzero(observed[:, joint_index])
        for left, right in pairwise(valid_indices):
            gap = int(right - left - 1)
            if gap < 1 or gap > max_gap:
                continue
            fractions = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
            xy[left + 1 : right, joint_index] = (
                xy[left, joint_index][None, :] * (1.0 - fractions[:, None])
                + xy[right, joint_index][None, :] * fractions[:, None]
            )
            confidence[left + 1 : right, joint_index] = (
                confidence[left, joint_index] * (1.0 - fractions)
                + confidence[right, joint_index] * fractions
            )
            usable[left + 1 : right, joint_index] = True
    xy[~usable] = 0.0
    confidence[~usable] = 0.0
    return xy, confidence, usable


def normalize_signer_coordinates(
    coordinates: NDArray[np.floating],
    usable_mask: NDArray[np.bool_],
    *,
    frame_size_hw: tuple[int, int],
    left_shoulder: int = 5,
    right_shoulder: int = 6,
    left_hip: int = 11,
    right_hip: int = 12,
    epsilon: float = 1e-6,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Normalize pixels by frame size, body center, and a robust per-frame scale."""

    xy = np.asarray(coordinates, dtype=np.float32)
    mask = np.asarray(usable_mask, dtype=np.bool_)
    if xy.ndim != 3 or xy.shape[-1] != 2 or mask.shape != xy.shape[:2]:
        raise ValueError("coordinates/mask must have shapes [T, V, 2] and [T, V].")
    height, width = frame_size_hw
    if height <= 0 or width <= 0:
        raise ValueError("frame_size_hw values must be positive.")

    unit = xy / np.asarray([width, height], dtype=np.float32)
    normalized = np.zeros_like(unit, dtype=np.float32)
    centers = np.zeros((xy.shape[0], 2), dtype=np.float32)
    scales = np.ones((xy.shape[0],), dtype=np.float32)
    for frame_index in range(xy.shape[0]):
        valid = mask[frame_index]
        if not np.any(valid):
            continue
        points = unit[frame_index, valid]
        center, body_scale = _body_reference(
            unit[frame_index],
            valid,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_hip=left_hip,
            right_hip=right_hip,
        )
        extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        scale = float(body_scale if body_scale > epsilon else extent)
        if not np.isfinite(scale) or scale <= epsilon:
            scale = 1.0
        centers[frame_index] = center
        scales[frame_index] = scale
        normalized[frame_index, valid] = (points - center) / scale
    normalized[~mask] = 0.0
    return normalized, centers, scales


def _body_reference(
    frame: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    left_shoulder: int,
    right_shoulder: int,
    left_hip: int,
    right_hip: int,
) -> tuple[NDArray[np.float32], float]:
    for left, right in (
        (left_shoulder, right_shoulder),
        (left_hip, right_hip),
    ):
        if left < len(valid) and right < len(valid) and valid[left] and valid[right]:
            center = (frame[left] + frame[right]) * 0.5
            return center, float(np.linalg.norm(frame[left] - frame[right]))
    points = frame[valid]
    return points.mean(axis=0), 0.0
