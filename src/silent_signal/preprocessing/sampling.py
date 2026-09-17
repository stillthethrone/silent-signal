"""Deterministic temporal sampling helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def temporal_sample_indices(frame_count: int, target_frames: int) -> NDArray[np.int64]:
    """Return evenly spaced nearest-frame indices, repeating short sequences."""

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1.")
    if target_frames < 1:
        raise ValueError("target_frames must be at least 1.")
    if frame_count == 1:
        return np.zeros((target_frames,), dtype=np.int64)
    positions = np.linspace(0.0, frame_count - 1, target_frames, dtype=np.float64)
    return np.rint(positions).astype(np.int64)
