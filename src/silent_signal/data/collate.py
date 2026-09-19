"""Framework-neutral batching for graph pose samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from silent_signal.preprocessing.pose_features import GraphPoseSample


@dataclass(frozen=True, slots=True)
class GraphPoseBatch:
    sample_ids: tuple[str, ...]
    features: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    frame_mask: NDArray[np.bool_]
    labels: NDArray[np.int64]
    adjacency: NDArray[np.float32]


def collate_graph_pose(samples: list[GraphPoseSample]) -> GraphPoseBatch:
    """Stack equal-shape graph samples and reject mixed graph layouts."""

    if not samples:
        raise ValueError("Cannot collate an empty graph batch.")
    adjacency = samples[0].adjacency
    if any(
        sample.features.shape != samples[0].features.shape
        or sample.adjacency.shape != adjacency.shape
        or not np.array_equal(sample.adjacency, adjacency)
        for sample in samples[1:]
    ):
        raise ValueError("All graph samples in a batch must share shape and adjacency.")
    return GraphPoseBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        features=np.stack([sample.features for sample in samples]),
        joint_mask=np.stack([sample.joint_mask for sample in samples]),
        frame_mask=np.stack([sample.frame_mask for sample in samples]),
        labels=np.asarray([sample.class_index for sample in samples], dtype=np.int64),
        adjacency=adjacency,
    )
