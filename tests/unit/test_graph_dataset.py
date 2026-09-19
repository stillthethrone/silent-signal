from __future__ import annotations

from pathlib import Path

import numpy as np

from silent_signal.contracts import ManifestRecord
from silent_signal.data.collate import collate_graph_pose
from silent_signal.data.dataset import GraphPoseDataset
from silent_signal.pose.cache import pose_cache_path
from silent_signal.preprocessing.cache import write_graph_pose_cache
from silent_signal.preprocessing.pose_features import GraphPoseSample


def test_graph_dataset_preserves_manifest_label_and_collates(tmp_path: Path) -> None:
    record = ManifestRecord(
        sample_id="sample-1",
        instance_id="instance-1",
        video_id="video-1",
        signer_id="signer-1",
        gloss_id="g1",
        gloss_name="hello",
        class_index=3,
        view="single",
        video_path="videos/sample.mp4",
        split="train",
    )
    sample = GraphPoseSample(
        sample_id=record.sample_id,
        class_index=record.class_index,
        split="train",
        features=np.zeros((4, 75, 7), dtype=np.float32),
        joint_mask=np.ones((4, 75), dtype=np.bool_),
        observed_mask=np.ones((4, 75), dtype=np.bool_),
        frame_mask=np.ones((4,), dtype=np.bool_),
        source_frame_indices=np.arange(4, dtype=np.int64),
        timestamps_seconds=np.arange(4, dtype=np.float64),
        adjacency=np.eye(75, dtype=np.float32),
        metadata={"preprocessing_fingerprint": "fingerprint"},
    )
    write_graph_pose_cache(pose_cache_path(tmp_path, record.sample_id), sample)

    dataset = GraphPoseDataset(
        [record],
        tmp_path,
        split="train",
        expected_fingerprint="fingerprint",
    )
    batch = collate_graph_pose([dataset[0], dataset[0]])

    assert len(dataset) == 1
    assert batch.features.shape == (2, 4, 75, 7)
    assert batch.labels.tolist() == [3, 3]
