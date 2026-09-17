"""Framework-neutral datasets backed by graph pose caches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from silent_signal.contracts import ManifestRecord
from silent_signal.pose.cache import pose_cache_path
from silent_signal.preprocessing.cache import read_graph_pose_cache
from silent_signal.preprocessing.pose_features import GraphPoseSample


class GraphPoseDataset(Sequence[GraphPoseSample]):
    """Lazy NumPy dataset that preserves manifest labels and official splits."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        cache_root: str | Path,
        *,
        split: str | None = None,
        expected_fingerprint: str | None = None,
    ) -> None:
        self.records = tuple(
            record for record in records if split is None or record.split == split
        )
        if not self.records:
            raise ValueError("GraphPoseDataset selection is empty.")
        self.cache_root = Path(cache_root)
        self.expected_fingerprint = expected_fingerprint

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphPoseSample:
        record = self.records[index]
        sample = read_graph_pose_cache(
            pose_cache_path(self.cache_root, record.sample_id),
            expected_sample_id=record.sample_id,
            expected_fingerprint=self.expected_fingerprint,
        )
        if sample.class_index != record.class_index or sample.split != record.split:
            raise ValueError(f"Graph cache label/split mismatch for {record.sample_id}.")
        return sample
