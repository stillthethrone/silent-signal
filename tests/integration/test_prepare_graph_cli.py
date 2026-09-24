from __future__ import annotations

from pathlib import Path

import numpy as np

from silent_signal.cli.prepare_graph import main
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import write_manifest
from silent_signal.pose.cache import pose_cache_path, write_pose_cache
from silent_signal.pose.interface import RawPoseSequence


def test_prepare_graph_cli_writes_and_resumes_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pose_root = tmp_path / "raw"
    graph_root = tmp_path / "graph"
    report = tmp_path / "report.json"
    config = tmp_path / "graph.yaml"
    record = ManifestRecord(
        sample_id="sample-1",
        instance_id="instance-1",
        video_id="video-1",
        signer_id="signer-1",
        gloss_id="g1",
        gloss_name="hello",
        class_index=0,
        view="single",
        video_path="videos/sample.mp4",
        split="train",
    )
    write_manifest([record], manifest)
    write_pose_cache(pose_cache_path(pose_root, record.sample_id), _sequence())
    config.write_text(
        """schema_version: 1
preprocessing:
  layout_name: coco_wholebody_75_v1
  target_frames: 8
  confidence_threshold: 0.3
  interpolation_max_gap: 1
  include_velocity: true
  include_bones: true
expected:
  extractor_fingerprint: extractor-v1
""",
        encoding="utf-8",
    )
    args = [
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--pose-root",
        str(pose_root),
        "--output-root",
        str(graph_root),
        "--report",
        str(report),
        "--progress-every",
        "1",
    ]

    assert main(args) == 0
    assert main(args) == 0
    assert '"resumed": 1' in report.read_text(encoding="utf-8")


def _sequence() -> RawPoseSequence:
    frames = 3
    return RawPoseSequence(
        sample_id="sample-1",
        source_video="videos/sample.mp4",
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps_seconds=np.arange(frames, dtype=np.float64) / 25.0,
        frame_size_hw=(480, 640),
        keypoints_xy=np.ones((frames, 133, 2), dtype=np.float32),
        keypoint_scores=np.full((frames, 133), 0.9, dtype=np.float32),
        bboxes_xyxy=np.tile(np.asarray([0, 0, 639, 479], dtype=np.float32), (frames, 1)),
        bbox_scores=np.ones((frames,), dtype=np.float32),
        person_detected=np.ones((frames,), dtype=np.bool_),
        metadata={"extractor_fingerprint": "extractor-v1", "video_sha256": "video-v1"},
    )
