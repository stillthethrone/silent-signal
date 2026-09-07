from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from silent_signal.cli.extract_pose import _select_records, main
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import write_manifest
from silent_signal.pose.cache import pose_cache_path, read_pose_cache, sha256_file
from silent_signal.pose.interface import RawPoseSequence


class _FakeExtractor:
    fingerprint = "fake-extractor-v1"

    def __init__(self) -> None:
        self.calls = 0

    def extract_video(self, video_path: Path, *, sample_id: str) -> RawPoseSequence:
        self.calls += 1
        return RawPoseSequence(
            sample_id=sample_id,
            source_video=str(video_path),
            frame_indices=np.asarray([0], dtype=np.int64),
            timestamps_seconds=np.asarray([0.0], dtype=np.float64),
            frame_size_hw=(100, 120),
            keypoints_xy=np.ones((1, 133, 2), dtype=np.float32),
            keypoint_scores=np.ones((1, 133), dtype=np.float32),
            bboxes_xyxy=np.asarray([[5, 5, 100, 95]], dtype=np.float32),
            bbox_scores=np.asarray([0.9], dtype=np.float32),
            person_detected=np.asarray([True], dtype=np.bool_),
            metadata={
                "extractor_fingerprint": self.fingerprint,
                "video_sha256": sha256_file(video_path),
            },
        )


def _write_pose_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "extractor": {
                    "name": "rtmpose_l_coco_wholebody_384x288",
                    "framework": "mmpose",
                    "variant": "rtmpose-l",
                    "training_dataset": "coco-wholebody",
                    "raw_layout": "coco_wholebody_133",
                    "input_size_wh": [288, 384],
                    "hash_source_video": True,
                    "pose_model": {
                        "config": str(
                            path.parent / "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
                        ),
                        "checkpoint": str(path.parent / "pose.pth"),
                        "checkpoint_url": "https://example.test/pose.pth",
                    },
                    "detector": {
                        "config": str(path.parent / "rtmdet_m_coco-person.py"),
                        "checkpoint": str(path.parent / "detector.pth"),
                        "checkpoint_url": "https://example.test/detector.pth",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_extract_cli_writes_cache_report_and_resumes_current_result(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ASL_Citizen"
    video = dataset_root / "videos" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        (
            ManifestRecord(
                sample_id="clip",
                instance_id="clip",
                video_id="clip",
                signer_id="signer",
                gloss_id="hello",
                gloss_name="hello",
                class_index=0,
                view="single",
                video_path="videos/clip.mp4",
                split="train",
            ),
        ),
        manifest,
    )
    config = tmp_path / "rtmpose.yaml"
    _write_pose_config(config)
    output_root = tmp_path / "pose-cache"
    report = tmp_path / "report.json"
    extractor = _FakeExtractor()
    arguments = [
        "extract",
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--dataset-root",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--report",
        str(report),
        "--split",
        "train",
        "--progress-every",
        "0",
    ]

    assert main(arguments, extractor_factory=lambda _config: extractor) == 0
    cache = pose_cache_path(output_root, "clip")
    assert read_pose_cache(cache).sample_id == "clip"
    assert json.loads(report.read_text(encoding="utf-8"))["extracted"] == 1

    assert main(arguments, extractor_factory=lambda _config: extractor) == 0
    assert extractor.calls == 1
    resumed = json.loads(report.read_text(encoding="utf-8"))
    assert resumed["resumed"] == 1
    assert resumed["extracted"] == 0
    assert resumed["selection"]["num_shards"] == 1


def test_record_shards_are_sorted_deterministic_and_disjoint() -> None:
    records = tuple(
        ManifestRecord(
            sample_id=sample_id,
            instance_id=sample_id,
            video_id=sample_id,
            signer_id="signer",
            gloss_id="hello",
            gloss_name="hello",
            class_index=0,
            view="single",
            video_path=f"videos/{sample_id}.mp4",
            split="train",
        )
        for sample_id in ("d", "b", "e", "a", "c")
    )

    first = _select_records(
        records,
        split="train",
        sample_ids=frozenset(),
        limit=None,
        num_shards=2,
        shard_index=0,
    )
    second = _select_records(
        tuple(reversed(records)),
        split="train",
        sample_ids=frozenset(),
        limit=None,
        num_shards=2,
        shard_index=1,
    )

    assert [record.sample_id for record in first] == ["a", "c", "e"]
    assert [record.sample_id for record in second] == ["b", "d"]
    assert {record.sample_id for record in first}.isdisjoint(record.sample_id for record in second)
