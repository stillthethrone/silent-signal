from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from silent_signal.cli import extract_pose, prepare_graph, prepare_multi_vsl_pose
from silent_signal.data.manifest import read_manifest
from silent_signal.pose.cache import pose_cache_path, sha256_file
from silent_signal.pose.interface import RawPoseSequence

_SPLITS = {
    "train": ("train_1_200_center_ord1.csv", "01"),
    "validation": ("val_1_200_center_ord1.csv", "02"),
    "test": ("test_1_200_center_ord1.csv", "03"),
}


class _FakeExtractor:
    fingerprint = "fake-rtmpose-v1"

    def extract_video(self, video_path: Path, *, sample_id: str) -> RawPoseSequence:
        frames = 4
        return RawPoseSequence(
            sample_id=sample_id,
            source_video=str(video_path),
            frame_indices=np.arange(frames, dtype=np.int64),
            timestamps_seconds=np.arange(frames, dtype=np.float64) / 30.0,
            frame_size_hw=(720, 566),
            keypoints_xy=np.random.default_rng(0)
            .uniform(50, 500, (frames, 133, 2))
            .astype(np.float32),
            keypoint_scores=np.full((frames, 133), 0.9, dtype=np.float32),
            bboxes_xyxy=np.tile(np.asarray([10, 10, 550, 710], dtype=np.float32), (frames, 1)),
            bbox_scores=np.ones((frames,), dtype=np.float32),
            person_detected=np.ones((frames,), dtype=np.bool_),
            metadata={
                "extractor_fingerprint": self.fingerprint,
                "video_sha256": sha256_file(video_path),
            },
        )


def _write_metadata(root: Path, videos: Path) -> None:
    root.mkdir()
    videos.mkdir()
    for split, (filename, signer) in _SPLITS.items():
        with (root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "label", "video_lb_id"])
            writer.writeheader()
            for label in range(3):
                copies = 3 - label if split == "train" else 1
                for copy in range(copies):
                    name = f"x_center_signer{signer}_center_ord1_{label}_{copy}_{split}.mp4"
                    writer.writerow({"name": name, "label": label, "video_lb_id": name[-20:]})
                    (videos / name).write_bytes(f"{split}-{label}-{copy}".encode())


def _write_pose_config(path: Path) -> None:
    extractor = {
        "name": "rtmpose_l_coco_wholebody_384x288",
        "framework": "mmpose",
        "variant": "rtmpose-l",
        "training_dataset": "coco-wholebody",
        "raw_layout": "coco_wholebody_133",
        "input_size_wh": [288, 384],
        "hash_source_video": True,
        "pose_model": {
            "config": str(path.parent / "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"),
            "checkpoint": str(path.parent / "pose.pth"),
            "checkpoint_url": "https://example.test/pose.pth",
        },
        "detector": {
            "config": str(path.parent / "rtmdet_m_coco-person.py"),
            "checkpoint": str(path.parent / "detector.pth"),
            "checkpoint_url": "https://example.test/detector.pth",
        },
    }
    path.write_text(yaml.safe_dump({"schema_version": 1, "extractor": extractor}), "utf-8")


def test_multi_vsl_manifest_feeds_pose_extraction_and_graph_preparation(tmp_path: Path) -> None:
    metadata, videos, prepared = tmp_path / "metadata", tmp_path / "videos", tmp_path / "prepared"
    _write_metadata(metadata, videos)

    required = tmp_path / "required.txt"
    common = ["--metadata-root", str(metadata), "--classes", "2"]
    assert prepare_multi_vsl_pose.main(["list-videos", *common, "--output", str(required)]) == 0
    assert len(required.read_text(encoding="utf-8").split()) == 9

    assert (
        prepare_multi_vsl_pose.main(
            ["build", *common, "--video-root", str(videos), "--output-root", str(prepared)]
        )
        == 0
    )
    manifest = prepared / "manifest.csv"
    records = read_manifest(manifest)
    assert len(records) == 9

    pose_config = tmp_path / "rtmpose.yaml"
    _write_pose_config(pose_config)
    raw_root = tmp_path / "raw"
    extract_args = [
        "extract",
        "--config",
        str(pose_config),
        "--manifest",
        str(manifest),
        "--dataset-root",
        str(videos),
        "--output-root",
        str(raw_root),
        "--report",
        str(tmp_path / "extract.json"),
        "--progress-every",
        "0",
    ]
    assert extract_pose.main(extract_args, extractor_factory=lambda _config: _FakeExtractor()) == 0
    assert all(pose_cache_path(raw_root, record.sample_id).is_file() for record in records)

    graph_config = yaml.safe_load(
        Path("configs/preprocessing/multi_vsl_graph.yaml").read_text(encoding="utf-8")
    )
    graph_config["expected"] = {
        "manifest_sha256": sha256_file(manifest),
        "extractor_fingerprint": _FakeExtractor.fingerprint,
        "clips": 9,
        "classes": 2,
        "splits": {"train": 5, "validation": 2, "test": 2},
    }
    graph_config_path = tmp_path / "graph.yaml"
    graph_config_path.write_text(yaml.safe_dump(graph_config), encoding="utf-8")
    graph_report = tmp_path / "graph.json"
    graph_args = [
        "--config",
        str(graph_config_path),
        "--manifest",
        str(manifest),
        "--pose-root",
        str(raw_root),
        "--output-root",
        str(tmp_path / "graph"),
        "--report",
        str(graph_report),
        "--progress-every",
        "0",
    ]
    assert prepare_graph.main(graph_args) == 0
    report = json.loads(graph_report.read_text(encoding="utf-8"))
    assert report["prepared"] == 9
    assert report["split_counts"] == {"train": 5, "validation": 2, "test": 2}
