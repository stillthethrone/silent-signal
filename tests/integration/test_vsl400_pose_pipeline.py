from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from silent_signal.cli import extract_pose, prepare, prepare_graph, select_classes
from silent_signal.data.manifest import read_manifest
from silent_signal.pose.cache import pose_cache_path, sha256_file
from silent_signal.pose.interface import RawPoseSequence


class _FakeExtractor:
    fingerprint = "fake-rtmpose-v1"

    def extract_video(self, video_path: Path, *, sample_id: str) -> RawPoseSequence:
        frames = 5
        return RawPoseSequence(
            sample_id=sample_id,
            source_video=str(video_path),
            frame_indices=np.arange(frames, dtype=np.int64),
            timestamps_seconds=np.arange(frames, dtype=np.float64) / 25.0,
            frame_size_hw=(1080, 1080),
            keypoints_xy=np.random.default_rng(1)
            .uniform(100, 900, (frames, 133, 2))
            .astype(np.float32),
            keypoint_scores=np.full((frames, 133), 0.9, dtype=np.float32),
            bboxes_xyxy=np.tile(np.asarray([50, 50, 1000, 1070], dtype=np.float32), (frames, 1)),
            bbox_scores=np.ones((frames,), dtype=np.float32),
            person_detected=np.ones((frames,), dtype=np.bool_),
            metadata={
                "extractor_fingerprint": self.fingerprint,
                "video_sha256": sha256_file(video_path),
            },
        )


def _pose_config(path: Path) -> None:
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


def test_vsl400_subset_feeds_pose_extraction_and_graph_preparation(
    config_file: Path, vsl400_root: Path, tmp_path: Path
) -> None:
    # 1. Full-dataset manifest and signer-disjoint split (notebook 00 / 11 step 1).
    assert prepare.main(["all", "--config", str(config_file), "--level", "metadata"]) == 0
    full_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    full_manifest = Path(full_config["outputs"]["manifest_parquet"])

    # 2. Class subset that keeps the split.
    prepared = tmp_path / "prepared"
    assert (
        select_classes.main(
            ["--manifest", str(full_manifest), "--output-root", str(prepared), "--classes", "2"]
        )
        == 0
    )
    selection = json.loads((prepared / "selection.json").read_text(encoding="utf-8"))
    assert [item["gloss_name"] for item in selection["classes"]] == ["xin chào", "cảm ơn"]
    assert selection["views"] == {"front": 32, "left": 32, "right": 32}

    # 3. Copy only the required videos, keeping their relative paths.
    local_root = tmp_path / "local"
    for relative in (prepared / "required_videos.txt").read_text(encoding="utf-8").split():
        destination = local_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vsl400_root / relative, destination)

    # 4. Re-validate the subset against the local copy with subset-specific counts.
    runtime = dict(full_config)
    runtime["dataset"] = {**full_config["dataset"], "root": str(local_root)}
    runtime["expected"] = {
        **full_config["expected"],
        "clips": sum(selection["clips"].values()),
        "glosses": selection["class_count"],
    }
    runtime["outputs"] = {
        "manifest_csv": str(prepared / "manifest.csv"),
        "manifest_parquet": str(prepared / "manifest.parquet"),
        "labels": str(prepared / "labels.json"),
        "split": str(prepared / "unused_split.json"),
        "report": str(prepared / "validation_report.json"),
        "invalid_records": str(prepared / "invalid_records.csv"),
    }
    runtime_config = tmp_path / "subset.yaml"
    runtime_config.write_text(yaml.safe_dump(runtime, allow_unicode=True), encoding="utf-8")
    validate_args = ["validate", "--config", str(runtime_config), "--level", "metadata"]
    assert prepare.main([*validate_args, "--manifest", str(prepared / "manifest.parquet")]) == 0
    manifest = prepared / "manifest.csv"
    records = read_manifest(manifest)
    assert len(records) == 96 and all(record.is_valid for record in records)

    # 5. Pose extraction and graph preparation.
    pose_config = tmp_path / "rtmpose.yaml"
    _pose_config(pose_config)
    raw_root = tmp_path / "raw"
    extract_args = [
        "extract",
        "--config",
        str(pose_config),
        "--manifest",
        str(manifest),
        "--dataset-root",
        str(local_root),
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
        Path("configs/preprocessing/coco_wholebody_75_t64.yaml").read_text(encoding="utf-8")
    )
    graph_config["expected"] = {
        "manifest_sha256": sha256_file(manifest),
        "extractor_fingerprint": _FakeExtractor.fingerprint,
        "clips": 96,
        "classes": 2,
        "splits": selection["clips"],
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
    assert json.loads(graph_report.read_text(encoding="utf-8"))["prepared"] == 96
