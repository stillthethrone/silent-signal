from __future__ import annotations

from pathlib import Path

import numpy as np

from silent_signal.cli.prepare_kaggle_vsl_mediapipe import main
from silent_signal.data.keypoint_pack import read_packed_keypoints
from silent_signal.data.manifest import read_manifest
from silent_signal.pose.cache import pose_cache_path
from silent_signal.preprocessing.cache import read_graph_pose_cache


def _write_class(root: Path, split: str, gloss: str, count: int) -> None:
    destination = root / split / gloss
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        values = np.ones((5, 76, 3), dtype=np.float32)
        values[:, :75, 0] *= np.linspace(0.1, 0.9, 75)
        values[:, :75, 1] *= np.linspace(0.2, 0.8, 75)
        np.save(destination / f"{index:06d}.npy", values)


def test_cli_build_and_convert_are_resumable(tmp_path: Path) -> None:
    keypoints = tmp_path / "keypoints"
    _write_class(keypoints, "train", "Xin chào", 5)
    _write_class(keypoints, "test", "Xin chào", 2)
    manifest = tmp_path / "manifest.csv"
    labels = tmp_path / "labels.json"
    selection = tmp_path / "selection.json"
    build_report = tmp_path / "build-report.json"

    assert (
        main(
            [
                "build",
                "--keypoint-root",
                str(keypoints),
                "--manifest",
                str(manifest),
                "--labels",
                str(labels),
                "--selection",
                str(selection),
                "--report",
                str(build_report),
                "--classes",
                "1",
                "--min-official-train-samples",
                "4",
                "--validation-fraction",
                "0.2",
            ]
        )
        == 0
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        """schema_version: 1
preprocessing:
  layout_name: mediapipe_holistic_75_v1
  target_frames: 8
  interpolation_max_gap: 1
  zero_epsilon: 1.0e-8
expected: {}
""",
        encoding="utf-8",
    )
    graph_root = tmp_path / "graph"
    convert_report = tmp_path / "convert-report.json"
    arguments = [
        "convert",
        "--keypoint-root",
        str(keypoints),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
        "--output-root",
        str(graph_root),
        "--report",
        str(convert_report),
        "--progress-every",
        "0",
    ]
    assert main(arguments) == 0
    assert main(arguments) == 0

    records = read_manifest(manifest)
    sample = read_graph_pose_cache(pose_cache_path(graph_root, records[0].sample_id))
    assert sample.features.shape == (8, 75, 7)

    packed_path = tmp_path / "keypoints.npz"
    pack_report = tmp_path / "pack-report.json"
    pack_arguments = [
        "pack",
        "--keypoint-root",
        str(keypoints),
        "--manifest",
        str(manifest),
        "--output",
        str(packed_path),
        "--report",
        str(pack_report),
        "--progress-every",
        "0",
    ]
    assert main(pack_arguments) == 0
    assert main(pack_arguments) == 0
    packed = read_packed_keypoints(packed_path)
    assert len(packed.sample_ids) == len(records)
    assert packed.sequence(0).shape == (5, 76, 3)
    assert packed.metadata["model_layout"] == "mediapipe_upper68_v1"

    assert main([*pack_arguments, "--dataset-handle", "different/dataset"]) == 2

    changed = keypoints / records[0].video_path
    values = np.load(changed, allow_pickle=False)
    values[0, 0, 0] += 1.0
    np.save(changed, values)
    assert main(pack_arguments) == 2  # stale packed data must never be reused silently
