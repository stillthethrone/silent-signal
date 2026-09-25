from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from silent_signal.cli import train_pose_transformer
from silent_signal.contracts import ManifestRecord
from silent_signal.data.keypoint_pack import read_packed_keypoints, write_packed_keypoints
from silent_signal.data.manifest import write_manifest

_SPLIT_OF_SIGNER = {
    "001": "train",
    "002": "train",
    "003": "train",
    "004": "validation",
    "005": "test",
}


def _dataset(tmp_path: Path, classes: int = 3) -> tuple[Path, Path]:
    """Three glosses whose right hand moves in different directions, five signers."""

    rng = np.random.default_rng(0)
    records, sequences = [], {}
    for signer, split in _SPLIT_OF_SIGNER.items():
        for label in range(classes):
            for repetition in range(4):
                video_id = f"{signer}{label}{repetition}"
                frames = int(rng.integers(20, 40))
                raw = rng.normal(0.0, 0.05, size=(frames, 76, 3)).astype(np.float32)
                direction = np.array(
                    [np.cos(label * 2.1), np.sin(label * 2.1), 0.0], dtype=np.float32
                )
                raw[:, 34:76:2] += (
                    np.linspace(0, 0.4, frames, dtype=np.float32)[:, None, None] * direction
                )
                raw[:, 25:33] = 0.0
                for view in ("front", "left"):
                    sample_id = f"{video_id}_{view}"
                    records.append(
                        ManifestRecord(
                            sample_id=sample_id,
                            instance_id=video_id,
                            video_id=video_id,
                            signer_id=signer,
                            gloss_id=f"g{label}",
                            gloss_name=f"từ {label}",
                            class_index=label,
                            view=view,
                            video_path=f"{view}_view/{video_id}.mp4",
                            split=split,
                        )
                    )
                sequences[f"{video_id}_front"] = raw
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    keypoints = tmp_path / "keypoints.npz"
    write_packed_keypoints(keypoints, sequences, {"format": "vsl_mediapipe_holistic_76"})
    return manifest, keypoints


def _args(manifest: Path, keypoints: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--keypoints",
        str(keypoints),
        "--output-root",
        str(output),
        "--target-frames",
        "16",
        "--batch-size",
        "8",
        "--learning-rate",
        "0.003",
        "--warmup-epochs",
        "1",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--graph-dim",
        "16",
        "--temporal-dim",
        "32",
        "--dropout",
        "0.0",
        *extra,
    ]


def test_training_writes_reports_and_resumes(tmp_path: Path) -> None:
    manifest, keypoints = _dataset(tmp_path)
    output = tmp_path / "run"
    assert (
        train_pose_transformer.main(
            _args(manifest, keypoints, output, "--epochs", "6", "--run-test")
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["data"]["clips"] == {"train": 36, "validation": 12, "test": 12}
    assert report["data"]["dropped_without_keypoints"] == 0
    assert report["data"]["class_names"] == ["từ 0", "từ 1", "từ 2"]
    assert report["epochs_completed"] == 6 and report["test_evaluated"]
    assert report["evaluation"]["validation"]["top1_accuracy"] >= 0.5
    for name in (
        "config.json",
        "history.json",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "predictions_validation.csv",
        "predictions_test.csv",
        "per_class_validation.csv",
        "confusion_test.csv",
    ):
        assert (output / name).is_file(), name
    with (output / "predictions_test.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 and {row["signer_id"] for row in rows} == {"005"}
    assert len(rows[0]["top5"].split("|")) == 3

    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    assert (
        train_pose_transformer.main(
            _args(manifest, keypoints, output, "--epochs", "6", "--run-test")
        )
        == 0
    )
    assert json.loads((output / "history.json").read_text(encoding="utf-8")) == history


def test_resume_refuses_different_settings_and_early_stopping_stops(tmp_path: Path, capsys) -> None:
    manifest, keypoints = _dataset(tmp_path)
    output = tmp_path / "run"
    assert train_pose_transformer.main(_args(manifest, keypoints, output, "--epochs", "2")) == 0
    original_config = (output / "config.json").read_text(encoding="utf-8")
    assert (
        train_pose_transformer.main(
            _args(manifest, keypoints, output, "--epochs", "2", "--seed", "7")
        )
        == 2
    )
    assert "different data or settings" in capsys.readouterr().err
    assert (output / "config.json").read_text(encoding="utf-8") == original_config

    stopped = tmp_path / "stopped"
    extra = ("--epochs", "30", "--patience", "1", "--min-delta", "100")
    assert train_pose_transformer.main(_args(manifest, keypoints, stopped, *extra)) == 0
    report = json.loads((stopped / "report.json").read_text(encoding="utf-8"))
    assert report["stopped_early"] and report["epochs_completed"] == 2


def test_training_refuses_incomplete_keypoint_pack(tmp_path: Path, capsys) -> None:
    manifest, keypoints = _dataset(tmp_path)
    packed = read_packed_keypoints(keypoints)
    sequences = {
        sample_id: packed.sequence(index)
        for index, sample_id in enumerate(packed.sample_ids)
        if sample_id != "00220_front"
    }
    write_packed_keypoints(keypoints, sequences, packed.metadata)

    assert train_pose_transformer.main(_args(manifest, keypoints, tmp_path / "run")) == 2
    assert "missing 1 manifest samples" in capsys.readouterr().err
