from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from silent_signal.contracts import ManifestRecord
from silent_signal.data.keypoint_pack import write_packed_keypoints
from silent_signal.data.manifest import write_manifest
from silent_signal.pose.cache import sha256_file
from silent_signal.preprocessing.mediapipe_features import MediaPipeFeatureConfig
from silent_signal.training import rgb_pose_fusion_trainer
from silent_signal.training.rgb_pose_fusion_trainer import (
    FusionTrainingConfig,
    load_fusion_data,
    train_rgb_pose_fusion,
)


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path, SimpleNamespace]:
    rng = np.random.default_rng(7)
    records: list[ManifestRecord] = []
    sequences: dict[str, np.ndarray] = {}
    rgb_by_id: dict[str, np.ndarray] = {}
    for label in range(3):
        for split, count in (("train", 6), ("validation", 2), ("test", 2)):
            for repetition in range(count):
                sample_id = f"{split}-{label}-{repetition}"
                raw = rng.normal(0, 0.03, size=(18, 76, 3)).astype(np.float32)
                raw[:, 54:, 0] += label * 0.4
                raw[:, 25:33] = 0
                records.append(
                    ManifestRecord(
                        sample_id=sample_id,
                        instance_id=sample_id,
                        video_id=sample_id,
                        signer_id=f"s-{split}",
                        gloss_id=f"g{label}",
                        gloss_name=f"từ {label}",
                        class_index=label,
                        view="front",
                        video_path=f"{split}/từ {label}/{repetition}.npy",
                        split=split,
                    )
                )
                sequences[sample_id] = raw
                feature = rng.normal(0, 0.02, size=(4, 8)).astype(np.float32)
                feature[:, label] += 1.0
                rgb_by_id[sample_id] = feature
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    keypoints = tmp_path / "pose.npz"
    write_packed_keypoints(keypoints, sequences, {"format": "test"})
    rgb_path = tmp_path / "rgb.npz"
    rgb_path.write_bytes(b"fingerprint-only; reader is replaced in this test")
    # Reverse order to prove joining is by sample_id, not by incidental array order.
    sample_ids = tuple(reversed(tuple(rgb_by_id)))
    features = np.stack([rgb_by_id[sample_id] for sample_id in sample_ids])
    packed = SimpleNamespace(
        sample_ids=sample_ids,
        features=features,
        metadata={
            "extraction_fingerprint": "synthetic-fingerprint",
            "manifest_sha256": sha256_file(manifest),
            "model_id": "synthetic",
            "model_revision": "test-revision",
            "preprocessing": {"frames": 4},
            "kind": "videomae_temporal_token_pack",
            "samples": len(sample_ids),
            "feature_shape": [4, 8],
        },
        index={sample_id: index for index, sample_id in enumerate(sample_ids)},
    )
    return manifest, keypoints, rgb_path, packed


def _training() -> FusionTrainingConfig:
    return FusionTrainingConfig(
        epochs=3,
        batch_size=6,
        learning_rate=0.003,
        warmup_epochs=1,
        patience=2,
        min_delta=0.0,
        workers=0,
        features=MediaPipeFeatureConfig(target_frames=8),
    )


def _pose() -> dict[str, int | float]:
    return {
        "graph_dim": 16,
        "graph_blocks": 1,
        "spatial_layers": 1,
        "spatial_heads": 4,
        "temporal_dim": 24,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "dropout": 0.0,
    }


def _fusion() -> dict[str, int | float]:
    return {
        "fusion_dim": 16,
        "rgb_layers": 1,
        "rgb_heads": 4,
        "dropout": 0.0,
        "modality_dropout": 0.0,
    }


def test_fusion_training_writes_metrics_and_resumes(tmp_path: Path, monkeypatch) -> None:
    manifest, keypoints, rgb_path, packed = _dataset(tmp_path)
    monkeypatch.setattr(rgb_pose_fusion_trainer, "read_rgb_feature_pack", lambda _path: packed)
    output = tmp_path / "run"
    messages: list[str] = []
    report = train_rgb_pose_fusion(
        manifest_path=manifest,
        keypoints_path=keypoints,
        rgb_pack_path=rgb_path,
        output_root=output,
        training=_training(),
        pose_overrides=_pose(),
        fusion_overrides=_fusion(),
        device_name="cpu",
        run_test=True,
        progress_every_batches=1,
        log=messages.append,
    )

    assert report["epochs_completed"] == 3 and report["test_evaluated"]
    assert report["data"]["clips"] == {"train": 18, "validation": 6, "test": 6}
    assert report["evaluation"]["test"]["samples"] == 6
    assert any("[train   1/3] batch" in message for message in messages)
    assert any("[test] evaluating best checkpoint" in message for message in messages)
    for name in (
        "config.json",
        "history.json",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "report.json",
        "predictions_validation.csv",
        "predictions_test.csv",
        "per_class_validation.csv",
        "confusion_test.csv",
    ):
        assert (output / name).is_file(), name

    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    resumed = train_rgb_pose_fusion(
        manifest_path=manifest,
        keypoints_path=keypoints,
        rgb_pack_path=rgb_path,
        output_root=output,
        training=_training(),
        pose_overrides=_pose(),
        fusion_overrides=_fusion(),
        device_name="cpu",
        resume=True,
    )
    assert resumed["epochs_completed"] == len(history)


def test_fusion_loader_rejects_missing_rgb_sample(tmp_path: Path, monkeypatch) -> None:
    manifest, keypoints, rgb_path, packed = _dataset(tmp_path)
    packed.sample_ids = packed.sample_ids[1:]
    packed.features = packed.features[1:]
    packed.index = {sample_id: index for index, sample_id in enumerate(packed.sample_ids)}
    monkeypatch.setattr(rgb_pose_fusion_trainer, "read_rgb_feature_pack", lambda _path: packed)

    with pytest.raises(ValueError, match="sample IDs differ"):
        load_fusion_data(manifest, keypoints, rgb_path, view="front")


def test_fusion_loader_rejects_rgb_pack_from_another_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, keypoints, rgb_path, packed = _dataset(tmp_path)
    packed.metadata["manifest_sha256"] = "wrong"
    monkeypatch.setattr(rgb_pose_fusion_trainer, "read_rgb_feature_pack", lambda _path: packed)

    with pytest.raises(ValueError, match="different manifest"):
        load_fusion_data(manifest, keypoints, rgb_path, view="front")
