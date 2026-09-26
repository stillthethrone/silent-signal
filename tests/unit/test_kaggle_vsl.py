from __future__ import annotations

from pathlib import Path

import numpy as np

from silent_signal.data.kaggle_vsl import build_kaggle_vsl_manifest, selection_payload


def _write_class(root: Path, split: str, gloss: str, count: int) -> None:
    destination = root / split / gloss
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        np.save(destination / f"{index:06d}.npy", np.ones((4, 76, 3), dtype=np.float32))


def test_kaggle_vsl_build_keeps_test_and_makes_deterministic_validation(tmp_path: Path) -> None:
    _write_class(tmp_path, "train", "Xin chào", 8)
    _write_class(tmp_path, "test", "Xin chào", 2)
    _write_class(tmp_path, "train", "Cảm ơn", 6)
    _write_class(tmp_path, "test", "Cảm ơn", 2)
    _write_class(tmp_path, "train", "Không đủ", 2)
    _write_class(tmp_path, "test", "Không đủ", 1)

    first = build_kaggle_vsl_manifest(
        tmp_path,
        classes=2,
        min_official_train_samples=4,
        validation_fraction=0.25,
        seed=42,
    )
    second = build_kaggle_vsl_manifest(
        tmp_path,
        classes=2,
        min_official_train_samples=4,
        validation_fraction=0.25,
        seed=42,
    )

    assert [item.gloss_name for item in first.labels] == ["Xin chào", "Cảm ơn"]
    assert first.split_counts == {"test": 4, "train": 10, "validation": 4}
    assert [(item.sample_id, item.split) for item in first.records] == [
        (item.sample_id, item.split) for item in second.records
    ]
    assert {item.signer_id for item in first.records} == {"unknown"}


def test_kaggle_vsl_zero_classes_selects_every_eligible_gloss_and_reports_strategy(
    tmp_path: Path,
) -> None:
    _write_class(tmp_path, "train", "A", 4)
    _write_class(tmp_path, "test", "A", 1)
    _write_class(tmp_path, "train", "B", 3)
    _write_class(tmp_path, "test", "B", 1)
    _write_class(tmp_path, "train", "No test", 4)

    result = build_kaggle_vsl_manifest(
        tmp_path,
        classes=0,
        min_official_train_samples=2,
        validation_fraction=0.25,
        seed=42,
    )
    payload = selection_payload(
        result,
        dataset_handle="owner/dataset",
        min_official_train_samples=2,
        validation_fraction=0.25,
        seed=42,
        requested_classes=0,
    )

    assert [item.gloss_name for item in result.labels] == ["A", "B"]
    assert payload["selection_strategy"] == (
        "all_eligible_glosses_descending_official_train_count_then_gloss"
    )
    assert payload["requested_classes"] == 0
