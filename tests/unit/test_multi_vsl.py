from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from silent_signal.data.manifest import read_manifest
from silent_signal.data.multi_vsl import (
    build_multi_vsl_pose_dataset,
    multi_vsl_pose_records,
    prepare_multi_vsl,
    select_multi_vsl,
)


def _write_split(root: Path, split: str, signer: int, class_count: int = 4) -> None:
    names = {
        "train": "train_1_200_center_ord1.csv",
        "validation": "val_1_200_center_ord1.csv",
        "test": "test_1_200_center_ord1.csv",
    }
    path = root / names[split]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "label", "video_lb_id"])
        writer.writeheader()
        for class_index in range(class_count):
            copies = class_count - class_index if split == "train" else 1
            for copy in range(copies):
                name = f"sample_signer{signer:02d}_center_ord1_{class_index}_{copy}_{split}.mp4"
                writer.writerow(
                    {
                        "name": name,
                        "label": class_index,
                        "video_lb_id": f"{split}-{class_index}-{copy}",
                    }
                )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "metadata"
    videos = tmp_path / "videos"
    metadata.mkdir()
    videos.mkdir()
    for split, signer in (("train", 1), ("validation", 2), ("test", 3)):
        _write_split(metadata, split, signer)
    for csv_path in metadata.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                (videos / row["name"]).touch()
    return metadata, videos


def test_prepare_multi_vsl_preserves_official_signer_splits(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)

    result = prepare_multi_vsl(
        metadata_root=metadata,
        video_root=videos,
        output_root=tmp_path / "output",
        class_count=3,
    )

    assert result.selected_classes == (0, 1, 2)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["signer_isolation"] == "passed"
    assert summary["signers"] == {"train": ["01"], "validation": ["02"], "test": ["03"]}
    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert "training clip count only" in selection["selection"]
    assert "never re-split" in selection["split_policy"]


def test_prepare_multi_vsl_requires_every_selected_video(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)
    next(videos.glob("*train.mp4")).unlink()

    with pytest.raises(FileNotFoundError, match="official videos"):
        prepare_multi_vsl(
            metadata_root=metadata,
            video_root=videos,
            output_root=tmp_path / "output",
            class_count=3,
        )


def test_prepare_multi_vsl_rejects_signer_leakage(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)
    validation = metadata / "val_1_200_center_ord1.csv"
    text = validation.read_text(encoding="utf-8").replace("signer02", "signer01")
    validation.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Signer leakage"):
        prepare_multi_vsl(
            metadata_root=metadata,
            video_root=videos,
            output_root=tmp_path / "output",
            class_count=3,
        )


def test_pose_records_share_rgb_sample_ids_and_rank_order(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)
    rgb = prepare_multi_vsl(
        metadata_root=metadata, video_root=videos, output_root=tmp_path / "rgb", class_count=3
    )
    with rgb.manifest_path.open(encoding="utf-8", newline="") as handle:
        rgb_ids = {row["sample_id"] for row in csv.DictReader(handle)}

    selection = select_multi_vsl(metadata, class_count=3)
    records = multi_vsl_pose_records(selection)

    assert selection.classes == (0, 1, 2)
    assert {record.sample_id for record in records} == rgb_ids
    assert {(r.gloss_id, r.class_index) for r in records} == {("0", 0), ("1", 1), ("2", 2)}
    assert {record.view for record in records} == {"center"}
    assert len(selection.required_videos) == len(records)
    assert selection.source_files["train"]["path"] == "train_1_200_center_ord1.csv"


def test_select_multi_vsl_zero_keeps_every_eligible_class(tmp_path: Path) -> None:
    metadata, _videos = _fixture(tmp_path)

    assert select_multi_vsl(metadata, class_count=0).classes == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="at least 2"):
        select_multi_vsl(metadata, class_count=1)


def test_build_pose_dataset_writes_official_split_artifacts(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)
    for video in videos.iterdir():
        video.write_bytes(b"video")

    dataset = build_multi_vsl_pose_dataset(
        metadata_root=metadata, video_root=videos, output_root=tmp_path / "pose", class_count=3
    )

    assert not dataset.validation.has_errors
    records = read_manifest(dataset.manifest_csv)
    assert read_manifest(dataset.manifest_parquet) == records
    assert {record.split for record in records} == {"train", "validation", "test"}
    split = json.loads(dataset.split_path.read_text(encoding="utf-8"))
    assert split["strategy"] == "official"
    assert split["signer_ids"] == {"train": ["01"], "validation": ["02"], "test": ["03"]}
    assert split["clip_counts"] == {"train": 9, "validation": 3, "test": 3}
    selection = json.loads(dataset.selection_path.read_text(encoding="utf-8"))
    assert [item["source_label"] for item in selection["classes"]] == [0, 1, 2]
    labels = json.loads(dataset.labels_path.read_text(encoding="utf-8"))
    assert labels["class_index_to_gloss"] == {"0": "VSL_001", "1": "VSL_002", "2": "VSL_003"}


def test_build_pose_dataset_marks_missing_and_empty_videos_invalid(tmp_path: Path) -> None:
    metadata, videos = _fixture(tmp_path)
    files = sorted(videos.iterdir())
    for video in files[1:]:
        video.write_bytes(b"video")
    missing = next(path for path in files[1:] if "_0_0_" in path.name)
    missing.unlink()

    dataset = build_multi_vsl_pose_dataset(
        metadata_root=metadata, video_root=videos, output_root=tmp_path / "pose", class_count=0
    )

    report = dataset.validation.to_report()
    assert report["passed"] is False
    assert report["issue_counts"]["by_code"] == {"video_empty": 1, "video_missing": 1}
