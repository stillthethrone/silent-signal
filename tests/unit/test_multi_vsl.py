from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from silent_signal.data.multi_vsl import prepare_multi_vsl


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
