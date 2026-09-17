from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_signal.cli.train_videomaev2_demo import _balanced_cap, _select_demo_rows


def _selection(path: Path) -> Path:
    payload = {
        "classes": [
            {
                "rank": index + 1,
                "subset_class_index": index,
                "gloss_name": f"WORD_{index:02d}",
            }
            for index in range(20)
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rows() -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"{split}-{class_index}",
            "video_path": f"videos/{split}-{class_index}.mp4",
            "gloss_name": f"WORD_{class_index:02d}",
            "class_index": str(class_index),
            "split": split,
        }
        for split in ("train", "validation", "test")
        for class_index in range(20)
    ]


def test_demo_selection_keeps_the_first_twenty_official_classes(tmp_path: Path) -> None:
    selected, classes = _select_demo_rows(_rows(), _selection(tmp_path / "selection.json"), 20)

    assert len(selected) == 60
    assert len(classes) == 20
    assert {row["split"] for row in selected} == {"train", "validation", "test"}
    assert {int(row["class_index"]) for row in selected} == set(range(20))


def test_demo_selection_rejects_cross_split_sample_leakage(tmp_path: Path) -> None:
    rows = _rows()
    rows[20]["sample_id"] = rows[0]["sample_id"]

    with pytest.raises(RuntimeError, match="Duplicate sample_id"):
        _select_demo_rows(rows, _selection(tmp_path / "selection.json"), 20)


def test_balanced_cap_represents_every_class_before_repeating() -> None:
    rows = [
        {"class_index": class_index, "sample_id": f"{class_index}-{copy}"}
        for class_index in range(20)
        for copy in range(3)
    ]

    selected = _balanced_cap(rows, 20, seed=42)

    assert len(selected) == 20
    assert {int(row["class_index"]) for row in selected} == set(range(20))
