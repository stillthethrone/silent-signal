from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_signal.cli.train_videomaev2_demo import _balanced_cap, _select_demo_rows

_ROOT = Path(__file__).parents[2]
_TRAINER = _ROOT / "src/silent_signal/cli/train_videomaev2_demo.py"


def _selection(path: Path, class_count: int = 30) -> Path:
    payload = {
        "classes": [
            {
                "rank": index + 1,
                "subset_class_index": index,
                "gloss_name": f"WORD_{index:02d}",
            }
            for index in range(class_count)
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rows(class_count: int = 30) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"{split}-{class_index}",
            "video_path": f"videos/{split}-{class_index}.mp4",
            "gloss_name": f"WORD_{class_index:02d}",
            "class_index": str(class_index),
            "split": split,
        }
        for split in ("train", "validation", "test")
        for class_index in range(class_count)
    ]


def test_demo_selection_keeps_the_first_thirty_official_classes(tmp_path: Path) -> None:
    selected, classes = _select_demo_rows(_rows(), _selection(tmp_path / "selection.json"), 30)

    assert len(selected) == 90
    assert len(classes) == 30
    assert {row["split"] for row in selected} == {"train", "validation", "test"}
    assert {int(row["class_index"]) for row in selected} == set(range(30))


def test_demo_selection_rejects_cross_split_sample_leakage(tmp_path: Path) -> None:
    rows = _rows()
    rows[30]["sample_id"] = rows[0]["sample_id"]

    with pytest.raises(RuntimeError, match="Duplicate sample_id"):
        _select_demo_rows(rows, _selection(tmp_path / "selection.json"), 30)


def test_demo_selection_skips_ranked_class_missing_an_official_split(tmp_path: Path) -> None:
    rows = [row for row in _rows(31) if not (row["class_index"] == "0" and row["split"] == "test")]

    selected, classes = _select_demo_rows(rows, _selection(tmp_path / "selection.json", 31), 30)

    assert {int(item["source_subset_class_index"]) for item in classes} == set(range(1, 31))
    assert {int(row["class_index"]) for row in selected} == set(range(30))
    assert all(int(row["source_class_index"]) >= 1 for row in selected)


def test_balanced_cap_represents_every_class_before_repeating() -> None:
    rows = [
        {"class_index": class_index, "sample_id": f"{class_index}-{copy}"}
        for class_index in range(30)
        for copy in range(3)
    ]

    selected = _balanced_cap(rows, 30, seed=42)

    assert len(selected) == 30
    assert {int(row["class_index"]) for row in selected} == set(range(30))


def test_trainer_freezes_videomae_and_trains_a_separate_rgb_transformer() -> None:
    source = _TRAINER.read_text(encoding="utf-8")

    assert "parameter.requires_grad = False" in source
    assert "visual.patch_embed(pixel_values)" in source
    assert "for block in visual.blocks" in source
    assert "tokens.mean(dim=2)" in source
    assert "torch.nn.TransformerEncoder(" in source
    assert "self.rgb_transformer(tokens)" in source
    assert "self.classifier(feature)" in source
    assert 'manifest_root / f"{split}.csv"' in source
    assert 'default=0,\n        help="Zero uses every official training clip."' in source
