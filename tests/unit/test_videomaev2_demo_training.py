from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_signal.cli.train_videomaev2_demo import (
    _balanced_cap,
    _select_demo_rows,
    _trailing_non_improving_epochs,
    build_parser,
)

_ROOT = Path(__file__).parents[2]
_TRAINER = _ROOT / "src/silent_signal/cli/train_videomaev2_demo.py"


def _selection(path: Path, class_count: int = 50) -> Path:
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


def _rows(class_count: int = 50) -> list[dict[str, str]]:
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


def test_demo_selection_keeps_the_first_fifty_official_classes(tmp_path: Path) -> None:
    selected, classes = _select_demo_rows(_rows(), _selection(tmp_path / "selection.json"), 50)

    assert len(selected) == 150
    assert len(classes) == 50
    assert {row["split"] for row in selected} == {"train", "validation", "test"}
    assert {int(row["class_index"]) for row in selected} == set(range(50))


def test_demo_selection_rejects_cross_split_sample_leakage(tmp_path: Path) -> None:
    rows = _rows()
    rows[50]["sample_id"] = rows[0]["sample_id"]

    with pytest.raises(RuntimeError, match="Duplicate sample_id"):
        _select_demo_rows(rows, _selection(tmp_path / "selection.json"), 50)


def test_demo_selection_skips_ranked_class_missing_an_official_split(tmp_path: Path) -> None:
    rows = [row for row in _rows(51) if not (row["class_index"] == "0" and row["split"] == "test")]

    selected, classes = _select_demo_rows(rows, _selection(tmp_path / "selection.json", 51), 50)

    assert {int(item["source_subset_class_index"]) for item in classes} == set(range(1, 51))
    assert {int(row["class_index"]) for row in selected} == set(range(50))
    assert all(int(row["source_class_index"]) >= 1 for row in selected)


def test_balanced_cap_represents_every_class_before_repeating() -> None:
    rows = [
        {"class_index": class_index, "sample_id": f"{class_index}-{copy}"}
        for class_index in range(50)
        for copy in range(3)
    ]

    selected = _balanced_cap(rows, 50, seed=42)

    assert len(selected) == 50
    assert {int(row["class_index"]) for row in selected} == set(range(50))


def test_early_stopping_counts_only_epochs_after_the_latest_improvement() -> None:
    history = [
        {"validation_loss": 3.0},
        {"validation_loss": 2.0},
        {"validation_loss": 2.1},
        {"validation_loss": 2.2},
    ]

    assert _trailing_non_improving_epochs(history, min_delta=0.0) == 2


def test_early_stopping_min_delta_rejects_tiny_improvements() -> None:
    history = [
        {"validation_loss": 2.0},
        {"validation_loss": 1.9995},
        {"validation_loss": 1.9990},
    ]

    assert _trailing_non_improving_epochs(history, min_delta=0.001) == 2


def test_regularized_baseline_defaults_are_conservative() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--selection-report",
            "selection.json",
            "--dataset-root",
            "dataset",
            "--output-root",
            "output",
        ]
    )

    assert args.rgb_dropout == 0.3
    assert args.label_smoothing == 0.1
    assert args.weight_decay == 0.01
    assert args.random_crop_scale_min == 0.85
    assert args.color_jitter == 0.1
    assert args.gradient_clip_norm == 1.0


def test_trainer_freezes_videomae_and_trains_a_separate_rgb_transformer() -> None:
    source = _TRAINER.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("USE_TF", "0")' in source
    assert 'os.environ.setdefault("USE_FLAX", "0")' in source
    assert "parameter.requires_grad = False" in source
    assert "visual.patch_embed(pixel_values)" in source
    assert "for block in visual.blocks" in source
    assert "tokens.mean(dim=2)" in source
    assert "torch.nn.TransformerEncoder(" in source
    assert "self.rgb_transformer(tokens)" in source
    assert "self.classifier(feature)" in source
    assert "CrossEntropyLoss(label_smoothing=args.label_smoothing)" in source
    assert "clip_grad_norm_" in source
    assert "random_crop_scale_min" in source
    assert "color_jitter" in source
    assert 'manifest_root / f"{split}.csv"' in source
    assert 'default=0,\n        help="Zero uses every official training clip."' in source
