from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_signal.cli.train_videomaev2_demo import (
    EarlyStoppingState,
    _balanced_cap,
    _select_demo_rows,
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


def _frozen_demo_selection(path: Path, class_count: int = 50) -> Path:
    payload = {
        "classes": [
            {
                "rank": index + 1,
                "class_index": index,
                "source_subset_class_index": index + 100,
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


def test_demo_selection_reuses_a_frozen_baseline_manifest(tmp_path: Path) -> None:
    rows = _rows()
    for row in rows:
        row["source_class_index"] = str(100 + int(row["class_index"]))

    selected, classes = _select_demo_rows(
        rows,
        _frozen_demo_selection(tmp_path / "selected_50_words.json"),
        50,
    )

    assert len(selected) == 150
    assert [int(item["source_manifest_class_index"]) for item in classes] == list(range(50))
    assert [int(item["source_subset_class_index"]) for item in classes] == list(range(100, 150))
    assert {int(row["source_class_index"]) for row in selected} == set(range(100, 150))
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


def test_early_stopping_requires_meaningful_improvement_and_minimum_epochs() -> None:
    state = EarlyStoppingState()

    assert state.update(1.0, epoch=1, min_delta=0.001)
    assert not state.update(0.9995, epoch=2, min_delta=0.001)
    assert state.epochs_without_improvement == 1
    assert state.update(0.998, epoch=3, min_delta=0.001)
    assert state.best_epoch == 3
    assert state.epochs_without_improvement == 0

    for epoch in range(4, 12):
        assert not state.update(0.998, epoch=epoch, min_delta=0.001)
    assert not state.should_stop(completed_epochs=9, min_epochs=10, patience=8)
    assert state.should_stop(completed_epochs=11, min_epochs=10, patience=8)


def test_early_stopping_state_round_trips_for_resume() -> None:
    initial = EarlyStoppingState.from_dict(EarlyStoppingState().to_dict())
    assert initial == EarlyStoppingState()

    trained = EarlyStoppingState()
    trained.update(2.5, epoch=1, min_delta=0.001)
    trained.update(2.5, epoch=2, min_delta=0.001)
    assert EarlyStoppingState.from_dict(trained.to_dict()) == trained


def test_regularization_defaults_are_exposed_by_the_cli() -> None:
    args = build_parser().parse_args(
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

    assert args.label_smoothing == 0.05
    assert args.early_stopping_min_epochs == 10
    assert args.early_stopping_patience == 8
    assert args.early_stopping_min_delta == 0.001
    assert args.lr_plateau_patience == 3
    assert args.lr_plateau_factor == 0.5
    assert args.minimum_learning_rate == 1e-6


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
    assert "ReduceLROnPlateau(" in source
    assert "CrossEntropyLoss(label_smoothing=args.label_smoothing)" in source
    assert "criterion=evaluation_criterion" in source
    assert '"early_stopping": early_stopping.to_dict()' in source
    assert '"scheduler_state": scheduler.state_dict()' in source
    assert 'manifest_root / f"{split}.csv"' in source
    assert 'default=0,\n        help="Zero uses every official training clip."' in source
