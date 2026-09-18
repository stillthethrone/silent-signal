from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from silent_signal.cli.analyze_videomaev2_demo import main

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")
pytest.importorskip("seaborn")
pytest.importorskip("sklearn")


def test_analysis_cli_completes_for_all_fifty_classes(tmp_path: Path) -> None:
    classes = [
        {
            "class_index": index,
            "gloss_name": f"WORD_{index:02d}",
            "counts": {"train": 10, "validation": 4, "test": 4},
        }
        for index in range(50)
    ]
    (tmp_path / "selected_50_words.json").write_text(
        json.dumps({"classes": classes}), encoding="utf-8"
    )
    history = [
        {
            "epoch": 1,
            "train_loss": 3.0,
            "train_top1": 0.2,
            "validation_loss": 3.1,
            "validation_top1": 0.18,
        },
        {
            "epoch": 2,
            "train_loss": 2.0,
            "train_top1": 0.7,
            "validation_loss": 2.5,
            "validation_top1": 0.5,
        },
    ]
    report = {
        "best_epoch": 2,
        "history": history,
        "evaluation": {"validation": {"available_samples": 200, "partial": False}},
    }
    (tmp_path / "baseline_report.json").write_text(json.dumps(report), encoding="utf-8")
    prediction_path = tmp_path / "validation_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "true_class",
                "pred_class",
                "true_gloss",
                "pred_gloss",
                "confidence",
                "split",
            ),
        )
        writer.writeheader()
        for class_index in range(50):
            for copy in range(4):
                predicted = class_index if copy < 2 else (class_index + 1) % 50
                writer.writerow(
                    {
                        "sample_id": f"{class_index}-{copy}",
                        "true_class": class_index,
                        "pred_class": predicted,
                        "true_gloss": f"WORD_{class_index:02d}",
                        "pred_gloss": f"WORD_{predicted:02d}",
                        "confidence": 0.9 if copy == 3 else 0.6,
                        "split": "validation",
                    }
                )

    assert (
        main(
            [
                "--baseline-root",
                str(tmp_path),
                "--split",
                "validation",
                "--top-errors",
                "15",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "validation_error_analysis.json").read_text(encoding="utf-8"))
    assert payload["top1_accuracy"] == 0.5
    assert payload["generalization"]["best_checkpoint"]["epoch"] == 2
    assert payload["class_support"]["analysis_classes_below_5"] == 50
