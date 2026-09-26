from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_signal.evaluation.training_curves import summarize_history


def _history(train_loss, validation_loss, train_top1, validation_top1, legacy=False):
    rows = []
    for epoch, values in enumerate(
        zip(train_loss, validation_loss, train_top1, validation_top1, strict=True), start=1
    ):
        row = {
            "epoch": epoch,
            "train_loss": values[0],
            "validation_loss": values[1],
            "train_top1": values[2],
            "validation_top1": values[3],
            "validation_top5": min(1.0, values[3] + 0.2),
            "validation_macro_f1": values[3] - 0.02,
            "learning_rate": 1e-3 / epoch,
        }
        if not legacy:
            row |= {"train_top5": min(1.0, values[2] + 0.1), "train_macro_f1": values[2] - 0.02}
        rows.append(row)
    return rows


def _report(epochs: int, stopped_early: bool) -> dict:
    return {
        "fingerprint": {"training": {"epochs": epochs}},
        "stopped_early": stopped_early,
        "data": {"classes": 70},
    }


def test_detects_overfitting_and_a_large_gap() -> None:
    history = _history(
        train_loss=[3.0, 2.0, 1.2, 0.8, 0.5, 0.3, 0.2],
        validation_loss=[3.1, 2.4, 2.0, 2.1, 2.3, 2.5, 2.7],
        train_top1=[0.1, 0.4, 0.6, 0.75, 0.85, 0.92, 0.96],
        validation_top1=[0.1, 0.3, 0.42, 0.42, 0.41, 0.42, 0.41],
    )
    summary = summarize_history(history, report=_report(7, stopped_early=True))

    assert summary["best_epoch"] == 3 and summary["epochs_since_best"] == 4
    assert summary["findings"] == ["overfitting"]
    assert summary["validation_loss_rise_since_best"] == pytest.approx(0.35)
    assert summary["chance_top1"] == pytest.approx(1 / 70)

    wide = _history([2, 1, 0.5], [2.2, 1.6, 1.5], [0.4, 0.8, 0.95], [0.3, 0.5, 0.55])
    assert "large_generalization_gap" in summarize_history(wide)["findings"]


def test_detects_underfitting_and_a_run_that_was_still_improving() -> None:
    history = _history([4.0, 3.8, 3.6], [4.1, 3.9, 3.7], [0.05, 0.1, 0.15], [0.04, 0.09, 0.13])
    summary = summarize_history(history, report=_report(3, stopped_early=False))

    assert summary["findings"] == ["underfitting", "still_improving"]
    assert summary["best"]["train_top5"] == pytest.approx(0.25)


def test_noisy_validation_and_legacy_histories() -> None:
    noisy = _history(
        [1.0] * 6, [1.5, 1.4, 1.45, 1.35, 1.42, 1.3], [0.9] * 6, [0.7, 0.5, 0.72, 0.52, 0.74, 0.55]
    )
    assert "noisy_validation" in summarize_history(noisy)["findings"]

    calm = _history(
        [2, 1.5, 1.2], [2.1, 1.7, 1.5], [0.5, 0.7, 0.75], [0.45, 0.62, 0.66], legacy=True
    )
    summary = summarize_history(calm)
    assert summary["findings"] == ["no_clear_issue"]
    assert summary["best"]["train_top5"] is None
    with pytest.raises(ValueError):
        summarize_history([])


def test_plot_and_cli_write_the_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from silent_signal.cli import plot_training

    history = _history(
        [3.0, 2.0, 1.5, 1.2], [3.1, 2.3, 2.0, 2.1], [0.1, 0.4, 0.6, 0.7], [0.1, 0.35, 0.45, 0.44]
    )
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "report.json").write_text(
        json.dumps(_report(4, stopped_early=False)), encoding="utf-8"
    )

    assert plot_training.main(["--run-root", str(tmp_path)]) == 0
    assert (tmp_path / "figures" / "training_curves.png").stat().st_size > 10_000
    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == 3
