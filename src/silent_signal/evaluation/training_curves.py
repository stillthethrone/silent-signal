"""Summaries, diagnostics and plots of train/validation curves from ``history.json``.

Train metrics are running values over augmented batches with dropout active, so they
understate what the model scores on clean training clips; read gaps with that in mind.
matplotlib is imported only when plotting.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

# Heuristic thresholds for the automatic findings (relative changes and accuracy points).
OVERFIT_MIN_EPOCHS_AFTER_BEST = 3
OVERFIT_VALIDATION_LOSS_RISE = 0.05
OVERFIT_TRAIN_LOSS_DROP = 0.05
LARGE_TOP1_GAP = 0.25
UNDERFIT_TRAIN_TOP1 = 0.5
NOISY_VALIDATION_FLUCTUATION = 0.05


def load_run(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Read ``history.json`` and, when training has finished, ``report.json``."""

    history = json.loads((run_root / "history.json").read_text(encoding="utf-8"))
    report_path = run_root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    return history, report


def summarize_history(
    history: Sequence[dict[str, Any]], *, report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Best/last epochs, train-validation gaps and heuristic findings.

    Findings: ``overfitting`` (validation loss rose while train loss kept falling after the
    best epoch), ``large_generalization_gap``, ``underfitting``, ``still_improving``
    (the best epoch is at the end of a run that used all its epochs), ``noisy_validation``
    (validation top-1 moves up and down beyond its trend) and ``no_clear_issue``.
    """

    rows = sorted(history, key=lambda row: row["epoch"])
    if not rows:
        raise ValueError("The training history is empty.")
    best = min(rows, key=lambda row: (row["validation_loss"], row["epoch"]))
    last = rows[-1]
    best_top1 = max(rows, key=lambda row: (row["validation_top1"], -row["epoch"]))
    since_best = last["epoch"] - best["epoch"]
    loss_rise = _relative(
        last["validation_loss"] - best["validation_loss"], best["validation_loss"]
    )
    train_drop = _relative(best["train_loss"] - last["train_loss"], best["train_loss"])
    gap_best = best["train_top1"] - best["validation_top1"]
    # Movement beyond the net trend over the last 10 epochs: 0 for a monotonic curve,
    # large when validation top-1 keeps going up and down (few validation signers).
    recent = [row["validation_top1"] for row in rows[-10:]]
    steps = [b - a for a, b in pairwise(recent)]
    jitter = (
        (sum(abs(step) for step in steps) - abs(sum(steps))) / len(steps)
        if len(recent) >= 3
        else None
    )
    requested = (report or {}).get("fingerprint", {}).get("training", {}).get("epochs")
    stopped_early = (report or {}).get("stopped_early")
    classes = (report or {}).get("data", {}).get("classes")

    findings = []
    if (
        since_best >= OVERFIT_MIN_EPOCHS_AFTER_BEST
        and loss_rise > OVERFIT_VALIDATION_LOSS_RISE
        and train_drop > OVERFIT_TRAIN_LOSS_DROP
    ):
        findings.append("overfitting")
    if gap_best > LARGE_TOP1_GAP:
        findings.append("large_generalization_gap")
    if best["train_top1"] < UNDERFIT_TRAIN_TOP1 and gap_best < 0.1:
        findings.append("underfitting")
    if requested and last["epoch"] >= requested and since_best <= 1 and not stopped_early:
        findings.append("still_improving")
    if jitter is not None and jitter > NOISY_VALIDATION_FLUCTUATION:
        findings.append("noisy_validation")
    return {
        "epochs": last["epoch"],
        "requested_epochs": requested,
        "stopped_early": stopped_early,
        "classes": classes,
        "chance_top1": 1 / classes if classes else None,
        "best_epoch": best["epoch"],
        "best": _metrics(best),
        "last": _metrics(last),
        "best_top1_epoch": best_top1["epoch"],
        "best_validation_top1": best_top1["validation_top1"],
        "epochs_since_best": since_best,
        "validation_loss_rise_since_best": loss_rise,
        "train_loss_drop_since_best": train_drop,
        "top1_gap_at_best": gap_best,
        "top1_gap_at_last": last["train_top1"] - last["validation_top1"],
        "validation_top1_fluctuation": jitter,
        "findings": findings or ["no_clear_issue"],
    }


def plot_training_curves(
    history: Sequence[dict[str, Any]],
    output_path: Path | None = None,
    *,
    summary: dict[str, Any] | None = None,
    title: str | None = None,
) -> Any:
    """Draw loss, top-1, top-5, macro-F1, generalization gap and learning rate (2 x 3)."""

    import matplotlib.pyplot as plt

    rows = sorted(history, key=lambda row: row["epoch"])
    summary = summary or summarize_history(rows)
    epochs = [row["epoch"] for row in rows]

    def series(key: str) -> list[float]:
        return [float(row.get(key, math.nan)) for row in rows]

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [
        (axes[0, 0], "Loss (cross-entropy)", "loss", None),
        (axes[0, 1], "Top-1 accuracy", "top1", (0, 1)),
        (axes[0, 2], "Top-5 accuracy", "top5", (0, 1)),
        (axes[1, 0], "Macro-F1", "macro_f1", (0, 1)),
    ]
    for axis, name, key, limits in panels:
        for split, color in (("train", "#1f77b4"), ("validation", "#ff7f0e")):
            values = series(f"{split}_{key}")
            if not all(math.isnan(value) for value in values):
                label = "train (augmented, dropout)" if split == "train" else "validation"
                axis.plot(epochs, values, color=color, marker="o", markersize=2.5, label=label)
        if key == "top1" and summary.get("chance_top1"):
            axis.axhline(
                summary["chance_top1"], color="gray", linewidth=0.8, linestyle="--", label="chance"
            )
        axis.set(title=name, xlabel="epoch")
        if limits:
            axis.set_ylim(*limits)
        axis.legend(fontsize=8)

    gap_axis = axes[1, 1]
    top1_gap = [t - v for t, v in zip(series("train_top1"), series("validation_top1"), strict=True)]
    loss_gap = [v - t for t, v in zip(series("train_loss"), series("validation_loss"), strict=True)]
    gap_axis.plot(epochs, top1_gap, color="#2ca02c", label="train - validation top-1")
    gap_axis.axhline(0, color="gray", linewidth=0.8)
    gap_axis.set(title="Generalization gap", xlabel="epoch", ylabel="top-1 gap")
    loss_axis = gap_axis.twinx()
    loss_axis.plot(
        epochs, loss_gap, color="#d62728", linestyle="--", label="validation - train loss"
    )
    loss_axis.set_ylabel("loss gap")
    handles = gap_axis.get_legend_handles_labels()
    extra = loss_axis.get_legend_handles_labels()
    gap_axis.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=8)

    rate_axis = axes[1, 2]
    rate_axis.plot(epochs, series("learning_rate"), color="#9467bd")
    rate_axis.set(title="Learning rate", xlabel="epoch")
    rate_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for axis in axes.flat:
        axis.axvline(summary["best_epoch"], color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.3)
    best = summary["best"]
    figure.suptitle(
        f"{title + ' — ' if title else ''}best epoch {summary['best_epoch']} (dotted): "
        f"val loss {best['validation_loss']:.3f} · top-1 {best['validation_top1']:.3f} · "
        f"top-5 {best['validation_top5']:.3f} · macro-F1 {best['validation_macro_f1']:.3f}",
        fontsize=12,
    )
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    keys = (
        "train_loss",
        "train_top1",
        "train_top5",
        "train_macro_f1",
        "validation_loss",
        "validation_top1",
        "validation_top5",
        "validation_macro_f1",
        "learning_rate",
    )
    return {key: row.get(key) for key in keys}


def _relative(change: float, base: float) -> float:
    return change / base if base > 0 else 0.0
