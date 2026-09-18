"""Analyze validation or test errors from the 50-class RGB Transformer demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silent_signal.cli.analyze_videomaev2_demo",
        description="Error analysis for the RGB-only VideoMAE V2 demo.",
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--top-errors", type=int, default=15)
    return parser


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    baseline_root = args.baseline_root.resolve()
    selection_path = baseline_root / "selected_50_words.json"
    baseline_report_path = baseline_root / "baseline_report.json"
    predictions_path = baseline_root / f"{args.split}_predictions.csv"
    for path in (selection_path, baseline_report_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    classes = list(selection["classes"])
    if len(classes) != 50:
        raise RuntimeError("Error analysis expects exactly 50 demo classes.")
    class_indices = [int(item["class_index"]) for item in classes]
    if class_indices != list(range(50)):
        raise RuntimeError("Demo class indices must be contiguous from 0 to 49.")
    labels = [str(item["gloss_name"]) for item in classes]
    predictions = pd.read_csv(predictions_path)
    required = {
        "sample_id",
        "true_class",
        "pred_class",
        "true_gloss",
        "pred_gloss",
        "confidence",
        "split",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise RuntimeError(f"Prediction CSV is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise RuntimeError("Prediction CSV is empty.")
    if set(predictions["split"]) != {args.split}:
        raise RuntimeError("Prediction rows do not match the requested split.")
    if predictions["sample_id"].duplicated().any():
        raise RuntimeError("Prediction CSV contains duplicate sample IDs.")
    if not set(predictions["true_class"]).issubset(class_indices):
        raise RuntimeError("Prediction CSV contains an unknown true class.")
    if not set(predictions["pred_class"]).issubset(class_indices):
        raise RuntimeError("Prediction CSV contains an unknown predicted class.")

    y_true = predictions["true_class"].astype(int).to_numpy()
    y_pred = predictions["pred_class"].astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=class_indices)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    metrics = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        [
            {
                "class_index": index,
                "gloss": label,
                "precision": metrics[label]["precision"],
                "recall": metrics[label]["recall"],
                "f1": metrics[label]["f1-score"],
                "support": int(metrics[label]["support"]),
                "errors": int(matrix[index].sum() - matrix[index, index]),
            }
            for index, label in enumerate(labels)
        ]
    )
    per_class_path = baseline_root / f"{args.split}_per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(26, 12))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set(title=f"{args.split.title()} confusion — counts", xlabel="Predicted", ylabel="True")
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="magma", ax=axes[1], vmin=0, vmax=1)
    axes[1].set(
        title=f"{args.split.title()} confusion — row normalized",
        xlabel="Predicted",
        ylabel="True",
    )
    tick_positions = np.arange(len(labels)) + 0.5
    for axis in axes:
        axis.set_xticks(tick_positions, labels, rotation=75, ha="right")
        axis.set_yticks(tick_positions, labels, rotation=0)
    figure.suptitle("VideoMAE V2 + RGB Transformer demo (50 classes; not final benchmark)")
    figure.tight_layout()
    confusion_path = baseline_root / f"{args.split}_confusion_matrices.png"
    figure.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    ordered = per_class.sort_values("f1", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 11))
    positions = np.arange(len(ordered))
    axis.barh(positions - 0.18, ordered["recall"], height=0.36, label="recall")
    axis.barh(positions + 0.18, ordered["f1"], height=0.36, label="F1")
    axis.set_yticks(positions, ordered["gloss"])
    axis.set(xlim=(0, 1), xlabel="Score", title=f"Per-class metrics on {args.split}")
    axis.legend()
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    per_class_plot = baseline_root / f"{args.split}_per_class_metrics.png"
    figure.savefig(per_class_plot, dpi=160, bbox_inches="tight")
    plt.close(figure)

    predictions["correct"] = predictions["true_class"] == predictions["pred_class"]
    figure, axis = plt.subplots(figsize=(9, 5))
    for correct, label, color in ((True, "correct", "#2a9d8f"), (False, "wrong", "#e76f51")):
        values = predictions.loc[predictions["correct"] == correct, "confidence"]
        if not values.empty:
            axis.hist(values, bins=10, alpha=0.65, label=label, color=color)
    axis.set(
        xlabel="Top-1 confidence",
        ylabel="Samples",
        title=f"Confidence of correct vs wrong predictions ({args.split})",
    )
    axis.legend()
    figure.tight_layout()
    confidence_path = baseline_root / f"{args.split}_confidence_histogram.png"
    figure.savefig(confidence_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    pairs = []
    for true_index in class_indices:
        for predicted_index in class_indices:
            if true_index != predicted_index and matrix[true_index, predicted_index] > 0:
                pairs.append(
                    {
                        "true_class": true_index,
                        "true_gloss": labels[true_index],
                        "pred_class": predicted_index,
                        "pred_gloss": labels[predicted_index],
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    pairs.sort(key=lambda item: (-item["count"], item["true_gloss"], item["pred_gloss"]))
    top_pairs = pairs[: args.top_errors]
    pair_labels = [f"{item['true_gloss']} → {item['pred_gloss']}" for item in reversed(top_pairs)]
    figure, axis = plt.subplots(figsize=(11, max(4, len(pair_labels) * 0.42)))
    axis.barh(pair_labels, [item["count"] for item in reversed(top_pairs)], color="#e76f51")
    axis.set(xlabel="Misclassified clips", title=f"Top confusion directions ({args.split})")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    confusion_pairs_path = baseline_root / f"{args.split}_top_confusions.png"
    figure.savefig(confusion_pairs_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    split_table = pd.DataFrame(
        [
            {
                "gloss": item["gloss_name"],
                **{split: item["counts"][split] for split in ("train", "validation", "test")},
            }
            for item in classes
        ]
    ).set_index("gloss")
    axis = split_table.plot(kind="bar", figsize=(14, 6), width=0.85)
    axis.set(
        ylabel="Clips",
        title="Train/validation/test counts for the selected 50 words",
    )
    axis.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    split_plot_path = baseline_root / "selected_50_official_split_counts.png"
    plt.savefig(split_plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    evaluation = baseline_report.get("evaluation", {}).get(args.split, {})
    accuracy = float((y_true == y_pred).mean())
    lowest_recall = per_class.sort_values(["recall", "support"]).head(5).to_dict("records")
    history = list(baseline_report.get("history", ()))
    best_epoch = int(baseline_report.get("best_epoch", 0))
    best_record = next(
        (item for item in history if int(item.get("epoch", -1)) == best_epoch),
        None,
    )
    final_record = history[-1] if history else None

    def gap_payload(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        train_top1 = float(record["train_top1"])
        validation_top1 = float(record["validation_top1"])
        payload = {
            "epoch": int(record["epoch"]),
            "train_top1": train_top1,
            "validation_top1": validation_top1,
            "top1_generalization_gap": train_top1 - validation_top1,
            "train_loss": float(record["train_loss"]),
            "validation_loss": float(record["validation_loss"]),
        }
        if "train_macro_f1" in record and "validation_macro_f1" in record:
            train_macro_f1 = float(record["train_macro_f1"])
            validation_macro_f1 = float(record["validation_macro_f1"])
            payload.update(
                {
                    "train_macro_f1": train_macro_f1,
                    "validation_macro_f1": validation_macro_f1,
                    "macro_f1_generalization_gap": train_macro_f1 - validation_macro_f1,
                }
            )
        return payload

    support_values = per_class["support"].astype(int)
    train_counts = np.asarray([int(item["counts"]["train"]) for item in classes], dtype=int)
    high_confidence_errors = int(
        ((~predictions["correct"]) & (predictions["confidence"] >= 0.8)).sum()
    )
    generalization = {
        "best_checkpoint": gap_payload(best_record),
        "last_trained_epoch": gap_payload(final_record),
        "overfit_warning": bool(
            best_record is not None
            and float(best_record["train_top1"]) - float(best_record["validation_top1"]) >= 0.2
        ),
    }
    class_support = {
        "train_min": int(train_counts.min()),
        "train_median": float(np.median(train_counts)),
        "train_max": int(train_counts.max()),
        "train_max_to_min_ratio": float(train_counts.max() / max(train_counts.min(), 1)),
        "analysis_min": int(support_values.min()),
        "analysis_median": float(support_values.median()),
        "analysis_max": int(support_values.max()),
        "analysis_classes_below_5": int((support_values < 5).sum()),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "state": "passed",
        "study_stage": "complete-data 50-class RGB-only demo error analysis",
        "analysis_split": args.split,
        "warning": (
            "Use validation errors to design improvements. Do not tune from test results; "
            "rerun the final study on all 200 classes."
        ),
        "evaluated_samples": len(predictions),
        "available_samples": evaluation.get("available_samples"),
        "partial_evaluation": evaluation.get("partial"),
        "top1_accuracy": accuracy,
        "macro_precision": metrics["macro avg"]["precision"],
        "macro_recall": metrics["macro avg"]["recall"],
        "macro_f1": metrics["macro avg"]["f1-score"],
        "generalization": generalization,
        "class_support": class_support,
        "high_confidence_errors_at_0_8": high_confidence_errors,
        "lowest_recall_classes": lowest_recall,
        "top_confusions": top_pairs,
        "artifacts": {
            "per_class_csv": str(per_class_path),
            "confusion_matrices": str(confusion_path),
            "per_class_plot": str(per_class_plot),
            "confidence_histogram": str(confidence_path),
            "top_confusions_plot": str(confusion_pairs_path),
            "official_split_counts": str(split_plot_path),
        },
    }
    report_path = baseline_root / f"{args.split}_error_analysis.json"
    _write_json_atomic(report_path, report)
    print(
        f"[{args.split}] top1={accuracy:.3f}; macro-F1={report['macro_f1']:.3f}; "
        f"samples={len(predictions)}/{evaluation.get('available_samples', '?')}",
        flush=True,
    )
    if generalization["best_checkpoint"] is not None:
        item = generalization["best_checkpoint"]
        print(
            f"Best epoch {item['epoch']}: train top1={item['train_top1']:.3f}; "
            f"validation top1={item['validation_top1']:.3f}; "
            f"gap={item['top1_generalization_gap']:.3f}; "
            f"overfit_warning={generalization['overfit_warning']}",
            flush=True,
        )
    print(
        "Class support: "
        f"train min/median/max={class_support['train_min']}/"
        f"{class_support['train_median']:.1f}/{class_support['train_max']}; "
        f"{args.split} classes below 5 clips={class_support['analysis_classes_below_5']}",
        flush=True,
    )
    print("\nCác lớp recall thấp nhất:", flush=True)
    for item in lowest_recall:
        print(
            f"- {item['gloss']}: recall={item['recall']:.3f}, "
            f"F1={item['f1']:.3f}, support={item['support']}",
            flush=True,
        )
    print("\nCác cặp nhầm nhiều nhất:", flush=True)
    for item in top_pairs:
        print(f"- {item['true_gloss']} -> {item['pred_gloss']}: {item['count']}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
