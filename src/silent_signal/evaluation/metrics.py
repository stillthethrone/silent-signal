"""Classification metrics for isolated sign recognition (NumPy only)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def classification_metrics(
    y_true: NDArray[np.integer],
    probabilities: NDArray[np.floating],
) -> dict[str, Any]:
    """Top-1/top-5 accuracy, macro precision/recall/F1, per-class scores and confusion.

    ``probabilities`` has shape ``[N, K]``. Classes absent from ``y_true`` still count in
    the macro averages with recall 0 when predicted, so every gloss weighs the same.
    """

    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or len(truth) != len(scores) or len(truth) == 0:
        raise ValueError("Expected probabilities [N, K] matching N labels.")
    classes = scores.shape[1]
    if truth.min() < 0 or truth.max() >= classes:
        raise ValueError("Labels are outside the probability columns.")
    predicted = scores.argmax(axis=1)
    top = np.argsort(-scores, axis=1)[:, : min(5, classes)]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(classes), where=predicted_count > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros(classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros(classes), where=denominator > 0
    )
    present = (support > 0) | (predicted_count > 0)
    return {
        "samples": len(truth),
        "top1_accuracy": float((predicted == truth).mean()),
        "top5_accuracy": float((top == truth[:, None]).any(axis=1).mean()),
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": [
            {
                "class_index": index,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(classes)
        ],
        "confusion": confusion.tolist(),
    }
