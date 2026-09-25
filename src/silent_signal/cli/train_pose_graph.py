"""Train and evaluate the graph-spatial-temporal pose recognizer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from silent_signal.data.collate import GraphPoseBatch, collate_graph_pose
from silent_signal.data.dataset import GraphPoseDataset
from silent_signal.data.manifest import read_manifest
from silent_signal.models.factory import (
    GraphEncoderExperimentConfig,
    build_pose_graph_recognizer,
    graph_encoder_config_dict,
    load_graph_encoder_config,
    model_fingerprint,
)
from silent_signal.pose.cache import runtime_environment, sha256_file, write_json_atomic
from silent_signal.preprocessing.pose_features import GraphPoseSample
from silent_signal.training.checkpoint import write_torch_checkpoint_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-train-pose-graph",
        description=(
            "Train Graph Encoder -> Spatial Transformer -> Temporal Transformer with "
            "validation early stopping and final split reports."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--project-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _train(args)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _train(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    if args.progress_every < 0:
        raise ValueError("--progress-every must not be negative.")
    experiment = load_graph_encoder_config(args.config)
    if experiment.training is None:
        raise ValueError("Experiment config requires a training mapping.")
    training = experiment.training
    manifest = args.manifest.resolve()
    manifest_hash = sha256_file(manifest)
    if experiment.manifest_sha256 != manifest_hash:
        raise ValueError("Manifest SHA-256 does not match the training config.")
    records = read_manifest(manifest)
    class_indices = sorted({record.class_index for record in records})
    expected_indices = list(range(experiment.model.num_classes))
    if class_indices != expected_indices:
        raise ValueError("Manifest classes are not contiguous or do not match num_classes.")
    labels = _read_labels(args.labels, experiment.model.num_classes)
    graph_root = args.graph_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _seed_everything(training.seed)

    datasets = {
        split: GraphPoseDataset(
            records,
            graph_root,
            split=split,
            expected_fingerprint=experiment.preprocessing_fingerprint,
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": _make_loader(
            datasets["train"],
            batch_size=training.batch_size,
            shuffle=True,
            num_workers=training.num_workers,
            seed=training.seed,
            device=device,
        ),
        "validation": _make_loader(
            datasets["validation"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            seed=training.seed,
            device=device,
        ),
        "test": _make_loader(
            datasets["test"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            seed=training.seed,
            device=device,
        ),
    }
    model = build_pose_graph_recognizer(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}; parameters={parameter_count:,}; "
        f"train/validation/test={len(datasets['train'])}/{len(datasets['validation'])}/"
        f"{len(datasets['test'])}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1
    last_checkpoint = output_root / "last_checkpoint.pt"
    best_checkpoint = output_root / "best_checkpoint.pt"
    if args.resume and last_checkpoint.is_file():
        checkpoint = _load_checkpoint(last_checkpoint, device)
        _validate_checkpoint(checkpoint, experiment, manifest_hash)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint.get("history", []))
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(
            f"Resume epoch {start_epoch}/{training.max_epochs}; "
            f"best epoch={best_epoch}; wait={bad_epochs}/{training.early_stopping_patience}",
            flush=True,
        )

    stopped_early = False
    for epoch in range(start_epoch, training.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            experiment.model.num_classes,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=training.gradient_clip,
            augment={
                "scale": training.coordinate_scale_jitter,
                "translation": training.coordinate_translation_jitter,
                "joint_dropout": training.joint_dropout,
            },
            progress_every=args.progress_every,
            stage=f"train {epoch}/{training.max_epochs}",
        )
        validation_metrics = _run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            experiment.model.num_classes,
            progress_every=args.progress_every,
            stage=f"validation {epoch}/{training.max_epochs}",
        )
        improved = validation_metrics["loss"] < (
            best_validation_loss - training.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_top1": validation_metrics["top1"],
            "validation_macro_precision": validation_metrics["macro_precision"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best": improved,
            "early_stopping_bad_epochs": bad_epochs,
            "duration_seconds": round(time.perf_counter() - epoch_started, 3),
        }
        history.append(record)
        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            experiment=experiment,
            manifest_sha256=manifest_hash,
            project_commit=args.project_commit,
            epoch=epoch,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            bad_epochs=bad_epochs,
        )
        write_torch_checkpoint_atomic(last_checkpoint, checkpoint_payload, overwrite=True)
        if improved:
            write_torch_checkpoint_atomic(best_checkpoint, checkpoint_payload, overwrite=True)
        write_json_atomic(output_root / "history.json", {"epochs": history})
        _plot_history(history, output_root / "training_curves.png", best_epoch)
        print(
            f"[epoch {epoch}/{training.max_epochs}] "
            f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.3f} "
            f"F1={train_metrics['macro_f1']:.3f} | "
            f"validation loss={validation_metrics['loss']:.4f} "
            f"top1={validation_metrics['top1']:.3f} "
            f"P/R/F1={validation_metrics['macro_precision']:.3f}/"
            f"{validation_metrics['macro_recall']:.3f}/"
            f"{validation_metrics['macro_f1']:.3f} | "
            f"best={improved} wait={bad_epochs}/{training.early_stopping_patience} "
            f"time={_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
        if (
            epoch >= training.early_stopping_min_epochs
            and bad_epochs >= training.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}; restoring best epoch {best_epoch}.",
                flush=True,
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("No best checkpoint was written.")
    best = _load_checkpoint(best_checkpoint, device)
    _validate_checkpoint(best, experiment, manifest_hash)
    model.load_state_dict(best["model_state_dict"])
    evaluation_splits = ["train", "validation"]
    if args.run_test:
        evaluation_splits.append("test")
    evaluations: dict[str, dict[str, Any]] = {}
    for split in evaluation_splits:
        metrics = _run_epoch(
            model,
            loaders[split],
            criterion,
            device,
            experiment.model.num_classes,
            collect_predictions=True,
            progress_every=args.progress_every,
            stage=f"final {split}",
        )
        evaluations[split] = metrics
        _write_predictions(output_root / f"{split}_predictions.csv", metrics, labels)
    _plot_split_metrics(evaluations, output_root / "split_metrics_comparison.png")
    if args.run_test:
        _plot_confusion(
            np.asarray(evaluations["test"]["confusion_matrix"]),
            labels,
            output_root / "test_confusion_matrix.png",
        )
        _plot_per_class_f1(
            evaluations["test"]["per_class"],
            labels,
            output_root / "test_per_class_f1.png",
            float(evaluations["test"]["macro_f1"]),
        )

    clean_evaluations = {
        split: {
            key: value
            for key, value in metrics.items()
            if key not in {"sample_ids", "targets", "predictions"}
        }
        for split, metrics in evaluations.items()
    }
    best_history = next(item for item in history if int(item["epoch"]) == best_epoch)
    report = {
        "schema_version": 1,
        "state": "complete",
        "project_commit": args.project_commit,
        "architecture": experiment.model.architecture,
        "model": asdict(experiment.model),
        "training": asdict(training),
        "evaluation_protocol": experiment.evaluation_protocol,
        "manifest_sha256": manifest_hash,
        "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
        "parameter_count": parameter_count,
        "device": str(device),
        "environment": runtime_environment(),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "early_stopping": {
            "triggered": stopped_early,
            "minimum_epochs": training.early_stopping_min_epochs,
            "patience": training.early_stopping_patience,
            "min_delta": training.early_stopping_min_delta,
        },
        "best_epoch_generalization_gap": {
            "top1": best_history["train_top1"] - best_history["validation_top1"],
            "macro_f1": (
                best_history["train_macro_f1"] - best_history["validation_macro_f1"]
            ),
        },
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "test_was_run": args.run_test,
        "evaluation": clean_evaluations,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "history": str(output_root / "history.json"),
            "training_curves": str(output_root / "training_curves.png"),
            "split_metrics_comparison": str(output_root / "split_metrics_comparison.png"),
            "test_confusion_matrix": (
                str(output_root / "test_confusion_matrix.png") if args.run_test else None
            ),
            "test_per_class_f1": (
                str(output_root / "test_per_class_f1.png") if args.run_test else None
            ),
        },
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }
    write_json_atomic(output_root / "training_report.json", report)
    console_report = {
        "state": report["state"],
        "architecture": report["architecture"],
        "completed_epochs": report["completed_epochs"],
        "best_epoch": report["best_epoch"],
        "best_validation_loss": report["best_validation_loss"],
        "early_stopping": report["early_stopping"],
        "evaluation": {
            split: {
                key: metrics[key]
                for key in ("loss", "top1", "macro_precision", "macro_recall", "macro_f1")
            }
            for split, metrics in clean_evaluations.items()
        },
        "artifacts": report["artifacts"],
    }
    print(json.dumps(console_report, ensure_ascii=False, indent=2), flush=True)


def _make_loader(
    dataset: GraphPoseDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[GraphPoseSample]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_samples,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _collate_samples(samples: list[GraphPoseSample]) -> GraphPoseBatch:
    return collate_graph_pose(samples)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[GraphPoseSample],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
    augment: Mapping[str, float] | None = None,
    collect_predictions: bool = False,
    progress_every: int = 20,
    stage: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    sample_ids: list[str] = []
    targets: list[int] = []
    predictions: list[int] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        features = torch.from_numpy(batch.features).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        joint_mask = torch.from_numpy(batch.joint_mask).to(device=device, non_blocking=True)
        frame_mask = torch.from_numpy(batch.frame_mask).to(device=device, non_blocking=True)
        labels = torch.from_numpy(batch.labels).to(device=device, non_blocking=True)
        adjacency = torch.from_numpy(batch.adjacency).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if training and augment is not None:
            features, joint_mask, frame_mask = _augment_graph_batch(
                features,
                joint_mask,
                frame_mask,
                scale_jitter=float(augment["scale"]),
                translation_jitter=float(augment["translation"]),
                joint_dropout=float(augment["joint_dropout"]),
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(features, joint_mask, frame_mask, adjacency)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss during {stage}.")
            if training:
                if scaler is None:
                    raise RuntimeError("Training requires a gradient scaler.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        predicted = logits.detach().argmax(dim=1)
        batch_targets = labels.detach().cpu().numpy().astype(np.int64)
        batch_predictions = predicted.cpu().numpy().astype(np.int64)
        np.add.at(confusion, (batch_targets, batch_predictions), 1)
        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_samples += count
        if collect_predictions:
            sample_ids.extend(batch.sample_ids)
            targets.extend(batch_targets.tolist())
            predictions.extend(batch_predictions.tolist())
        if progress_every and (
            batch_index == 1 or batch_index == len(loader) or batch_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - started
            rate = batch_index / elapsed if elapsed else 0.0
            eta = (len(loader) - batch_index) / rate if rate else 0.0
            print(
                f"[{stage}] batch {batch_index}/{len(loader)} | "
                f"loss={total_loss / total_samples:.4f} | ETA={_duration(eta)}",
                flush=True,
            )
    metrics = _classification_metrics(confusion)
    metrics["loss"] = total_loss / max(1, total_samples)
    metrics["samples"] = total_samples
    if collect_predictions:
        metrics["sample_ids"] = sample_ids
        metrics["targets"] = targets
        metrics["predictions"] = predictions
    return metrics


def _augment_graph_batch(
    features: Tensor,
    joint_mask: Tensor,
    frame_mask: Tensor,
    *,
    scale_jitter: float,
    translation_jitter: float,
    joint_dropout: float,
) -> tuple[Tensor, Tensor, Tensor]:
    features = features.clone()
    batch = features.shape[0]
    scale = 1.0 + (
        torch.rand(batch, 1, 1, 1, device=features.device) * 2.0 - 1.0
    ) * scale_jitter
    geometric_channels = torch.tensor([0, 1, 3, 4, 5, 6], device=features.device)
    features[..., geometric_channels] = features[..., geometric_channels] * scale
    translation = (
        torch.rand(batch, 1, 1, 2, device=features.device) * 2.0 - 1.0
    ) * translation_jitter
    features[..., :2] = features[..., :2] + translation
    if joint_dropout > 0:
        retained = torch.rand_like(joint_mask, dtype=torch.float32) >= joint_dropout
        joint_mask = joint_mask & retained
        frame_mask = frame_mask & joint_mask.any(dim=2)
    features = features * joint_mask.unsqueeze(-1).to(dtype=features.dtype)
    return features, joint_mask, frame_mask


def _classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    true_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=true_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(confusion.sum())
    return {
        "top1": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(true_count[index]),
            }
            for index in range(confusion.shape[0])
        ],
    }


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    experiment: GraphEncoderExperimentConfig,
    manifest_sha256: str,
    project_commit: str | None,
    epoch: int,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_validation_loss: float,
    bad_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pose_graph_full_training_checkpoint",
        "project_commit": project_commit,
        "model_fingerprint": model_fingerprint(experiment.model),
        "experiment": graph_encoder_config_dict(experiment),
        "manifest_sha256": manifest_sha256,
        "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "bad_epochs": bad_epochs,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return loaded


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    experiment: GraphEncoderExperimentConfig,
    manifest_sha256: str,
) -> None:
    if checkpoint.get("model_fingerprint") != model_fingerprint(experiment.model):
        raise ValueError("Checkpoint model fingerprint does not match the config.")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint manifest does not match the current manifest.")
    if checkpoint.get("preprocessing_fingerprint") != experiment.preprocessing_fingerprint:
        raise ValueError("Checkpoint preprocessing fingerprint does not match graph caches.")


def _read_labels(path: Path, expected_classes: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("class_index_to_gloss")
    if not isinstance(mapping, dict):
        raise ValueError("Labels file is missing class_index_to_gloss.")
    labels = [str(mapping[str(index)]) for index in range(expected_classes)]
    if len(labels) != expected_classes:
        raise ValueError("Labels count does not match model classes.")
    return labels


def _write_predictions(
    path: Path,
    metrics: Mapping[str, Any],
    labels: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("sample_id", "target_index", "target_gloss", "predicted_index", "predicted_gloss")
        )
        for sample_id, target, predicted in zip(
            metrics["sample_ids"],
            metrics["targets"],
            metrics["predictions"],
            strict=True,
        ):
            writer.writerow((sample_id, target, labels[target], predicted, labels[predicted]))


def _plot_history(history: Sequence[Mapping[str, Any]], path: Path, best_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [item["train_top1"] for item in history], label="train")
    axes[1].plot(epochs, [item["validation_top1"] for item in history], label="validation")
    axes[1].set(title="Top-1 accuracy", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[2].plot(epochs, [item["train_macro_f1"] for item in history], label="train")
    axes[2].plot(epochs, [item["validation_macro_f1"] for item in history], label="validation")
    axes[2].set(title="Macro-F1", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.axvline(best_epoch, color="red", linestyle="--", label=f"best={best_epoch}")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Graph + Spatial Transformer + Temporal Transformer")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_split_metrics(evaluations: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    splits = list(evaluations)
    metric_keys = ("top1", "macro_precision", "macro_recall", "macro_f1")
    metric_labels = ("Top-1", "Macro-P", "Macro-R", "Macro-F1")
    positions = np.arange(len(metric_keys))
    width = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(10, 5))
    for split_index, split in enumerate(splits):
        offset = (split_index - (len(splits) - 1) / 2) * width
        values = [float(evaluations[split][key]) for key in metric_keys]
        bars = axis.bar(positions + offset, values, width=width, label=split)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(positions, metric_labels)
    axis.set(title="Best-checkpoint split comparison", ylabel="Score", ylim=(0, 1.08))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion(confusion: np.ndarray, labels: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums > 0,
    )
    size = max(12.0, len(labels) * 0.30)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set(title="Normalized confusion matrix on test", xlabel="Predicted", ylabel="True")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Recall per class")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_per_class_f1(
    per_class: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    path: Path,
    macro_f1: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ordered = sorted(per_class, key=lambda item: (float(item["f1"]), int(item["class_index"])))
    names = [f"{labels[int(item['class_index'])]} (n={int(item['support'])})" for item in ordered]
    values = [float(item["f1"]) for item in ordered]
    colors = [
        "#d62728" if value < 0.4 else "#ffb000" if value < macro_f1 else "#2ca02c"
        for value in values
    ]
    figure, axis = plt.subplots(figsize=(10, max(8, len(labels) * 0.30)))
    axis.barh(range(len(names)), values, color=colors)
    axis.axvline(macro_f1, color="blue", linestyle="--", label=f"Macro-F1={macro_f1:.3f}")
    axis.set_yticks(range(len(names)), names, fontsize=8)
    axis.set(title="F1-score by class on test", xlabel="F1-score", xlim=(0, 1))
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    entrypoint()
