"""Train and evaluate the frozen ASL top-50 graph Transformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from silent_signal.contracts import ManifestRecord
from silent_signal.data.collate import GraphPoseBatch, collate_graph_pose
from silent_signal.data.dataset import GraphPoseDataset
from silent_signal.data.manifest import ManifestError, read_manifest
from silent_signal.models.factory import (
    FullTrainingConfig,
    build_pose_graph_recognizer,
    graph_encoder_config_dict,
    load_graph_encoder_config,
    model_fingerprint,
)
from silent_signal.pose.cache import PoseCacheError, sha256_file, write_json_atomic
from silent_signal.training.checkpoint import write_torch_checkpoint_atomic

_SPLITS = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-train-graph-transformer",
        description=(
            "Train Graph Encoder + Spatial Transformer + Temporal Transformer "
            "on official ASL Citizen splits."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project-commit")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.perf_counter()
    try:
        experiment = load_graph_encoder_config(args.config)
        if experiment.training is None:
            raise ValueError("Graph encoder config does not define a training mapping.")
        training = _apply_overrides(experiment.training, args)
        manifest_path = args.manifest.resolve()
        manifest_sha256 = sha256_file(manifest_path)
        if (
            experiment.manifest_sha256 is not None
            and manifest_sha256 != experiment.manifest_sha256
        ):
            raise ValueError("Manifest SHA-256 does not match the experiment config.")
        records = tuple(read_manifest(manifest_path))
        split_summary = _validate_records(records, experiment.model.num_classes)
        device = _resolve_device(args.device)
        _seed_everything(training.seed)

        output_root = args.output_root.resolve()
        best_path = output_root / "best.pt"
        last_path = output_root / "last.pt"
        history_path = output_root / "history.json"
        report_path = output_root / "evaluation.json"
        if report_path.exists() and not args.overwrite and args.resume is None:
            raise FileExistsError(
                f"Completed report already exists: {report_path}; use --overwrite to replace it."
            )
        if best_path.exists() and not args.overwrite and args.resume is None:
            raise FileExistsError(
                f"Best checkpoint already exists: {best_path}; resume or use --overwrite."
            )
        output_root.mkdir(parents=True, exist_ok=True)

        datasets = {
            split: GraphPoseDataset(
                records,
                args.graph_root.resolve(),
                split=split,
                expected_fingerprint=experiment.preprocessing_fingerprint,
            )
            for split in _SPLITS
        }
        loaders = _make_loaders(datasets, training, device=device)
        model = build_pose_graph_recognizer(experiment.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training.max_epochs,
        )
        class_weights = _class_weights(records, experiment.model.num_classes).to(device)
        train_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=training.label_smoothing,
        )
        evaluation_criterion = nn.CrossEntropyLoss()

        history: list[dict[str, Any]] = []
        start_epoch = 1
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        if args.resume is not None:
            resume_payload = _load_checkpoint(args.resume.resolve(), device)
            _validate_resume(
                resume_payload,
                manifest_sha256=manifest_sha256,
                preprocessing_fingerprint=experiment.preprocessing_fingerprint,
                active_model_fingerprint=model_fingerprint(experiment.model),
            )
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            history = list(resume_payload.get("history", []))
            start_epoch = int(resume_payload["epoch"]) + 1
            best_metric = float(resume_payload["best_validation_macro_f1"])
            best_epoch = int(resume_payload["best_epoch"])
            epochs_without_improvement = int(
                resume_payload.get("epochs_without_improvement", 0)
            )

        print(
            json.dumps(
                {
                    "device": str(device),
                    "split_counts": split_summary["clip_counts"],
                    "batch_size": training.batch_size,
                    "max_epochs": training.max_epochs,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "start_epoch": start_epoch,
                },
                indent=2,
            ),
            flush=True,
        )

        stopped_early = False
        for epoch in range(start_epoch, training.max_epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = _run_epoch(
                model,
                loaders["train"],
                train_criterion,
                device=device,
                num_classes=experiment.model.num_classes,
                optimizer=optimizer,
                gradient_clip=training.gradient_clip,
            )
            validation_metrics = _run_epoch(
                model,
                loaders["validation"],
                evaluation_criterion,
                device=device,
                num_classes=experiment.model.num_classes,
            )
            scheduler.step()
            current_metric = float(validation_metrics["macro_f1"])
            improved = current_metric > best_metric + training.early_stopping_min_delta
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_result = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "validation": validation_metrics,
                "improved": improved,
                "duration_seconds": round(time.perf_counter() - epoch_started, 3),
            }
            history.append(epoch_result)
            checkpoint = _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_metric=best_metric,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                experiment=experiment,
                manifest_sha256=manifest_sha256,
                project_commit=args.project_commit,
            )
            write_torch_checkpoint_atomic(last_path, checkpoint, overwrite=True)
            if improved:
                write_torch_checkpoint_atomic(best_path, checkpoint, overwrite=True)
            write_json_atomic(
                history_path,
                {
                    "schema_version": 1,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_metric,
                    "epochs": history,
                },
            )
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"train_top1={train_metrics['top1']:.4f} "
                f"val_loss={validation_metrics['loss']:.5f} "
                f"val_top1={validation_metrics['top1']:.4f} "
                f"val_macro_f1={current_metric:.4f} best_epoch={best_epoch}",
                flush=True,
            )
            if (
                epoch >= training.early_stopping_min_epochs
                and epochs_without_improvement >= training.early_stopping_patience
            ):
                stopped_early = True
                break

        if not best_path.is_file():
            raise RuntimeError("Training did not produce a best checkpoint.")
        best_payload = _load_checkpoint(best_path, device)
        model.load_state_dict(best_payload["model_state_dict"])
        test_metrics = _run_epoch(
            model,
            loaders["test"],
            evaluation_criterion,
            device=device,
            num_classes=experiment.model.num_classes,
        )
        glosses = _glosses_by_class(records, experiment.model.num_classes)
        test_metrics["per_class"] = [
            {"class_index": index, "gloss_name": glosses[index], **metrics}
            for index, metrics in enumerate(test_metrics.pop("per_class"))
        ]
        report = {
            "schema_version": 1,
            "state": "completed",
            "study": "ASL Citizen official-split top-50 pose-only baseline",
            "architecture": experiment.model.architecture,
            "project_commit": args.project_commit,
            "device": str(device),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "graph_root": str(args.graph_root.resolve()),
            "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
            "model_fingerprint": model_fingerprint(experiment.model),
            "config": graph_encoder_config_dict(experiment),
            "split": split_summary,
            "class_weights": class_weights.detach().cpu().tolist(),
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_macro_f1": float(
                best_payload["best_validation_macro_f1"]
            ),
            "stopped_early": stopped_early,
            "completed_epochs": len(history),
            "test": test_metrics,
            "artifacts": {
                "best_checkpoint": str(best_path),
                "best_checkpoint_sha256": sha256_file(best_path),
                "last_checkpoint": str(last_path),
                "history": str(history_path),
            },
            "duration_seconds": round(time.perf_counter() - started_at, 3),
        }
        write_json_atomic(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (
        FileExistsError,
        ManifestError,
        OSError,
        PoseCacheError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _apply_overrides(config: FullTrainingConfig, args: argparse.Namespace) -> FullTrainingConfig:
    return replace(
        config,
        max_epochs=args.epochs or config.max_epochs,
        batch_size=args.batch_size or config.batch_size,
        num_workers=(config.num_workers if args.num_workers is None else args.num_workers),
        seed=config.seed if args.seed is None else args.seed,
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_records(
    records: Sequence[ManifestRecord], num_classes: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("Manifest is empty.")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest contains duplicate sample_id values.")
    classes = {record.class_index for record in records}
    if classes != set(range(num_classes)):
        raise ValueError(
            f"Manifest classes must be contiguous 0..{num_classes - 1}; found {sorted(classes)}."
        )
    split_counts = Counter(str(record.split) for record in records)
    if set(split_counts) != set(_SPLITS):
        raise ValueError("Manifest must contain train, validation, and test splits.")
    for class_index in range(num_classes):
        coverage = {
            str(record.split) for record in records if record.class_index == class_index
        }
        if coverage != set(_SPLITS):
            raise ValueError(f"Class {class_index} is not represented in every split.")
    signer_ids = {
        split: {record.signer_id for record in records if record.split == split}
        for split in _SPLITS
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = signer_ids[first] & signer_ids[second]
        if overlap:
            raise ValueError(f"Signer leakage between {first}/{second}: {sorted(overlap)}.")
    total = len(records)
    return {
        "policy": "official ASL Citizen; never re-split",
        "clip_counts": {split: split_counts[split] for split in _SPLITS},
        "clip_percentages": {
            split: round(split_counts[split] / total * 100.0, 4) for split in _SPLITS
        },
        "signer_counts": {split: len(signer_ids[split]) for split in _SPLITS},
        "signer_ids": {split: sorted(signer_ids[split]) for split in _SPLITS},
    }


def _make_loaders(
    datasets: dict[str, GraphPoseDataset],
    config: FullTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, DataLoader[Any]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return {
        split: DataLoader(
            cast(Any, dataset),
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            collate_fn=collate_graph_pose,
        )
        for split, dataset in datasets.items()
    }


def _class_weights(records: Sequence[ManifestRecord], num_classes: int) -> Tensor:
    counts = np.bincount(
        [record.class_index for record in records if record.split == "train"],
        minlength=num_classes,
    )
    if np.any(counts == 0):
        raise ValueError("Every class must have at least one train sample.")
    weights = counts.sum() / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    *,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_samples = 0
    top5_correct = 0
    for batch in loader:
        if not isinstance(batch, GraphPoseBatch):
            raise RuntimeError("Graph DataLoader returned an unexpected batch type.")
        features = torch.from_numpy(batch.features).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        joint_mask = torch.from_numpy(batch.joint_mask).to(
            device=device, non_blocking=True
        )
        frame_mask = torch.from_numpy(batch.frame_mask).to(
            device=device, non_blocking=True
        )
        labels = torch.from_numpy(batch.labels).to(device=device, non_blocking=True)
        adjacency = torch.from_numpy(batch.adjacency).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, joint_mask, frame_mask, adjacency)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Model produced a non-finite loss.")
            if optimizer is not None:
                loss.backward()
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        predictions = logits.argmax(dim=1)
        encoded = labels * num_classes + predictions
        confusion += torch.bincount(
            encoded.detach().cpu(), minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
        top_k = min(5, num_classes)
        top5_correct += int(
            (logits.topk(top_k, dim=1).indices == labels.unsqueeze(1))
            .any(dim=1)
            .sum()
            .item()
        )
    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples.")
    metrics = _metrics_from_confusion(confusion)
    metrics.update(
        {
            "loss": total_loss / total_samples,
            "top5": top5_correct / total_samples,
            "samples": total_samples,
            "confusion_matrix": confusion.tolist(),
        }
    )
    return metrics


def _metrics_from_confusion(confusion: Tensor) -> dict[str, Any]:
    values = confusion.to(dtype=torch.float64)
    true_positive = values.diag()
    predicted = values.sum(dim=0)
    actual = values.sum(dim=1)
    precision = torch.where(predicted > 0, true_positive / predicted, 0.0)
    recall = torch.where(actual > 0, true_positive / actual, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2.0 * precision * recall / denominator, 0.0)
    total = float(values.sum())
    per_class = [
        {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(actual[index]),
        }
        for index in range(confusion.shape[0])
    ]
    return {
        "top1": float(true_positive.sum()) / total if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
    }


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: list[dict[str, Any]],
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
    experiment: Any,
    manifest_sha256: str,
    project_commit: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "graph_spatial_temporal_transformer_training_checkpoint",
        "project_commit": project_commit,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "manifest_sha256": manifest_sha256,
        "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
        "model_fingerprint": model_fingerprint(experiment.model),
        "model_config": asdict(experiment.model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return payload


def _validate_resume(
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    preprocessing_fingerprint: str,
    active_model_fingerprint: str,
) -> None:
    expected = {
        "manifest_sha256": manifest_sha256,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "model_fingerprint": active_model_fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Resume checkpoint {key} does not match the active run.")


def _glosses_by_class(
    records: Sequence[ManifestRecord], num_classes: int
) -> list[str]:
    result: list[str | None] = [None] * num_classes
    for record in records:
        current = result[record.class_index]
        if current is not None and current != record.gloss_name:
            raise ValueError(f"Class {record.class_index} maps to multiple gloss names.")
        result[record.class_index] = record.gloss_name
    if any(value is None for value in result):
        raise ValueError("At least one class has no gloss name.")
    return [str(value) for value in result]


if __name__ == "__main__":
    entrypoint()
