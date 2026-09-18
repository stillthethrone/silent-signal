"""Train the controlled 50-class RGB + graph-spatial-temporal Transformer demo."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from silent_signal.cli.train_videomaev2_demo import (
    _balanced_cap,
    _checkpoint_payload,
    _loader,
    _macro_f1_from_predictions,
    _make_dataset_class,
    _read_rows,
    _safe_video_path,
    _save_torch_atomic,
    _trailing_non_improving_epochs,
    _write_json_atomic,
    _write_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silent_signal.cli.train_dual_stream_demo",
        description=(
            "Frozen VideoMAE + RGB Transformer and Graph -> Spatial Transformer -> "
            "Temporal Transformer with cross-attention fusion."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--graph-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="OpenGVLab/VideoMAEv2-Base")
    parser.add_argument("--model-revision", default="0e826d7e85e39f9d951e331cd91c5c2d8142d385")
    parser.add_argument("--classes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--rgb-embedding-dim", type=int, default=128)
    parser.add_argument("--rgb-layers", type=int, default=1)
    parser.add_argument("--rgb-heads", type=int, default=4)
    parser.add_argument("--rgb-dropout", type=float, default=0.4)
    parser.add_argument("--pose-embedding-dim", type=int, default=128)
    parser.add_argument("--pose-graph-layers", type=int, default=1)
    parser.add_argument("--pose-spatial-layers", type=int, default=1)
    parser.add_argument("--pose-temporal-layers", type=int, default=1)
    parser.add_argument("--pose-heads", type=int, default=4)
    parser.add_argument("--pose-dropout", type=float, default=0.3)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--random-crop-scale-min", type=float, default=0.85)
    parser.add_argument("--color-jitter", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument("--overfit-monitor-patience", type=int, default=3)
    parser.add_argument("--overfit-min-epoch", type=int, default=6)
    parser.add_argument("--overfit-loss-gap", type=float, default=0.5)
    parser.add_argument("--overfit-top1-gap", type=float, default=0.2)
    parser.add_argument("--overfit-validation-loss-regression", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    return parser


def _validate_selected_manifest(
    rows: list[dict[str, str]], selection: dict[str, Any], class_count: int
) -> dict[int, str]:
    required = {"sample_id", "video_path", "class_index", "source_class_index", "split"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Baseline manifest is missing columns: {sorted(missing)}")
    if {str(row["split"]) for row in rows} != {"train", "validation", "test"}:
        raise RuntimeError("The selected manifest must preserve all three official splits.")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample_id in the selected baseline manifest.")
    classes = list(selection.get("classes", ()))
    if len(classes) != class_count:
        raise RuntimeError(f"Expected {class_count} selected classes, found {len(classes)}.")
    gloss_by_class = {int(item["class_index"]): str(item["gloss_name"]) for item in classes}
    if set(gloss_by_class) != set(range(class_count)):
        raise RuntimeError("Selected class indices must be contiguous from zero.")
    for split in ("train", "validation", "test"):
        found = {int(row["class_index"]) for row in rows if row["split"] == split}
        if found != set(range(class_count)):
            raise RuntimeError(f"Split {split} does not contain every selected class.")
    return gloss_by_class


def _make_dual_dataset_class(
    torch: Any,
    video_dataset_class: Any,
    read_graph_pose_cache: Any,
    pose_cache_path: Any,
):
    class DualStreamDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: list[dict[str, Any]],
            dataset_root: Path,
            graph_root: Path,
            *,
            expected_fingerprint: str,
            training: bool,
            random_crop_scale_min: float,
            color_jitter: float,
        ) -> None:
            self.rows = rows
            self.graph_root = graph_root
            self.expected_fingerprint = expected_fingerprint
            self.video_dataset = video_dataset_class(
                rows,
                dataset_root,
                training=training,
                random_crop_scale_min=random_crop_scale_min,
                color_jitter=color_jitter,
            )

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            video, label, sample_id, video_path = self.video_dataset[index]
            row = self.rows[index]
            graph = read_graph_pose_cache(
                pose_cache_path(self.graph_root, sample_id),
                expected_sample_id=sample_id,
                expected_fingerprint=self.expected_fingerprint,
            )
            if graph.class_index != int(row["source_class_index"]):
                raise RuntimeError(f"Graph source class mismatch for {sample_id}.")
            if graph.split != str(row["split"]):
                raise RuntimeError(f"Graph split mismatch for {sample_id}.")
            return (
                video,
                torch.from_numpy(graph.features),
                torch.from_numpy(graph.joint_mask),
                torch.from_numpy(graph.frame_mask),
                torch.from_numpy(graph.adjacency),
                label,
                sample_id,
                video_path,
            )

    return DualStreamDataset


def _run_epoch(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    criterion: Any,
    optimizer: Any | None,
    scaler: Any,
    phase: str,
    epoch: int,
    epochs: int,
    class_count: int,
    progress_every: int,
    start_batch: int = 0,
    checkpoint_every: int = 0,
    checkpoint_callback: Any | None = None,
    gradient_clip_norm: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    total_batches = len(loader)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(
        f"[{phase}] epoch {epoch + 1}/{epochs} START | batches={total_batches} | "
        f"resume_batch={start_batch} | device={device}",
        flush=True,
    )
    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue
        (
            videos,
            pose_features,
            joint_mask,
            frame_mask,
            adjacency,
            labels,
            sample_ids,
            video_paths,
        ) = batch
        videos = videos.to(device, non_blocking=True)
        pose_features = pose_features.to(device, non_blocking=True)
        joint_mask = joint_mask.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)
        adjacency = adjacency.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    pixel_values=videos,
                    pose_features=pose_features,
                    joint_mask=joint_mask,
                    frame_mask=frame_mask,
                    adjacency=adjacency,
                )
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
        batch_size = labels.numel()
        probabilities = logits.detach().softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((predicted == labels).sum())
        total_samples += batch_size
        for offset in range(batch_size):
            predictions.append(
                {
                    "sample_id": sample_ids[offset],
                    "video_path": video_paths[offset],
                    "true_class": int(labels[offset].detach().cpu()),
                    "pred_class": int(predicted[offset].detach().cpu()),
                    "confidence": float(confidence[offset].detach().cpu()),
                }
            )
        completed = batch_index + 1
        if completed % progress_every == 0 or completed == total_batches:
            elapsed = time.perf_counter() - started
            processed_batches = completed - start_batch
            eta = elapsed / max(processed_batches, 1) * max(0, total_batches - completed)
            print(
                f"[{phase}] epoch {epoch + 1}/{epochs} | batch {completed}/{total_batches} | "
                f"samples {total_samples} | loss {total_loss / total_samples:.4f} | "
                f"top1 {total_correct / total_samples:.3f} | ETA {eta / 60:.1f} min",
                flush=True,
            )
        if (
            training
            and checkpoint_every > 0
            and completed % checkpoint_every == 0
            and completed < total_batches
            and checkpoint_callback is not None
        ):
            checkpoint_callback(completed)
    if total_samples == 0:
        raise RuntimeError(f"No samples processed during {phase}.")
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": total_correct / total_samples,
        "macro_f1": _macro_f1_from_predictions(predictions, class_count),
        "samples": float(total_samples),
        "duration_seconds": time.perf_counter() - started,
    }, predictions


def _plot_history(plt: Any, history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="train")
    axes[0].plot(
        epochs,
        [item["validation_loss"] for item in history],
        marker="o",
        label="validation",
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        axes[1].plot(
            epochs,
            [item[f"{split}_top1"] for item in history],
            color=color,
            marker="o",
            label=f"{split} top-1",
        )
        axes[1].plot(
            epochs,
            [item[f"{split}_macro_f1"] for item in history],
            color=color,
            marker="x",
            linestyle="--",
            label=f"{split} macro-F1",
        )
    axes[1].set(title="Top-1 and macro-F1", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.suptitle(
        "VideoMAE + RGB Transformer || Graph + Spatial/Temporal Transformer — 50 classes"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _comparison_payload(
    baseline_report_path: Path | None, dual_evaluation: dict[str, Any]
) -> dict[str, Any] | None:
    if baseline_report_path is None or not baseline_report_path.is_file():
        return None
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_validation = baseline.get("evaluation", {}).get("validation", {})
    if not baseline_validation:
        return None
    return {
        "baseline_report": str(baseline_report_path),
        "baseline_validation_top1": baseline_validation.get("top1_accuracy"),
        "baseline_validation_macro_f1": baseline_validation.get("macro_f1"),
        "dual_validation_top1": dual_evaluation.get("top1_accuracy"),
        "dual_validation_macro_f1": dual_evaluation.get("macro_f1"),
        "top1_delta": float(dual_evaluation["top1_accuracy"])
        - float(baseline_validation["top1_accuracy"]),
        "macro_f1_delta": float(dual_evaluation["macro_f1"])
        - float(baseline_validation["macro_f1"]),
    }


def _is_overfit_epoch(
    record: dict[str, Any],
    *,
    best_validation_loss: float,
    min_epoch: int,
    loss_gap: float,
    top1_gap: float,
    validation_loss_regression: float,
) -> bool:
    """Return true only for a sustained generalization failure, not a healthy gap alone."""

    if int(record["epoch"]) < min_epoch:
        return False
    return bool(
        float(record["validation_loss"]) - float(record["train_loss"]) >= loss_gap
        and float(record["train_top1"]) - float(record["validation_top1"]) >= top1_gap
        and float(record["validation_loss"]) - best_validation_loss
        >= validation_loss_regression
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.classes != 50:
        raise RuntimeError("This controlled dual-stream demo is fixed to 50 classes.")
    if args.rgb_embedding_dim != args.pose_embedding_dim:
        raise ValueError("RGB and pose embedding dimensions must match for cross-attention.")
    if min(args.epochs, args.batch_size, args.progress_every) < 1:
        raise ValueError("epochs, batch-size, and progress-every must be positive.")
    if min(args.overfit_monitor_patience, args.overfit_min_epoch) < 0:
        raise ValueError("overfit patience and minimum epoch must be non-negative.")
    if min(
        args.overfit_loss_gap,
        args.overfit_top1_gap,
        args.overfit_validation_loss_regression,
    ) < 0:
        raise ValueError("overfit thresholds must be non-negative.")

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel

    from silent_signal.models.dual_stream import (
        DualStreamRecognizer,
        FrozenVideoMAETemporalEncoder,
    )
    from silent_signal.models.encoders.pose_spatiotemporal_transformer import (
        GraphSpatialTemporalTransformerEncoder,
    )
    from silent_signal.pose.cache import pose_cache_path
    from silent_signal.preprocessing.cache import read_graph_pose_cache

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)

    rows = _read_rows(args.manifest.resolve())
    selection = json.loads(args.selection_report.resolve().read_text(encoding="utf-8"))
    gloss_by_class = _validate_selected_manifest(rows, selection, args.classes)
    graph_report = json.loads(args.graph_report.resolve().read_text(encoding="utf-8"))
    if int(graph_report.get("failed", 0)) != 0:
        raise RuntimeError("Graph preparation report contains failures.")
    graph_fingerprint = str(graph_report.get("preprocessing_fingerprint", ""))
    if len(graph_fingerprint) != 64:
        raise RuntimeError("Graph preparation report has no valid preprocessing fingerprint.")
    missing_videos = [
        str(row["video_path"])
        for row in rows
        if not _safe_video_path(args.dataset_root, str(row["video_path"])).is_file()
    ]
    missing_graphs = [
        str(row["sample_id"])
        for row in rows
        if not pose_cache_path(args.graph_root, str(row["sample_id"])).is_file()
    ]
    if missing_videos:
        raise FileNotFoundError(f"Missing {len(missing_videos)} videos; first={missing_videos[0]}")
    if missing_graphs:
        raise FileNotFoundError(
            f"Missing {len(missing_graphs)} selected graph caches; first={missing_graphs[0]}"
        )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    print(
        "Selected paired samples:",
        {split: len(items) for split, items in split_rows.items()},
        flush=True,
    )
    print("Graph cache reuse PASS:", args.graph_root.resolve(), flush=True)

    config = AutoConfig.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    backbone = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        config=config,
        trust_remote_code=True,
    )
    rgb_encoder = FrozenVideoMAETemporalEncoder(
        backbone,
        embedding_dim=args.rgb_embedding_dim,
        layers=args.rgb_layers,
        heads=args.rgb_heads,
        dropout=args.rgb_dropout,
    )
    pose_encoder = GraphSpatialTemporalTransformerEncoder(
        input_dim=7,
        num_nodes=75,
        max_frames=64,
        embedding_dim=args.pose_embedding_dim,
        graph_layers=args.pose_graph_layers,
        spatial_layers=args.pose_spatial_layers,
        temporal_layers=args.pose_temporal_layers,
        heads=args.pose_heads,
        dropout=args.pose_dropout,
    )
    model = DualStreamRecognizer(
        rgb_encoder,
        pose_encoder,
        embedding_dim=args.rgb_embedding_dim,
        num_classes=args.classes,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
    ).to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}; parameters={total_parameters:,}; trainable={trainable:,}; "
        "VideoMAE_frozen=True; Graph/Spatial/Temporal_trainable=True; "
        "fusion=bidirectional_cross_attention",
        flush=True,
    )

    VideoDataset = _make_dataset_class(torch, cv2, np)
    DualDataset = _make_dual_dataset_class(
        torch,
        VideoDataset,
        read_graph_pose_cache,
        pose_cache_path,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_root / "last_checkpoint.pt"
    best_checkpoint = output_root / "best_checkpoint.pt"
    metadata = {
        "architecture": (
            "frozen VideoMAE -> compact RGB Transformer || cached whole-body keypoints -> "
            "Graph Encoder -> Spatial Transformer -> Temporal Transformer -> "
            "bidirectional multi-head Cross-Attention -> Feature Fusion -> classifier"
        ),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "class_count": args.classes,
        "videomae_frozen": True,
        "graph_cache_frozen_features": True,
        "pose_encoder_trainable": True,
        "graph_preprocessing_fingerprint": graph_fingerprint,
        "rgb_embedding_dim": args.rgb_embedding_dim,
        "rgb_layers": args.rgb_layers,
        "rgb_heads": args.rgb_heads,
        "rgb_dropout": args.rgb_dropout,
        "pose_embedding_dim": args.pose_embedding_dim,
        "pose_graph_layers": args.pose_graph_layers,
        "pose_spatial_layers": args.pose_spatial_layers,
        "pose_temporal_layers": args.pose_temporal_layers,
        "pose_heads": args.pose_heads,
        "pose_dropout": args.pose_dropout,
        "fusion_heads": args.fusion_heads,
        "fusion_dropout": args.fusion_dropout,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "overfit_monitor": {
            "patience": args.overfit_monitor_patience,
            "min_epoch": args.overfit_min_epoch,
            "loss_gap": args.overfit_loss_gap,
            "top1_gap": args.overfit_top1_gap,
            "validation_loss_regression": args.overfit_validation_loss_regression,
        },
    }
    start_epoch = 0
    start_batch = 0
    best_validation_loss = float("inf")
    bad_epochs = 0
    overfit_epochs = 0
    stopped_early = False
    stop_reason: str | None = None
    history: list[dict[str, Any]] = []
    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata") != metadata:
            raise RuntimeError("Existing dual-stream checkpoint has different metadata.")
        model.load_trainable_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("next_batch", 0))
        best_validation_loss = float(checkpoint["best_validation_loss"])
        history = list(checkpoint.get("history", ()))
        bad_epochs = int(
            checkpoint.get(
                "early_stopping_bad_epochs",
                _trailing_non_improving_epochs(history, args.early_stopping_min_delta),
            )
        )
        overfit_epochs = int(checkpoint.get("overfit_bad_epochs", 0))
        print(
            f"RESUME: epoch {start_epoch + 1}, batch {start_batch}; "
            f"early-stop wait={bad_epochs}/{args.early_stopping_patience}; "
            f"overfit wait={overfit_epochs}/{args.overfit_monitor_patience}",
            flush=True,
        )

    def dataset(items: list[dict[str, Any]], *, training: bool):
        return DualDataset(
            items,
            args.dataset_root,
            args.graph_root,
            expected_fingerprint=graph_fingerprint,
            training=training,
            random_crop_scale_min=args.random_crop_scale_min,
            color_jitter=args.color_jitter,
        )

    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        train_rows = _balanced_cap(
            split_rows["train"],
            args.max_train_batches * args.batch_size,
            args.seed + epoch,
        )
        train_loader = _loader(
            torch,
            dataset(train_rows, training=True),
            args.batch_size,
            args.num_workers,
        )

        def save_progress(
            next_batch: int,
            current_epoch: int = epoch,
            current_best: float = best_validation_loss,
            current_bad_epochs: int = bad_epochs,
            current_overfit_epochs: int = overfit_epochs,
        ) -> None:
            progress_payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=current_epoch,
                next_batch=next_batch,
                best_validation_loss=current_best,
                early_stopping_bad_epochs=current_bad_epochs,
                history=history,
                metadata=metadata,
            )
            progress_payload["overfit_bad_epochs"] = current_overfit_epochs
            _save_torch_atomic(torch, last_checkpoint, progress_payload)
            print(
                f"[checkpoint] epoch {current_epoch + 1}, next batch {next_batch}",
                flush=True,
            )

        train_metrics, _ = _run_epoch(
            torch=torch,
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            phase="train",
            epoch=epoch,
            epochs=args.epochs,
            class_count=args.classes,
            progress_every=args.progress_every,
            start_batch=start_batch if epoch == start_epoch else 0,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=save_progress,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        validation_rows = _balanced_cap(
            split_rows["validation"],
            args.max_eval_batches * args.batch_size,
            args.seed,
            shuffle=False,
        )
        validation_metrics, _ = _run_epoch(
            torch=torch,
            model=model,
            loader=_loader(
                torch,
                dataset(validation_rows, training=False),
                args.batch_size,
                args.num_workers,
            ),
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            phase="validation",
            epoch=epoch,
            epochs=args.epochs,
            class_count=args.classes,
            progress_every=args.progress_every,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_samples": int(train_metrics["samples"]),
            "validation_loss": validation_metrics["loss"],
            "validation_top1": validation_metrics["top1_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_samples": int(validation_metrics["samples"]),
        }
        improved = validation_metrics["loss"] < (
            best_validation_loss - args.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = validation_metrics["loss"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        record["loss_generalization_gap"] = (
            record["validation_loss"] - record["train_loss"]
        )
        record["top1_generalization_gap"] = record["train_top1"] - record["validation_top1"]
        overfit_signal = _is_overfit_epoch(
            record,
            best_validation_loss=best_validation_loss,
            min_epoch=args.overfit_min_epoch,
            loss_gap=args.overfit_loss_gap,
            top1_gap=args.overfit_top1_gap,
            validation_loss_regression=args.overfit_validation_loss_regression,
        )
        overfit_epochs = overfit_epochs + 1 if overfit_signal else 0
        record["improved"] = improved
        record["early_stopping_bad_epochs"] = bad_epochs
        record["overfit_signal"] = overfit_signal
        record["overfit_bad_epochs"] = overfit_epochs
        history.append(record)
        payload = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch + 1,
            next_batch=0,
            best_validation_loss=best_validation_loss,
            early_stopping_bad_epochs=bad_epochs,
            history=history,
            metadata=metadata,
        )
        payload["overfit_bad_epochs"] = overfit_epochs
        if improved:
            _save_torch_atomic(torch, best_checkpoint, payload)
        _save_torch_atomic(torch, last_checkpoint, payload)
        _write_json_atomic(output_root / "history.json", {"history": history})
        _plot_history(plt, history, output_root / "training_curves.png")
        print(
            f"[epoch {epoch + 1}] train top1={record['train_top1']:.3f}, "
            f"macro-F1={record['train_macro_f1']:.3f}; validation "
            f"top1={record['validation_top1']:.3f}, "
            f"macro-F1={record['validation_macro_f1']:.3f}; best={improved}; "
            f"gap(loss/top1)={record['loss_generalization_gap']:.3f}/"
            f"{record['top1_generalization_gap']:.3f}; "
            f"early-stop wait={bad_epochs}/{args.early_stopping_patience}; "
            f"overfit wait={overfit_epochs}/{args.overfit_monitor_patience}",
            flush=True,
        )
        start_batch = 0
        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            stopped_early = True
            stop_reason = "validation_loss_no_improvement"
            print(
                f"EARLY STOP at epoch {epoch + 1}: validation loss did not improve; "
                "restoring best checkpoint.",
                flush=True,
            )
            break
        if (
            args.overfit_monitor_patience > 0
            and overfit_epochs >= args.overfit_monitor_patience
        ):
            stopped_early = True
            stop_reason = "sustained_overfit_signal"
            print(
                f"OVERFIT STOP at epoch {epoch + 1}: loss/top-1 gaps remained large while "
                "validation loss regressed from its best value; restoring best checkpoint.",
                flush=True,
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("No best dual-stream checkpoint was created.")
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(best["model_state"])
    model.to(device)
    evaluation: dict[str, Any] = {}
    for split in ["validation"] + (["test"] if args.run_test else []):
        evaluation_rows = _balanced_cap(
            split_rows[split],
            args.max_eval_batches * args.batch_size,
            args.seed,
            shuffle=False,
        )
        metrics, predictions = _run_epoch(
            torch=torch,
            model=model,
            loader=_loader(
                torch,
                dataset(evaluation_rows, training=False),
                args.batch_size,
                args.num_workers,
            ),
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            phase=f"final-{split}",
            epoch=max(0, len(history) - 1),
            epochs=max(1, len(history)),
            class_count=args.classes,
            progress_every=args.progress_every,
        )
        prediction_path = output_root / f"{split}_predictions.csv"
        _write_predictions(prediction_path, predictions, gloss_by_class, split)
        evaluation[split] = {
            **metrics,
            "samples": int(metrics["samples"]),
            "available_samples": len(split_rows[split]),
            "predictions": str(prediction_path),
            "partial": int(metrics["samples"]) < len(split_rows[split]),
        }
    comparison = _comparison_payload(args.baseline_report, evaluation["validation"])
    report = {
        "schema_version": 1,
        "state": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "study_stage": "paired 50-class document-specified dual-stream improvement demo",
        "split_policy": "same official split and same 50 samples/classes as RGB baseline",
        "model": metadata,
        "device": str(device),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["best_validation_loss"]),
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "triggered": stopped_early,
            "reason": stop_reason,
        },
        "overfit_monitor": {
            "patience": args.overfit_monitor_patience,
            "min_epoch": args.overfit_min_epoch,
            "loss_gap": args.overfit_loss_gap,
            "top1_gap": args.overfit_top1_gap,
            "validation_loss_regression": args.overfit_validation_loss_regression,
            "final_wait": overfit_epochs,
            "triggered": stop_reason == "sustained_overfit_signal",
        },
        "history": history,
        "evaluation": evaluation,
        "baseline_comparison": comparison,
        "duration_seconds": time.perf_counter() - run_started,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "training_curves": str(output_root / "training_curves.png"),
            "selection_report": str(args.selection_report.resolve()),
            "selected_manifest": str(args.manifest.resolve()),
            "graph_report": str(args.graph_report.resolve()),
        },
    }
    _write_json_atomic(output_root / "dual_stream_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
