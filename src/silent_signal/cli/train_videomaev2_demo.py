"""Train a bounded RGB-only VideoMAE V2 baseline on an official ASL split.

Heavy video and training dependencies are imported only inside ``main`` so the
core package remains usable without the optional Colab stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silent_signal.cli.train_videomaev2_demo",
        description="Bounded VideoMAE V2 RGB-only research baseline.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="OpenGVLab/VideoMAEv2-Base")
    parser.add_argument("--model-revision", default="0e826d7e85e39f9d951e331cd91c5c2d8142d385")
    parser.add_argument("--classes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--max-eval-batches", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_torch_atomic(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    required = {"sample_id", "video_path", "gloss_name", "class_index", "split"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def _select_demo_rows(
    rows: list[dict[str, str]], selection_path: Path, class_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ranked = list(selection.get("classes", ()))[:class_count]
    if len(ranked) != class_count:
        raise RuntimeError(f"Selection report has fewer than {class_count} ranked classes.")
    if [item["rank"] for item in ranked] != list(range(1, class_count + 1)):
        raise RuntimeError("The demo must use the first contiguous ASL-LEX ranks.")
    if [item["subset_class_index"] for item in ranked] != list(range(class_count)):
        raise RuntimeError("Top-ranked class indices are not contiguous from zero.")

    selected_indices = {int(item["subset_class_index"]) for item in ranked}
    selected: list[dict[str, Any]] = []
    for row in rows:
        class_index = int(row["class_index"])
        if class_index in selected_indices:
            item: dict[str, Any] = dict(row)
            item["class_index"] = class_index
            selected.append(item)

    splits = {"train", "validation", "test"}
    found_splits = {str(row["split"]) for row in selected}
    if found_splits != splits:
        raise RuntimeError(f"Expected official splits {splits}, found {found_splits}.")
    sample_ids = [str(row["sample_id"]) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample_id found in the filtered manifest.")
    split_ids = {
        split: {str(row["sample_id"]) for row in selected if row["split"] == split}
        for split in sorted(splits)
    }
    if split_ids["train"] & split_ids["validation"]:
        raise RuntimeError("Leakage between train and validation.")
    if split_ids["train"] & split_ids["test"]:
        raise RuntimeError("Leakage between train and test.")
    if split_ids["validation"] & split_ids["test"]:
        raise RuntimeError("Leakage between validation and test.")

    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in selected:
        counts[(int(row["class_index"]), str(row["split"]))] += 1
    for class_item in ranked:
        class_index = int(class_item["subset_class_index"])
        for split in splits:
            if counts[(class_index, split)] == 0:
                raise RuntimeError(
                    f"Top-20 word {class_item['gloss_name']!r} has no official {split} clips."
                )
    return selected, ranked


def _write_demo_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _balanced_cap(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        result = list(rows)
        random.Random(seed).shuffle(result)
        return result
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["class_index"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    result: list[dict[str, Any]] = []
    active = sorted(groups)
    while active and len(result) < maximum:
        next_active: list[int] = []
        for class_index in active:
            group = groups[class_index]
            if group and len(result) < maximum:
                result.append(group.pop())
            if group:
                next_active.append(class_index)
        active = next_active
    rng.shuffle(result)
    return result


def _safe_video_path(dataset_root: Path, relative: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Video path escapes dataset root: {relative}") from exc
    return candidate


def _make_dataset_class(torch: Any, cv2: Any, np: Any):
    class VideoDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: list[dict[str, Any]],
            dataset_root: Path,
            *,
            frames: int = 16,
            size: int = 224,
            training: bool = False,
        ) -> None:
            self.rows = rows
            self.dataset_root = dataset_root
            self.frames = frames
            self.size = size
            self.training = training

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            path = _safe_video_path(self.dataset_root, str(row["video_path"]))
            video = self._decode(path)
            return (
                video,
                int(row["class_index"]),
                str(row["sample_id"]),
                str(row["video_path"]),
            )

        def _decode(self, path: Path):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                capture.release()
                raise RuntimeError(f"Video has no frames: {path}")
            edges = np.linspace(0, frame_count, self.frames + 1)
            if self.training:
                targets = [
                    min(frame_count - 1, int(np.random.uniform(edges[i], edges[i + 1])))
                    for i in range(self.frames)
                ]
            else:
                targets = [
                    min(frame_count - 1, int((edges[i] + edges[i + 1]) / 2))
                    for i in range(self.frames)
                ]
            target_set = set(targets)
            decoded: dict[int, Any] = {}
            position = 0
            while position <= targets[-1]:
                ok, frame = capture.read()
                if not ok:
                    break
                if position in target_set:
                    decoded[position] = self._resize_crop(frame)
                position += 1
            capture.release()
            if not decoded:
                raise RuntimeError(f"Failed to decode frames: {path}")
            available = sorted(decoded)
            frames = [
                decoded[min(available, key=lambda value: abs(value - target))] for target in targets
            ]
            array = np.stack(frames).astype("float32") / 255.0
            array = (array - 0.5) / 0.5
            return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()

        def _resize_crop(self, bgr: Any):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            scale = self.size / min(height, width)
            resized = cv2.resize(
                rgb,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            top = max(0, (resized.shape[0] - self.size) // 2)
            left = max(0, (resized.shape[1] - self.size) // 2)
            return resized[top : top + self.size, left : left + self.size]

    return VideoDataset


def _loader(torch: Any, dataset: Any, batch_size: int, workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    *,
    epoch: int,
    next_batch: int,
    best_validation_loss: float,
    history: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "metadata": metadata,
    }


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
    progress_every: int,
    start_batch: int = 0,
    checkpoint_every: int = 0,
    checkpoint_callback: Any | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    total_batches = len(loader)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_index, (videos, labels, sample_ids, video_paths) in enumerate(loader):
        if batch_index < start_batch:
            continue
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixel_values=videos)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
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
            remaining = max(0, total_batches - completed)
            eta = elapsed / max(processed_batches, 1) * remaining
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
        "samples": float(total_samples),
        "duration_seconds": time.perf_counter() - started,
    }, predictions


def _write_predictions(
    path: Path, predictions: list[dict[str, Any]], gloss_by_class: dict[int, str], split: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in predictions:
        rows.append(
            {
                **item,
                "true_gloss": gloss_by_class[item["true_class"]],
                "pred_gloss": gloss_by_class[item["pred_class"]],
                "split": split,
            }
        )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_history(plt: Any, history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="train")
    axes[0].plot(
        epochs, [item["validation_loss"] for item in history], marker="o", label="validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [item["train_top1"] for item in history], marker="o", label="train")
    axes[1].plot(
        epochs, [item["validation_top1"] for item in history], marker="o", label="validation"
    )
    axes[1].set(title="Top-1 accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.suptitle("VideoMAE V2 RGB-only demo — not a final benchmark")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.classes != 20:
        raise RuntimeError("This bounded research notebook is intentionally fixed to 20 classes.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel

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

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = _read_rows(args.manifest.resolve())
    selected_rows, ranked = _select_demo_rows(
        manifest_rows, args.selection_report.resolve(), args.classes
    )
    split_rows = {
        split: [row for row in selected_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    missing_videos = [
        str(row["video_path"])
        for row in selected_rows
        if not _safe_video_path(args.dataset_root, str(row["video_path"])).is_file()
    ]
    if missing_videos:
        raise FileNotFoundError(
            f"{len(missing_videos)} selected videos are missing; first: {missing_videos[0]}"
        )

    demo_manifest = output_root / "demo20_manifest.csv"
    _write_demo_manifest(demo_manifest, selected_rows)
    manifest_sha = _sha256(demo_manifest)
    selection_sha = _sha256(args.selection_report.resolve())
    gloss_by_class = {int(item["subset_class_index"]): str(item["gloss_name"]) for item in ranked}
    selected_words = []
    for item in ranked:
        class_index = int(item["subset_class_index"])
        counts = {
            split: sum(int(row["class_index"]) == class_index for row in split_rows[split])
            for split in split_rows
        }
        total = sum(counts.values())
        selected_words.append(
            {
                "rank": int(item["rank"]),
                "class_index": class_index,
                "gloss_name": item["gloss_name"],
                "sign_frequency_mean": item["sign_frequency_mean"],
                "counts": counts,
                "percentages": {
                    split: round(count / total * 100, 2) for split, count in counts.items()
                },
            }
        )
    selection_payload = {
        "schema_version": 1,
        "warning": "DEMO 20 classes; do not compare these metrics with the final 200-class study.",
        "selection": "first 20 ASL-LEX frequency ranks from the frozen top-200 report",
        "split_policy": "official ASL Citizen train/validation/test; never re-split",
        "manifest_sha256": manifest_sha,
        "source_selection_sha256": selection_sha,
        "clips": {split: len(rows) for split, rows in split_rows.items()},
        "classes": selected_words,
    }
    _write_json_atomic(output_root / "selected_20_words.json", selection_payload)
    print("\n20 TỪ DEMO (xếp theo ASL-LEX SignFrequency):", flush=True)
    for item in selected_words:
        counts = item["counts"]
        percentages = item["percentages"]
        print(
            f"{item['rank']:02d}. {item['gloss_name']:<24} | "
            f"train {counts['train']:>3} ({percentages['train']:>5.1f}%) | "
            f"val {counts['validation']:>3} ({percentages['validation']:>5.1f}%) | "
            f"test {counts['test']:>3} ({percentages['test']:>5.1f}%)",
            flush=True,
        )
    print("Split isolation: PASS — no sample_id overlap.\n", flush=True)

    print(f"Loading {args.model_id}@{args.model_revision} ...", flush=True)
    config = AutoConfig.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        config=config,
        trust_remote_code=True,
    )
    model.model.reset_classifier(args.classes)
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.model.head.parameters():
            parameter.requires_grad = True
    model.to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}; parameters={total_parameters:,}; trainable={trainable:,}; "
        f"freeze_backbone={args.freeze_backbone}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    VideoDataset = _make_dataset_class(torch, cv2, np)
    last_checkpoint = output_root / "last_checkpoint.pt"
    best_checkpoint = output_root / "best_checkpoint.pt"
    metadata = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "class_count": args.classes,
        "manifest_sha256": manifest_sha,
        "selection_sha256": selection_sha,
        "freeze_backbone": args.freeze_backbone,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
    }
    start_epoch = 0
    start_batch = 0
    best_validation_loss = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata") != metadata:
            raise RuntimeError("Existing checkpoint metadata differs from this run configuration.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("next_batch", 0))
        best_validation_loss = float(checkpoint["best_validation_loss"])
        history = list(checkpoint.get("history", ()))
        print(
            f"RESUME: epoch {start_epoch + 1}, batch {start_batch}; history={len(history)} epochs",
            flush=True,
        )

    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        maximum_train = args.max_train_batches * args.batch_size
        epoch_rows = _balanced_cap(split_rows["train"], maximum_train, args.seed + epoch)
        train_dataset = VideoDataset(epoch_rows, args.dataset_root, training=True)
        train_loader = _loader(torch, train_dataset, args.batch_size, args.num_workers)

        def save_progress(
            next_batch: int,
            current_epoch: int = epoch,
            current_best: float = best_validation_loss,
        ) -> None:
            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=current_epoch,
                next_batch=next_batch,
                best_validation_loss=current_best,
                history=history,
                metadata=metadata,
            )
            _save_torch_atomic(torch, last_checkpoint, payload)
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
            progress_every=args.progress_every,
            start_batch=start_batch if epoch == start_epoch else 0,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=save_progress,
        )
        validation_rows = _balanced_cap(
            split_rows["validation"], args.max_eval_batches * args.batch_size, args.seed
        )
        validation_loader = _loader(
            torch,
            VideoDataset(validation_rows, args.dataset_root, training=False),
            args.batch_size,
            args.num_workers,
        )
        validation_metrics, _ = _run_epoch(
            torch=torch,
            model=model,
            loader=validation_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            phase="validation",
            epoch=epoch,
            epochs=args.epochs,
            progress_every=args.progress_every,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "train_samples": int(train_metrics["samples"]),
            "validation_loss": validation_metrics["loss"],
            "validation_top1": validation_metrics["top1_accuracy"],
            "validation_samples": int(validation_metrics["samples"]),
        }
        history.append(record)
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            _save_torch_atomic(
                torch,
                best_checkpoint,
                _checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    next_batch=0,
                    best_validation_loss=best_validation_loss,
                    history=history,
                    metadata=metadata,
                ),
            )
        _save_torch_atomic(
            torch,
            last_checkpoint,
            _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch + 1,
                next_batch=0,
                best_validation_loss=best_validation_loss,
                history=history,
                metadata=metadata,
            ),
        )
        _write_json_atomic(output_root / "history.json", {"history": history})
        _plot_history(plt, history, output_root / "training_curves.png")
        print(
            f"[epoch {epoch + 1}] train top1={record['train_top1']:.3f}; "
            f"validation top1={record['validation_top1']:.3f}; best={improved}",
            flush=True,
        )
        start_batch = 0

    if not best_checkpoint.is_file():
        raise RuntimeError("No best checkpoint was created.")
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    model.to(device)

    evaluation: dict[str, Any] = {}
    requested_splits = ["validation"] + (["test"] if args.run_test else [])
    for split in requested_splits:
        evaluation_rows = _balanced_cap(
            split_rows[split], args.max_eval_batches * args.batch_size, args.seed
        )
        evaluation_loader = _loader(
            torch,
            VideoDataset(evaluation_rows, args.dataset_root, training=False),
            args.batch_size,
            args.num_workers,
        )
        metrics, predictions = _run_epoch(
            torch=torch,
            model=model,
            loader=evaluation_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            phase=f"final-{split}",
            epoch=max(0, len(history) - 1),
            epochs=max(1, len(history)),
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

    report = {
        "schema_version": 1,
        "state": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "study_stage": "bounded 20-class RGB-only demo; not the final 200-class benchmark",
        "split_policy": "official ASL Citizen train/validation/test; test excluded from tuning",
        "model": metadata,
        "device": str(device),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "max_train_batches_per_epoch": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "history": history,
        "evaluation": evaluation,
        "duration_seconds": time.perf_counter() - run_started,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "selected_words": str(output_root / "selected_20_words.json"),
            "training_curves": str(output_root / "training_curves.png"),
        },
    }
    _write_json_atomic(output_root / "baseline_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
