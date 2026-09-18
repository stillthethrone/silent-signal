"""Train a full-data RGB Transformer baseline on official ASL splits.

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
    parser.add_argument("--classes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Zero uses every official training clip.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Zero uses every official validation/test clip.",
    )
    parser.add_argument("--rgb-embedding-dim", type=int, default=256)
    parser.add_argument("--rgb-layers", type=int, default=2)
    parser.add_argument("--rgb-heads", type=int, default=8)
    parser.add_argument("--rgb-dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
        help="Stop after this many validation epochs without loss improvement; zero disables.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
    source_ranked = list(selection.get("classes", ()))
    source_counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        source_counts[(int(row["class_index"]), str(row["split"]))] += 1
    required_splits = ("train", "validation", "test")
    eligible = [
        item
        for item in source_ranked
        if all(
            source_counts[(int(item["subset_class_index"]), split)] > 0 for split in required_splits
        )
    ]
    ranked = []
    for demo_class_index, item in enumerate(eligible[:class_count]):
        ranked.append(
            {
                **item,
                "source_subset_class_index": int(item["subset_class_index"]),
                "demo_class_index": demo_class_index,
            }
        )
    if len(ranked) != class_count:
        raise RuntimeError(
            f"Only {len(ranked)} ranked classes contain clips in every official split; "
            f"{class_count} are required."
        )
    source_to_demo = {
        int(item["source_subset_class_index"]): int(item["demo_class_index"]) for item in ranked
    }
    selected: list[dict[str, Any]] = []
    for row in rows:
        source_class_index = int(row["class_index"])
        if source_class_index in source_to_demo:
            item: dict[str, Any] = dict(row)
            item["source_class_index"] = source_class_index
            item["class_index"] = source_to_demo[source_class_index]
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
        class_index = int(class_item["demo_class_index"])
        for split in splits:
            if counts[(class_index, split)] == 0:
                raise RuntimeError(
                    f"Selected word {class_item['gloss_name']!r} has no official {split} clips."
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


def _balanced_cap(
    rows: list[dict[str, Any]], maximum: int, seed: int, *, shuffle: bool = True
) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        result = list(rows)
        if shuffle:
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
    if shuffle:
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


def _make_rgb_temporal_model_class(torch: Any):
    class FrozenVideoMAEWithRGBTransformer(torch.nn.Module):
        """Frozen VideoMAE V2 tube tokens followed by a trainable temporal encoder."""

        def __init__(
            self,
            backbone: Any,
            *,
            class_count: int,
            embedding_dim: int,
            layers: int,
            heads: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if embedding_dim % heads:
                raise ValueError("rgb-embedding-dim must be divisible by rgb-heads.")
            self.backbone = backbone
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            visual = self.backbone.model
            source_dim = int(visual.embed_dim)
            temporal_tokens = int(visual.patch_embed.num_patches) // (
                int(visual.patch_embed.img_size[0] // visual.patch_embed.patch_size[0])
                * int(visual.patch_embed.img_size[1] // visual.patch_embed.patch_size[1])
            )
            self.temporal_tokens = temporal_tokens
            self.projection = torch.nn.Linear(source_dim, embedding_dim)
            self.temporal_position = torch.nn.Parameter(
                torch.zeros(1, temporal_tokens, embedding_dim)
            )
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.rgb_transformer = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=layers, enable_nested_tensor=False
            )
            self.output_norm = torch.nn.LayerNorm(embedding_dim)
            self.classifier = torch.nn.Linear(embedding_dim, class_count)
            torch.nn.init.trunc_normal_(self.temporal_position, std=0.02)

        def train(self, mode: bool = True):
            super().train(mode)
            self.backbone.eval()
            return self

        def _frozen_temporal_tokens(self, pixel_values: Any):
            visual = self.backbone.model
            with torch.no_grad():
                tokens = visual.patch_embed(pixel_values)
                if visual.pos_embed is not None:
                    position = visual.pos_embed.expand(tokens.size(0), -1, -1)
                    tokens = tokens + position.type_as(tokens).to(tokens.device).detach()
                tokens = visual.pos_drop(tokens)
                for block in visual.blocks:
                    tokens = block(tokens)
                temporal = pixel_values.shape[2] // int(visual.tubelet_size)
                if tokens.shape[1] % temporal:
                    raise RuntimeError("VideoMAE token count is not divisible by temporal tubes.")
                spatial = tokens.shape[1] // temporal
                tokens = tokens.reshape(tokens.shape[0], temporal, spatial, tokens.shape[2])
                tokens = tokens.mean(dim=2)
                if visual.fc_norm is not None:
                    tokens = visual.fc_norm(tokens)
                else:
                    tokens = visual.norm(tokens)
            return tokens

        def forward(self, pixel_values: Any):
            tokens = self._frozen_temporal_tokens(pixel_values)
            tokens = self.projection(tokens)
            if tokens.shape[1] != self.temporal_tokens:
                raise RuntimeError(
                    f"Expected {self.temporal_tokens} temporal tokens, got {tokens.shape[1]}."
                )
            tokens = tokens + self.temporal_position
            tokens = self.rgb_transformer(tokens)
            feature = self.output_norm(tokens).mean(dim=1)
            return self.classifier(feature)

        def trainable_state_dict(self) -> dict[str, Any]:
            return {
                name: value
                for name, value in self.state_dict().items()
                if not name.startswith("backbone.")
            }

        def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
            incompatible = self.load_state_dict(state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            non_backbone_missing = [
                name for name in incompatible.missing_keys if not name.startswith("backbone.")
            ]
            if unexpected or non_backbone_missing:
                raise RuntimeError(
                    "Invalid RGB Transformer checkpoint: "
                    f"unexpected={unexpected}, missing={non_backbone_missing}"
                )

    return FrozenVideoMAEWithRGBTransformer


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
    early_stopping_bad_epochs: int,
    history: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.trainable_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "best_validation_loss": best_validation_loss,
        "early_stopping_bad_epochs": early_stopping_bad_epochs,
        "history": history,
        "metadata": metadata,
    }


def _trailing_non_improving_epochs(
    history: list[dict[str, Any]], min_delta: float
) -> int:
    best = float("inf")
    bad_epochs = 0
    for item in history:
        validation_loss = float(item["validation_loss"])
        if validation_loss < best - min_delta:
            best = validation_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
    return bad_epochs


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


def _plot_history(
    plt: Any, history: list[dict[str, Any]], path: Path, class_count: int
) -> None:
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
    figure.suptitle(
        f"Frozen VideoMAE V2 + trainable RGB Transformer — {class_count}-class demo"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.classes != 50:
        raise RuntimeError("This research demo is intentionally fixed to 50 classes.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min-delta must be non-negative.")

    # This is a PyTorch-only pipeline. Colab also preinstalls TensorFlow/JAX; letting
    # Transformers probe those optional backends can import a JAX build that is
    # incompatible with the runtime NumPy even though neither backend is used here.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_JAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

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

    manifest_root = output_root / "manifests"
    demo_manifest = manifest_root / "all.csv"
    _write_demo_manifest(demo_manifest, selected_rows)
    split_manifest_paths = {}
    for split, rows in split_rows.items():
        split_path = manifest_root / f"{split}.csv"
        _write_demo_manifest(split_path, rows)
        split_manifest_paths[split] = split_path
    manifest_sha = _sha256(demo_manifest)
    selection_sha = _sha256(args.selection_report.resolve())
    gloss_by_class = {int(item["demo_class_index"]): str(item["gloss_name"]) for item in ranked}
    selected_words = []
    for item in ranked:
        class_index = int(item["demo_class_index"])
        counts = {
            split: sum(int(row["class_index"]) == class_index for row in split_rows[split])
            for split in split_rows
        }
        total = sum(counts.values())
        selected_words.append(
            {
                "rank": int(item["rank"]),
                "class_index": class_index,
                "source_subset_class_index": int(item["source_subset_class_index"]),
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
        "warning": "DEMO 50 classes; do not compare these metrics with the final 200-class study.",
        "selection": (
            "highest-ranked 50 ASL-LEX classes from the frozen top-200 report "
            "that contain clips in every official split"
        ),
        "split_policy": "official ASL Citizen train/validation/test; never re-split",
        "manifest_sha256": manifest_sha,
        "source_selection_sha256": selection_sha,
        "clips": {split: len(rows) for split, rows in split_rows.items()},
        "manifests": {
            "all": str(demo_manifest),
            **{split: str(path) for split, path in split_manifest_paths.items()},
        },
        "classes": selected_words,
    }
    _write_json_atomic(output_root / "selected_50_words.json", selection_payload)
    print("\n50 TỪ DEMO (xếp theo ASL-LEX SignFrequency):", flush=True)
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
    print("Manifest tách riêng trên Drive:", flush=True)
    for split, path in split_manifest_paths.items():
        print(f"- {split}: {path} ({len(split_rows[split])} clips)", flush=True)

    print(f"Loading {args.model_id}@{args.model_revision} ...", flush=True)
    config = AutoConfig.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True
    )
    backbone = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        config=config,
        trust_remote_code=True,
    )
    RGBTemporalModel = _make_rgb_temporal_model_class(torch)
    model = RGBTemporalModel(
        backbone,
        class_count=args.classes,
        embedding_dim=args.rgb_embedding_dim,
        layers=args.rgb_layers,
        heads=args.rgb_heads,
        dropout=args.rgb_dropout,
    )
    model.to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}; parameters={total_parameters:,}; trainable={trainable:,}; "
        f"VideoMAE_frozen=True; RGB_Transformer={args.rgb_layers} layers",
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
        "architecture": "frozen VideoMAE V2 -> spatial pooling -> RGB Transformer -> classifier",
        "videomae_frozen": True,
        "rgb_embedding_dim": args.rgb_embedding_dim,
        "rgb_layers": args.rgb_layers,
        "rgb_heads": args.rgb_heads,
        "rgb_dropout": args.rgb_dropout,
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
    early_stopping_bad_epochs = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata") != metadata:
            raise RuntimeError("Existing checkpoint metadata differs from this run configuration.")
        model.load_trainable_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("next_batch", 0))
        best_validation_loss = float(checkpoint["best_validation_loss"])
        history = list(checkpoint.get("history", ()))
        early_stopping_bad_epochs = int(
            checkpoint.get(
                "early_stopping_bad_epochs",
                _trailing_non_improving_epochs(history, args.early_stopping_min_delta),
            )
        )
        print(
            f"RESUME: epoch {start_epoch + 1}, batch {start_batch}; "
            f"history={len(history)} epochs; early-stop wait={early_stopping_bad_epochs}/"
            f"{args.early_stopping_patience}",
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
            current_bad_epochs: int = early_stopping_bad_epochs,
        ) -> None:
            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=current_epoch,
                next_batch=next_batch,
                best_validation_loss=current_best,
                early_stopping_bad_epochs=current_bad_epochs,
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
            split_rows["validation"],
            args.max_eval_batches * args.batch_size,
            args.seed,
            shuffle=False,
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
        improved = (
            validation_metrics["loss"]
            < best_validation_loss - args.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = validation_metrics["loss"]
            early_stopping_bad_epochs = 0
        else:
            early_stopping_bad_epochs += 1
        record["improved"] = improved
        record["early_stopping_bad_epochs"] = early_stopping_bad_epochs
        history.append(record)
        if improved:
            _save_torch_atomic(
                torch,
                best_checkpoint,
                _checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    next_batch=0,
                    best_validation_loss=best_validation_loss,
                    early_stopping_bad_epochs=early_stopping_bad_epochs,
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
                early_stopping_bad_epochs=early_stopping_bad_epochs,
                history=history,
                metadata=metadata,
            ),
        )
        _write_json_atomic(output_root / "history.json", {"history": history})
        _plot_history(plt, history, output_root / "training_curves.png", args.classes)
        print(
            f"[epoch {epoch + 1}] train top1={record['train_top1']:.3f}; "
            f"validation top1={record['validation_top1']:.3f}; best={improved}; "
            f"early-stop wait={early_stopping_bad_epochs}/{args.early_stopping_patience}",
            flush=True,
        )
        start_batch = 0
        if (
            args.early_stopping_patience > 0
            and early_stopping_bad_epochs >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"EARLY STOP at epoch {epoch + 1}: validation loss did not improve by "
                f"at least {args.early_stopping_min_delta:g} for "
                f"{early_stopping_bad_epochs} consecutive epochs. "
                f"Restoring best checkpoint.",
                flush=True,
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("No best checkpoint was created.")
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(best["model_state"])
    model.to(device)

    evaluation: dict[str, Any] = {}
    requested_splits = ["validation"] + (["test"] if args.run_test else [])
    for split in requested_splits:
        evaluation_rows = _balanced_cap(
            split_rows[split],
            args.max_eval_batches * args.batch_size,
            args.seed,
            shuffle=False,
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
        "study_stage": "complete-data 50-class RGB-only demo; not the final 200-class benchmark",
        "split_policy": "official ASL Citizen train/validation/test; test excluded from tuning",
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
            "stopped_epoch": int(history[-1]["epoch"]) if stopped_early else None,
        },
        "max_train_batches_per_epoch": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "history": history,
        "evaluation": evaluation,
        "duration_seconds": time.perf_counter() - run_started,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "selected_words": str(output_root / "selected_50_words.json"),
            "manifests": {
                "all": str(demo_manifest),
                **{split: str(path) for split, path in split_manifest_paths.items()},
            },
            "training_curves": str(output_root / "training_curves.png"),
        },
    }
    _write_json_atomic(output_root / "baseline_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
