"""Train cached-VideoMAE + MediaPipe pose late fusion on one pinned manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from silent_signal.contracts import ManifestRecord
from silent_signal.data.rgb_feature_pack import read_rgb_feature_pack
from silent_signal.evaluation.metrics import classification_metrics
from silent_signal.models.encoders.pose_transformer import PoseTransformerConfig
from silent_signal.models.rgb_pose_fusion import RGBPoseFusionClassifier, RGBPoseFusionConfig
from silent_signal.pose.cache import sha256_file
from silent_signal.pose.layouts import get_pose_layout
from silent_signal.preprocessing.mediapipe_features import (
    AugmentationConfig,
    MediaPipeFeatureConfig,
)
from silent_signal.preprocessing.pose_features import normalized_adjacency
from silent_signal.training.checkpoint import write_torch_checkpoint_atomic
from silent_signal.training.pose_trainer import PoseSequenceDataset, load_split_data


@dataclass(frozen=True, slots=True)
class FusionTrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.10
    class_balance_power: float = 0.50
    max_class_weight: float = 2.0
    warmup_epochs: int = 5
    patience: int = 10
    min_delta: float = 0.002
    gradient_clip: float = 1.0
    seed: int = 42
    workers: int = 2
    view: str = "front"
    features: MediaPipeFeatureConfig = field(default_factory=MediaPipeFeatureConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive.")
        if not 0 <= self.label_smoothing < 1 or self.weight_decay < 0:
            raise ValueError("Invalid regularization settings.")
        if self.patience < 0 or self.min_delta < 0 or self.gradient_clip < 0:
            raise ValueError("Invalid early-stopping or gradient-clip settings.")
        if not 0 <= self.class_balance_power <= 1 or self.max_class_weight < 1:
            raise ValueError("Invalid class-balance settings.")


class RGBPoseDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Join dynamic pose views and fixed frozen-RGB tokens by manifest sample ID."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        rgb_features: np.ndarray,
        labels: Sequence[int],
        features: MediaPipeFeatureConfig,
        *,
        augmentation: AugmentationConfig | None = None,
        seed: int = 0,
    ) -> None:
        if len(sequences) != len(rgb_features) or len(sequences) != len(labels):
            raise ValueError("Pose, RGB and label counts must match.")
        rgb = np.asarray(rgb_features)
        if rgb.ndim != 3 or rgb.shape[1] < 1 or rgb.shape[2] < 1:
            raise ValueError("RGB features must have shape [N, L, D].")
        if not np.isfinite(rgb).all():
            raise ValueError("RGB features contain NaN or infinity.")
        self.pose = PoseSequenceDataset(
            sequences,
            labels,
            features,
            augmentation=augmentation,
            seed=seed,
        )
        self.rgb = rgb

    def set_epoch(self, epoch: int) -> None:
        self.pose.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.pose)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pose_features, joint_mask, frame_mask, label = self.pose[index]
        rgb = torch.from_numpy(np.asarray(self.rgb[index], dtype=np.float32))
        rgb_mask = torch.ones(rgb.shape[0], dtype=torch.bool)
        return pose_features, joint_mask, frame_mask, rgb, rgb_mask, label


def load_fusion_data(
    manifest_path: Path,
    keypoints_path: Path,
    rgb_pack_path: Path,
    *,
    view: str,
) -> tuple[
    dict[str, list[ManifestRecord]],
    dict[str, list[np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Load both modalities and require exact RGB coverage of the pose manifest."""

    records, sequences, summary = load_split_data(
        manifest_path, keypoints_path, view=view
    )
    packed = read_rgb_feature_pack(rgb_pack_path)
    rgb_ids = tuple(str(item) for item in packed.sample_ids)
    rgb_index: Mapping[str, int] = packed.index
    manifest_ids = {
        record.sample_id for split_records in records.values() for record in split_records
    }
    missing = sorted(manifest_ids - set(rgb_ids))
    extra = sorted(set(rgb_ids) - manifest_ids)
    if missing or extra:
        raise ValueError(
            "RGB feature pack and manifest sample IDs differ: "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    all_features = np.asarray(packed.features)
    if all_features.ndim != 3 or len(all_features) != len(rgb_ids):
        raise ValueError("Packed RGB features must have shape [N, L, D].")
    if not np.isfinite(all_features).all():
        raise ValueError("Packed RGB features contain NaN or infinity.")
    _validate_rgb_metadata(
        packed.metadata,
        manifest_sha256=sha256_file(manifest_path),
        samples=len(rgb_ids),
        feature_shape=tuple(int(value) for value in all_features.shape[1:]),
    )
    rgb_by_split = {
        split: np.asarray(
            [all_features[rgb_index[record.sample_id]] for record in rows]
        )
        for split, rows in records.items()
    }
    summary = {
        **summary,
        "rgb_tokens": int(all_features.shape[1]),
        "rgb_feature_dim": int(all_features.shape[2]),
        "rgb_metadata": dict(packed.metadata),
    }
    return records, sequences, rgb_by_split, summary


def _validate_rgb_metadata(
    metadata: Mapping[str, Any],
    *,
    manifest_sha256: str,
    samples: int,
    feature_shape: tuple[int, int],
) -> None:
    required = {
        "extraction_fingerprint",
        "manifest_sha256",
        "model_id",
        "model_revision",
        "preprocessing",
        "kind",
        "samples",
        "feature_shape",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"RGB feature metadata is missing required fields: {missing}.")
    if metadata["kind"] != "videomae_temporal_token_pack":
        raise ValueError(f"Unexpected RGB feature pack kind: {metadata['kind']!r}.")
    if metadata["manifest_sha256"] != manifest_sha256:
        raise ValueError("RGB feature pack was extracted for a different manifest.")
    if int(metadata["samples"]) != samples:
        raise ValueError("RGB feature metadata sample count does not match the pack.")
    raw_shape = metadata["feature_shape"]
    if (
        not isinstance(raw_shape, Sequence)
        or isinstance(raw_shape, str | bytes)
        or len(raw_shape) != 2
    ):
        raise ValueError("RGB feature metadata shape must contain [tokens, channels].")
    if tuple(int(value) for value in raw_shape) != feature_shape:
        raise ValueError("RGB feature metadata shape does not match the pack.")
    for key in ("extraction_fingerprint", "model_id", "model_revision"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"RGB feature metadata field {key!r} must be a non-empty string.")
    if not isinstance(metadata["preprocessing"], Mapping):
        raise ValueError("RGB feature preprocessing metadata must be an object.")


def train_rgb_pose_fusion(
    *,
    manifest_path: Path,
    keypoints_path: Path,
    rgb_pack_path: Path,
    output_root: Path,
    training: FusionTrainingConfig,
    pose_overrides: dict[str, Any] | None = None,
    fusion_overrides: dict[str, Any] | None = None,
    device_name: str = "auto",
    resume: bool = True,
    run_test: bool = False,
    project_commit: str | None = None,
    progress_every_batches: int = 25,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train with validation early stopping and evaluate only the best checkpoint."""

    if progress_every_batches < 0:
        raise ValueError("progress_every_batches must not be negative.")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log("[setup] loading packed pose and cached RGB features...")
    records, sequences, rgb, data = load_fusion_data(
        manifest_path, keypoints_path, rgb_pack_path, view=training.view
    )
    log(
        f"[data] {data['classes']} classes | clips {data['clips']} | "
        f"RGB [{data['rgb_tokens']}, {data['rgb_feature_dim']}]"
    )
    if not records["validation"]:
        raise ValueError("A validation split is required for early stopping.")

    layout = get_pose_layout(training.features.layout_name)
    pose_config = PoseTransformerConfig(
        **{
            "input_dim": training.features.channels,
            "num_nodes": layout.num_joints,
            "max_frames": training.features.target_frames,
            "graph_dim": 96,
            "graph_blocks": 2,
            "spatial_layers": 2,
            "spatial_heads": 4,
            "temporal_dim": 192,
            "temporal_layers": 2,
            "temporal_heads": 4,
            "feedforward_ratio": 2,
            "dropout": 0.30,
            "num_classes": data["classes"],
            **(pose_overrides or {}),
        }
    )
    fusion_config = RGBPoseFusionConfig(
        **{
            "rgb_source_dim": data["rgb_feature_dim"],
            "max_rgb_tokens": data["rgb_tokens"],
            "num_classes": data["classes"],
            **(fusion_overrides or {}),
        }
    )
    device = _device(device_name)
    _seed_everything(training.seed)
    log("[setup] verifying manifest, pose-pack and RGB-pack fingerprints...")
    fingerprint = {
        "manifest_sha256": _path_sha256(manifest_path),
        "keypoints_sha256": _path_sha256(keypoints_path),
        "rgb_pack_sha256": _path_sha256(rgb_pack_path),
        "model_fingerprint": fusion_config.fingerprint(pose_config),
        "training": asdict(training),
        "project_commit": project_commit,
    }
    config_payload = {
        "pose_model": asdict(pose_config),
        "fusion_model": asdict(fusion_config),
        **fingerprint,
        "data": _without(data, "keypoints_metadata"),
    }

    labels = {split: [row.class_index for row in rows] for split, rows in records.items()}
    train_set = RGBPoseDataset(
        sequences["train"],
        rgb["train"],
        labels["train"],
        training.features,
        augmentation=training.augmentation,
        seed=training.seed,
    )
    log(f"[setup] preparing {len(records['validation']):,} validation samples...")
    eval_sets = {
        "validation": RGBPoseDataset(
            sequences["validation"],
            rgb["validation"],
            labels["validation"],
            training.features,
        )
    }
    adjacency = torch.from_numpy(normalized_adjacency(layout)).to(device)
    model = RGBPoseFusionClassifier(pose_config, fusion_config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    log(f"[model] {parameters:,} trainable parameters on {device}")
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, training.weight_decay), lr=training.learning_rate
    )
    steps_per_epoch = math.ceil(len(train_set) / training.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _warmup_cosine(
            training.warmup_epochs * steps_per_epoch,
            training.epochs * steps_per_epoch,
        ),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    class_weights = _class_weights(
        labels["train"],
        data["classes"],
        training.class_balance_power,
        training.max_class_weight,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=training.label_smoothing
    )

    last_path = output_root / "last_checkpoint.pt"
    best_path = output_root / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_loss, bad_epochs = 0, math.inf, 0
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "last_checkpoint.pt was made with different data or settings; use a new folder."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        best_loss = float(state["best_loss"])
        bad_epochs = int(state["bad_epochs"])
        history = list(state["history"])
        log(f"[resume] epoch {start_epoch}; best validation loss {best_loss:.4f}")

    (output_root / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stopped_early = False
    for epoch in range(start_epoch, training.epochs):
        if bad_epochs >= training.patience > 0:
            stopped_early = True
            log(f"[early stop] no validation-loss improvement for {bad_epochs} epochs")
            break
        epoch_started = time.perf_counter()
        train_set.set_epoch(epoch)
        loader = _loader(train_set, training, shuffle=True, seed=training.seed + epoch)
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        log(f"[train {epoch + 1:>3}/{training.epochs}] starting {len(loader)} batches...")
        for batch_index, batch in enumerate(loader, start=1):
            pose, joint_mask, frame_mask, rgb_tokens, rgb_mask, target = [
                item.to(device) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(
                    pose, joint_mask, frame_mask, adjacency, rgb_tokens, rgb_mask
                )
                loss = criterion(logits.float(), target)
            scaler.scale(loss).backward()
            if training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach()) * len(target)
            correct += int((logits.argmax(dim=1) == target).sum())
            seen += len(target)
            if progress_every_batches and (
                batch_index == 1
                or batch_index == len(loader)
                or batch_index % progress_every_batches == 0
            ):
                elapsed = time.perf_counter() - epoch_started
                rate = batch_index / elapsed if elapsed else 0.0
                eta = (len(loader) - batch_index) / rate / 60 if rate else 0.0
                log(
                    f"[train {epoch + 1:>3}/{training.epochs}] "
                    f"batch {batch_index:>3}/{len(loader)} | loss {total_loss / seen:.3f} | "
                    f"top1 {correct / seen:.3f} | ETA {eta:.1f} min"
                )

        log(f"[validation] epoch {epoch + 1}/{training.epochs} | evaluating...")
        validation = _evaluate(
            model, eval_sets["validation"], training, adjacency, device, criterion
        )
        improved = validation["loss"] < best_loss - training.min_delta
        if improved:
            best_loss, bad_epochs = validation["loss"], 0
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / seen,
            "train_top1": correct / seen,
            "validation_loss": validation["loss"],
            "validation_top1": validation["metrics"]["top1_accuracy"],
            "validation_top5": validation["metrics"]["top5_accuracy"],
            "validation_macro_f1": validation["metrics"]["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
            "improved": improved,
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(record)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "bad_epochs": bad_epochs,
            "history": history,
            "fingerprint": fingerprint,
            "pose_config": asdict(pose_config),
            "fusion_config": asdict(fusion_config),
        }
        if improved:
            write_torch_checkpoint_atomic(best_path, state, overwrite=True)
        write_torch_checkpoint_atomic(last_path, state, overwrite=True)
        (output_root / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        status = "best" if improved else f"wait {bad_epochs}/{training.patience}"
        log(
            f"[epoch {epoch + 1:>3}/{training.epochs}] train loss {record['train_loss']:.3f} "
            f"top1 {record['train_top1']:.3f} | val loss {record['validation_loss']:.3f} "
            f"top1 {record['validation_top1']:.3f} top5 {record['validation_top5']:.3f} "
            f"F1 {record['validation_macro_f1']:.3f} | {status} | {record['seconds']}s"
        )
    else:
        stopped_early = bad_epochs >= training.patience > 0

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written.")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    if run_test and records["test"]:
        log(f"[test] preparing {len(records['test']):,} samples after checkpoint selection...")
        eval_sets["test"] = RGBPoseDataset(
            sequences["test"], rgb["test"], labels["test"], training.features
        )
    evaluation: dict[str, Any] = {}
    selected_splits = ["validation"] + (
        ["test"] if run_test and "test" in eval_sets else []
    )
    for split in selected_splits:
        log(f"[{split}] evaluating best checkpoint...")
        result = _evaluate(model, eval_sets[split], training, adjacency, device, criterion)
        _write_predictions(
            output_root / f"predictions_{split}.csv", records[split], result, data
        )
        _write_per_class(output_root / f"per_class_{split}.csv", result["metrics"], data)
        np.savetxt(
            output_root / f"confusion_{split}.csv",
            np.asarray(result["metrics"]["confusion"]),
            fmt="%d",
            delimiter=",",
        )
        evaluation[split] = {
            "loss": result["loss"],
            **_without(result["metrics"], "per_class", "confusion"),
        }
        scores = evaluation[split]
        log(
            f"[{split}] loss {scores['loss']:.3f} | top1 {scores['top1_accuracy']:.3f} | "
            f"top5 {scores['top5_accuracy']:.3f} | macro-F1 {scores['macro_f1']:.3f}"
        )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "parameters": parameters,
        "pose_model": asdict(pose_config),
        "fusion_model": asdict(fusion_config),
        "fingerprint": fingerprint,
        "data": _without(data, "keypoints_metadata"),
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["best_loss"]),
        "stopped_early": stopped_early,
        "test_evaluated": "test" in evaluation,
        "evaluation": evaluation,
        "duration_seconds": round(time.perf_counter() - started, 1),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _evaluate(
    model: RGBPoseFusionClassifier,
    dataset: RGBPoseDataset,
    training: FusionTrainingConfig,
    adjacency: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    probabilities, targets, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in _loader(dataset, training, shuffle=False, seed=0):
            pose, joint_mask, frame_mask, rgb, rgb_mask, target = [
                item.to(device) for item in batch
            ]
            logits = model(
                pose, joint_mask, frame_mask, adjacency, rgb, rgb_mask
            ).float()
            total_loss += float(criterion(logits, target)) * len(target)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
    truth = np.concatenate(targets)
    scores = np.concatenate(probabilities).astype(np.float64)
    return {
        "loss": total_loss / len(truth),
        "metrics": classification_metrics(truth, scores),
        "probabilities": scores,
        "truth": truth,
    }


def _write_predictions(
    path: Path,
    rows: Sequence[ManifestRecord],
    result: dict[str, Any],
    data: dict[str, Any],
) -> None:
    names = data["class_names"]
    probabilities = result["probabilities"]
    ranked = np.argsort(-probabilities, axis=1)[:, :5]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "video_id",
                "true_class",
                "true_gloss",
                "pred_class",
                "pred_gloss",
                "confidence",
                "correct",
                "top5",
            ]
        )
        for record, scores, top in zip(rows, probabilities, ranked, strict=True):
            predicted = int(top[0])
            writer.writerow(
                [
                    record.sample_id,
                    record.video_id,
                    record.class_index,
                    names[record.class_index],
                    predicted,
                    names[predicted],
                    f"{scores[predicted]:.4f}",
                    int(predicted == record.class_index),
                    "|".join(names[int(index)] for index in top),
                ]
            )


def _write_per_class(path: Path, metrics: dict[str, Any], data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_index", "gloss", "support", "precision", "recall", "f1"],
        )
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({**row, "gloss": data["class_names"][row["class_index"]]})


def _loader(
    dataset: RGBPoseDataset,
    training: FusionTrainingConfig,
    *,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=training.workers if dataset.pose.augmentation else 0,
        pin_memory=torch.cuda.is_available(),
    )


def _class_weights(
    labels: Sequence[int], classes: int, power: float, maximum: float
) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every class needs at least one training sample.")
    weights = np.power(counts.mean() / counts, power)
    weights /= weights.mean()
    weights = np.clip(weights, 1.0 / maximum, maximum)
    return torch.tensor(weights, dtype=torch.float32)


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim < 2 or "position" in name or "token" in name else decay
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _warmup_cosine(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _path_sha256(path: Path) -> str:
    source = path.resolve()
    digest = hashlib.sha256()
    paths = (
        [source]
        if source.is_file()
        else sorted(item for item in source.rglob("*") if item.is_file())
    )
    if not paths:
        raise FileNotFoundError(f"Fingerprint input is empty: {source}")
    for item in paths:
        relative = item.name if source.is_file() else item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}
