"""Train the pose Graph-Transformer on packed MediaPipe keypoints with a split manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from silent_signal.contracts import ManifestRecord
from silent_signal.data.keypoint_pack import read_packed_keypoints, select_sequences
from silent_signal.data.manifest import read_manifest
from silent_signal.evaluation.metrics import classification_metrics
from silent_signal.models.encoders.pose_transformer import (
    PoseGraphTransformer,
    PoseTransformerConfig,
)
from silent_signal.pose.layouts import get_pose_layout
from silent_signal.preprocessing.mediapipe_features import (
    AugmentationConfig,
    MediaPipeFeatureConfig,
    mediapipe_graph_features,
)
from silent_signal.preprocessing.pose_features import normalized_adjacency
from silent_signal.training.checkpoint import write_torch_checkpoint_atomic

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    patience: int = 15
    min_delta: float = 0.0
    gradient_clip: float = 1.0
    seed: int = 42
    workers: int = 2
    view: str = "front"
    features: MediaPipeFeatureConfig = field(default_factory=MediaPipeFeatureConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive.")
        if not 0 <= self.label_smoothing < 1 or self.weight_decay < 0 or self.patience < 0:
            raise ValueError("Invalid regularization or early-stopping settings.")


class PoseSequenceDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Features are cached for evaluation; training views change with ``set_epoch``."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        labels: Sequence[int],
        features: MediaPipeFeatureConfig,
        *,
        augmentation: AugmentationConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.sequences = list(sequences)
        self.labels = [int(label) for label in labels]
        self.features = features
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0
        self._cache = (
            None if augmentation else [self._build(index, None) for index in range(len(self))]
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self._cache is not None:
            return self._cache[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        return self._build(index, rng)

    def _build(self, index: int, rng: np.random.Generator | None) -> tuple[torch.Tensor, ...]:
        features, joint_mask, frame_mask = mediapipe_graph_features(
            self.sequences[index], self.features, rng=rng, augmentation=self.augmentation
        )
        return (
            torch.from_numpy(features),
            torch.from_numpy(joint_mask),
            torch.from_numpy(frame_mask),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def load_split_data(
    manifest_path: Path, keypoints_path: Path, *, view: str
) -> tuple[dict[str, list[ManifestRecord]], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Join the manifest's split and labels with the packed keypoints of one view."""

    packed = read_packed_keypoints(keypoints_path)
    packed_ids = set(packed.sample_ids)
    records = [record for record in read_manifest(manifest_path) if record.view == view]
    if not records:
        raise ValueError(f"No {view!r} records in {manifest_path}.")
    missing = sorted(record.sample_id for record in records if record.sample_id not in packed_ids)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"Packed keypoints are missing {len(missing)} manifest samples: {preview}{suffix}"
        )
    kept = sorted(records, key=lambda r: r.sample_id)
    classes = sorted({record.class_index for record in records})
    if classes != list(range(len(classes))):
        raise ValueError("Manifest class indices must be contiguous from 0.")
    by_split = {split: [r for r in kept if r.split == split] for split in SPLITS}
    missing_train = set(classes) - {r.class_index for r in by_split["train"]}
    if missing_train:
        raise ValueError(f"Classes without packed training clips: {sorted(missing_train)}")
    sequences = {
        split: select_sequences(packed, [r.sample_id for r in rows])
        for split, rows in by_split.items()
    }
    names = {record.class_index: record.gloss_name for record in records}
    summary = {
        "view": view,
        "manifest_records": len(records),
        "packed_records": len(kept),
        "dropped_without_keypoints": len(records) - len(kept),
        "classes": len(classes),
        "class_names": [names[index] for index in classes],
        "clips": {split: len(rows) for split, rows in by_split.items()},
        "signers": {split: sorted({r.signer_id for r in rows}) for split, rows in by_split.items()},
        "keypoints_metadata": packed.metadata,
    }
    return by_split, sequences, summary


def train_pose_transformer(
    *,
    manifest_path: Path,
    keypoints_path: Path,
    output_root: Path,
    training: TrainingConfig,
    model_overrides: dict[str, Any] | None = None,
    device_name: str = "auto",
    resume: bool = True,
    run_test: bool = False,
    project_commit: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train with validation early stopping, then evaluate the best checkpoint."""

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, sequences, data_summary = load_split_data(
        manifest_path, keypoints_path, view=training.view
    )
    layout = get_pose_layout(training.features.layout_name)
    model_config = PoseTransformerConfig(
        **{
            "input_dim": training.features.channels,
            "num_nodes": layout.num_joints,
            "max_frames": training.features.target_frames,
            "num_classes": data_summary["classes"],
            **(model_overrides or {}),
        }
    )
    device = _device(device_name)
    _seed_everything(training.seed)
    fingerprint = {
        "manifest_sha256": _sha256(manifest_path),
        "keypoints_sha256": _sha256(keypoints_path),
        "model_fingerprint": model_config.fingerprint(),
        "training": asdict(training),
        "project_commit": project_commit,
    }
    config_payload = {
        "model": asdict(model_config),
        **fingerprint,
        "data": _without(data_summary, "keypoints_metadata"),
    }
    log(
        f"[data] {data_summary['classes']} classes | clips {data_summary['clips']} | "
        f"signers { {k: len(v) for k, v in data_summary['signers'].items()} } | "
        f"dropped without keypoints: {data_summary['dropped_without_keypoints']}"
    )

    labels = {split: [r.class_index for r in rows] for split, rows in records.items()}
    train_set = PoseSequenceDataset(
        sequences["train"],
        labels["train"],
        training.features,
        augmentation=training.augmentation,
        seed=training.seed,
    )
    eval_sets = {
        split: PoseSequenceDataset(sequences[split], labels[split], training.features)
        for split in ("validation", "test")
        if records[split]
    }
    if "validation" not in eval_sets:
        raise ValueError("A validation split is required for early stopping.")
    adjacency = torch.from_numpy(normalized_adjacency(layout)).to(device)

    model = PoseGraphTransformer(model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    log(f"[model] {parameters:,} parameters on {device}")
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, training.weight_decay), lr=training.learning_rate
    )
    steps_per_epoch = math.ceil(len(train_set) / training.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _warmup_cosine(training.warmup_epochs * steps_per_epoch, training.epochs * steps_per_epoch),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=training.label_smoothing)

    last_path, best_path = output_root / "last_checkpoint.pt", output_root / "best_checkpoint.pt"
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
        start_epoch, best_loss, bad_epochs = state["epoch"], state["best_loss"], state["bad_epochs"]
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
        for features, joint_mask, frame_mask, target in loader:
            features, joint_mask = features.to(device), joint_mask.to(device)
            frame_mask, target = frame_mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(features, joint_mask, frame_mask, adjacency)
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
            "model_config": asdict(model_config),
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
            f"F1 {record['validation_macro_f1']:.3f} | lr {record['learning_rate']:.2e} | "
            f"{status} | {record['seconds']}s"
        )
    else:
        stopped_early = bad_epochs >= training.patience > 0

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written.")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    evaluation: dict[str, Any] = {}
    for split in ["validation"] + (["test"] if run_test and "test" in eval_sets else []):
        result = _evaluate(model, eval_sets[split], training, adjacency, device, criterion)
        _write_predictions(
            output_root / f"predictions_{split}.csv", records[split], result, data_summary
        )
        _write_per_class(output_root / f"per_class_{split}.csv", result["metrics"], data_summary)
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
            f"[{split}] loss {result['loss']:.3f} | top1 {scores['top1_accuracy']:.3f} | "
            f"top5 {scores['top5_accuracy']:.3f} | macro-F1 {scores['macro_f1']:.3f}"
        )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "parameters": parameters,
        "model": asdict(model_config),
        "fingerprint": fingerprint,
        "data": _without(data_summary, "keypoints_metadata"),
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
    model: PoseGraphTransformer,
    dataset: PoseSequenceDataset,
    training: TrainingConfig,
    adjacency: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    probabilities, targets, total_loss = [], [], 0.0
    with torch.no_grad():
        for features, joint_mask, frame_mask, target in _loader(
            dataset, training, shuffle=False, seed=0
        ):
            logits = model(
                features.to(device), joint_mask.to(device), frame_mask.to(device), adjacency
            ).float()
            total_loss += float(criterion(logits, target.to(device))) * len(target)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(target.numpy())
    truth = np.concatenate(targets)
    scores = np.concatenate(probabilities).astype(np.float64)
    return {
        "loss": total_loss / len(truth),
        "metrics": classification_metrics(truth, scores),
        "probabilities": scores,
        "truth": truth,
    }


def _write_predictions(
    path: Path, rows: Sequence[ManifestRecord], result: dict[str, Any], data: dict[str, Any]
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
                "signer_id",
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
                    record.signer_id,
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
            handle, fieldnames=["class_index", "gloss", "support", "precision", "recall", "f1"]
        )
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({**row, "gloss": data["class_names"][row["class_index"]]})


def _loader(
    dataset: PoseSequenceDataset, training: TrainingConfig, *, shuffle: bool, seed: int
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=training.workers if dataset.augmentation else 0,
        pin_memory=torch.cuda.is_available(),
    )


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (
            no_decay if parameter.ndim < 2 or "embedding" in name or "token" in name else decay
        ).append(parameter)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}
