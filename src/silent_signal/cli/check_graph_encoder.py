"""Smoke-train and validate the ASL Citizen top-200 graph encoder."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from silent_signal.data.collate import collate_graph_pose
from silent_signal.data.dataset import GraphPoseDataset
from silent_signal.data.manifest import read_manifest
from silent_signal.models.factory import (
    build_pose_graph_recognizer,
    graph_encoder_config_dict,
    load_graph_encoder_config,
    model_fingerprint,
)
from silent_signal.pose.cache import sha256_file, write_json_atomic
from silent_signal.training.checkpoint import write_torch_checkpoint_atomic

_DEFAULT_CONFIG = Path("configs/model/asl_citizen_graph_encoder.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-check-graph-encoder",
        description="Validate graph caches with a forward/backward smoke training run.",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--project-commit")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.perf_counter()
    try:
        _log("setup", "loading pinned model configuration")
        experiment = load_graph_encoder_config(args.config)
        manifest_path = args.manifest.resolve()
        _log("setup", f"hashing manifest: {manifest_path}")
        manifest_sha256 = sha256_file(manifest_path)
        if (
            experiment.manifest_sha256 is not None
            and manifest_sha256 != experiment.manifest_sha256
        ):
            raise ValueError(
                "Manifest SHA-256 does not match the graph encoder experiment config."
            )
        _log("setup", "reading official manifest and selecting train samples")
        records = tuple(
            sorted(
                (record for record in read_manifest(manifest_path) if record.split == "train"),
                key=lambda record: record.sample_id,
            )
        )
        limit = args.limit or experiment.smoke.sample_limit
        batch_size = args.batch_size or experiment.smoke.batch_size
        train_steps = args.train_steps or experiment.smoke.train_steps
        if min(limit, batch_size, train_steps) < 1:
            raise ValueError("limit, batch-size, and train-steps must be positive.")
        records = records[:limit]
        dataset = GraphPoseDataset(
            records,
            args.graph_root.resolve(),
            expected_fingerprint=experiment.preprocessing_fingerprint,
        )
        device = _resolve_device(args.device)
        _log(
            "setup",
            f"selected={len(dataset)} batch_size={batch_size} steps={train_steps} "
            f"device={device}",
        )
        result = _run_smoke_training(
            dataset,
            experiment=experiment,
            device=device,
            batch_size=batch_size,
            train_steps=train_steps,
        )
        checkpoint_payload = {
            "schema_version": 1,
            "kind": "graph_encoder_smoke_checkpoint",
            "project_commit": args.project_commit,
            "model_fingerprint": model_fingerprint(experiment.model),
            "experiment": graph_encoder_config_dict(experiment),
            "manifest_sha256": manifest_sha256,
            "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
            "model_state_dict": result.pop("model_state_dict"),
            "optimizer_state_dict": result.pop("optimizer_state_dict"),
            "completed_steps": train_steps,
        }
        checkpoint_path = args.checkpoint.resolve()
        _log("checkpoint", f"writing atomically to {checkpoint_path}")
        checkpoint_started = time.perf_counter()
        write_torch_checkpoint_atomic(
            checkpoint_path,
            checkpoint_payload,
            overwrite=args.overwrite,
        )
        _log(
            "checkpoint",
            "write complete in "
            f"{_duration(time.perf_counter() - checkpoint_started)}; hashing file",
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "state": "passed",
            "project_commit": args.project_commit,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "selected_train_samples": len(dataset),
            "batch_size": batch_size,
            "train_steps": train_steps,
            "model": asdict(experiment.model),
            "model_fingerprint": model_fingerprint(experiment.model),
            "manifest_sha256": manifest_sha256,
            "preprocessing_fingerprint": experiment.preprocessing_fingerprint,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            **result,
        }
        _log("report", f"writing report to {args.report.resolve()}")
        write_json_atomic(args.report.resolve(), report)
        _log("done", f"all checks passed in {_duration(time.perf_counter() - started_at)}")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _run_smoke_training(
    dataset: GraphPoseDataset,
    *,
    experiment: Any,
    device: torch.device,
    batch_size: int,
    train_steps: int,
) -> dict[str, Any]:
    seed = experiment.smoke.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _log("model", "building graph encoder and moving it to the selected device")
    model_started = time.perf_counter()
    model = build_pose_graph_recognizer(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.smoke.learning_rate,
        weight_decay=experiment.smoke.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    _log(
        "model",
        f"ready in {_duration(time.perf_counter() - model_started)}; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}",
    )
    order = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    losses: list[float] = []
    accuracies: list[float] = []
    max_gradient_norm = 0.0
    mask_invariance_max_error: float | None = None
    model.train()
    training_started = time.perf_counter()
    for step in range(train_steps):
        step_started = time.perf_counter()
        indices = _cyclic_batch(order, step * batch_size, batch_size)
        _log(
            f"step {step + 1}/{train_steps}",
            f"loading {len(indices)} graph caches from Drive",
        )
        load_started = time.perf_counter()
        batch = collate_graph_pose([dataset[index] for index in indices])
        _log(
            f"step {step + 1}/{train_steps}",
            f"batch loaded in {_duration(time.perf_counter() - load_started)}; copying to {device}",
        )
        features = torch.from_numpy(batch.features).to(device=device, dtype=torch.float32)
        joint_mask = torch.from_numpy(batch.joint_mask).to(device=device)
        frame_mask = torch.from_numpy(batch.frame_mask).to(device=device)
        labels = torch.from_numpy(batch.labels).to(device=device)
        adjacency = torch.from_numpy(batch.adjacency).to(device=device, dtype=torch.float32)
        if step == 0:
            _log("validation", "checking that masked joints cannot change model output")
            mask_invariance_max_error = _check_mask_invariance(
                model,
                features,
                joint_mask,
                frame_mask,
                adjacency,
            )
            _log(
                "validation",
                f"mask invariance passed; max_error={mask_invariance_max_error:.3e}",
            )
        optimizer.zero_grad(set_to_none=True)
        _log(f"step {step + 1}/{train_steps}", "forward pass")
        logits = model(features, joint_mask, frame_mask, adjacency)
        if logits.shape != (len(indices), experiment.model.num_classes):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Graph encoder produced non-finite logits.")
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Graph encoder produced a non-finite loss.")
        _log(f"step {step + 1}/{train_steps}", "backward pass")
        loss.backward()
        gradient_norm = _gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("Graph encoder produced non-finite gradients.")
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        _log(f"step {step + 1}/{train_steps}", "optimizer step")
        optimizer.step()
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
        print(
            f"[{step + 1}/{train_steps}] loss={losses[-1]:.6f} "
            f"top1={accuracies[-1]:.4f} grad_norm={gradient_norm:.4f} "
            f"step_time={_duration(time.perf_counter() - step_started)} "
            f"total={_duration(time.perf_counter() - training_started)}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "losses": losses,
        "top1_accuracies": accuracies,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "max_gradient_norm": max_gradient_norm,
        "mask_invariance_max_error": mask_invariance_max_error,
        "logits_shape": [batch_size, experiment.model.num_classes],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def _cyclic_batch(order: list[int], start: int, size: int) -> list[int]:
    if not order:
        raise ValueError("Smoke dataset is empty.")
    return [order[(start + offset) % len(order)] for offset in range(size)]


@torch.no_grad()
def _check_mask_invariance(
    model: nn.Module,
    features: Tensor,
    joint_mask: Tensor,
    frame_mask: Tensor,
    adjacency: Tensor,
) -> float:
    was_training = model.training
    model.eval()
    baseline = model(features, joint_mask, frame_mask, adjacency)
    corrupted = features.clone()
    corrupted[~joint_mask] = 10000.0
    candidate = model(corrupted, joint_mask, frame_mask, adjacency)
    if was_training:
        model.train()
    error = float((baseline - candidate).abs().max().cpu())
    if error > 1e-5:
        raise RuntimeError(f"Mask invariance check failed with max error {error}.")
    return error


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm().cpu())
            total += value * value
    return math.sqrt(total)


def _log(stage: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{stage}] {message}", file=sys.stderr, flush=True)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    entrypoint()
