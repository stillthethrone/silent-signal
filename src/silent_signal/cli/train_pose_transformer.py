"""Train and evaluate the pose Graph Encoder + Spatial/Temporal Transformer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.data.manifest import ManifestError
from silent_signal.preprocessing.mediapipe_features import MediaPipeFeatureConfig
from silent_signal.training.pose_trainer import TrainingConfig, train_pose_transformer

_MODEL_OPTIONS = {
    "graph_dim": int,
    "graph_blocks": int,
    "spatial_layers": int,
    "spatial_heads": int,
    "temporal_dim": int,
    "temporal_layers": int,
    "temporal_heads": int,
    "feedforward_ratio": int,
    "dropout": float,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-train-pose-transformer",
        description="Train the pose branch on packed MediaPipe keypoints and a split manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True, help="Packed keypoints .npz.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--view", default="front")
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15, help="0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true", help="Ignore last_checkpoint.pt.")
    parser.add_argument("--run-test", action="store_true", help="Also evaluate the test split.")
    for name, kind in _MODEL_OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        training = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            workers=args.workers,
            view=args.view,
            features=MediaPipeFeatureConfig(target_frames=args.target_frames),
        )
        overrides = {
            name: getattr(args, name) for name in _MODEL_OPTIONS if getattr(args, name) is not None
        }
        report = train_pose_transformer(
            manifest_path=args.manifest,
            keypoints_path=args.keypoints,
            output_root=args.output_root,
            training=training,
            model_overrides=overrides,
            device_name=args.device,
            resume=not args.no_resume,
            run_test=args.run_test,
            log=lambda message: print(message, flush=True),
        )
        summary = {key: report[key] for key in ("best_epoch", "stopped_early", "evaluation")}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (KeyError, ManifestError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
