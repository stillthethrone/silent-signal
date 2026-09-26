"""Train cached VideoMAE + MediaPipe pose gated fusion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.data.manifest import ManifestError
from silent_signal.preprocessing.mediapipe_features import MediaPipeFeatureConfig
from silent_signal.training.rgb_pose_fusion_trainer import (
    FusionTrainingConfig,
    train_rgb_pose_fusion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silent_signal.cli.train_rgb_pose_fusion",
        description="Train a small gated fusion model from packed pose and frozen RGB tokens.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--rgb-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--view", default="front")
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--class-balance-power", type=float, default=0.50)
    parser.add_argument("--max-class-weight", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project-commit")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-test", action="store_true")

    parser.add_argument("--graph-dim", type=int, default=96)
    parser.add_argument("--graph-blocks", type=int, default=2)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--spatial-heads", type=int, default=4)
    parser.add_argument("--temporal-dim", type=int, default=192)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--pose-feedforward-ratio", type=int, default=2)
    parser.add_argument("--pose-dropout", type=float, default=0.30)

    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--rgb-layers", type=int, default=1)
    parser.add_argument("--rgb-heads", type=int, default=4)
    parser.add_argument("--fusion-feedforward-ratio", type=int, default=2)
    parser.add_argument("--fusion-dropout", type=float, default=0.30)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        training = FusionTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            class_balance_power=args.class_balance_power,
            max_class_weight=args.max_class_weight,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            workers=args.workers,
            view=args.view,
            features=MediaPipeFeatureConfig(target_frames=args.target_frames),
        )
        pose_overrides = {
            "graph_dim": args.graph_dim,
            "graph_blocks": args.graph_blocks,
            "spatial_layers": args.spatial_layers,
            "spatial_heads": args.spatial_heads,
            "temporal_dim": args.temporal_dim,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
            "feedforward_ratio": args.pose_feedforward_ratio,
            "dropout": args.pose_dropout,
        }
        fusion_overrides = {
            "fusion_dim": args.fusion_dim,
            "rgb_layers": args.rgb_layers,
            "rgb_heads": args.rgb_heads,
            "feedforward_ratio": args.fusion_feedforward_ratio,
            "dropout": args.fusion_dropout,
            "modality_dropout": args.modality_dropout,
        }
        report = train_rgb_pose_fusion(
            manifest_path=args.manifest,
            keypoints_path=args.keypoints,
            rgb_pack_path=args.rgb_features,
            output_root=args.output_root,
            training=training,
            pose_overrides=pose_overrides,
            fusion_overrides=fusion_overrides,
            device_name=args.device,
            resume=not args.no_resume,
            run_test=args.run_test,
            project_commit=args.project_commit,
            progress_every_batches=args.progress_every_batches,
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
