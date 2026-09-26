"""Prepare the cropped Kaggle VSL MediaPipe release for graph training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from silent_signal.contracts import ManifestRecord
from silent_signal.data.kaggle_vsl import (
    build_kaggle_vsl_manifest,
    class_count_summary,
    selection_payload,
)
from silent_signal.data.keypoint_pack import write_packed_keypoints
from silent_signal.data.manifest import (
    ManifestError,
    read_manifest,
    write_labels,
    write_manifest,
)
from silent_signal.pose.cache import PoseCacheError, pose_cache_path, sha256_file, write_json_atomic
from silent_signal.preprocessing.cache import (
    graph_cache_is_current,
    write_graph_pose_cache,
)
from silent_signal.preprocessing.mediapipe_features import (
    MediaPipeGraphPreprocessConfig,
    load_mediapipe_array,
    mediapipe_preprocessing_fingerprint,
    prepare_mediapipe_graph_pose,
    validate_raw_keypoints,
)

DEFAULT_DATASET_HANDLE = "nguyenanfms/vsl-vietnamese-sign-language-v2/versions/8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-prepare-kaggle-vsl-mediapipe",
        description="Build a VSL manifest and convert MediaPipe .npy arrays to graph caches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Select classes and build manifest metadata.")
    build.add_argument("--keypoint-root", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--labels", type=Path, required=True)
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--classes", type=int, default=50)
    build.add_argument("--min-official-train-samples", type=int, default=40)
    build.add_argument("--validation-fraction", type=float, default=0.20)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--dataset-handle", default=DEFAULT_DATASET_HANDLE)

    convert = subparsers.add_parser("convert", help="Create resumable graph-ready caches.")
    convert.add_argument("--keypoint-root", type=Path, required=True)
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--report", type=Path, required=True)
    convert.add_argument("--limit", type=int)
    convert.add_argument("--progress-every", type=int, default=50)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--continue-on-error", action="store_true")

    pack = subparsers.add_parser(
        "pack", help="Pack manifest-selected raw [T,76,3] arrays into one resumable NPZ."
    )
    pack.add_argument("--keypoint-root", type=Path, required=True)
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--report", type=Path, required=True)
    pack.add_argument("--dataset-handle", default=DEFAULT_DATASET_HANDLE)
    pack.add_argument("--progress-every", type=int, default=100)
    pack.add_argument("--overwrite", action="store_true")

    fingerprint = subparsers.add_parser("fingerprint", help="Print preprocessing identity.")
    fingerprint.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return _build(args)
        if args.command == "convert":
            return _convert(args)
        if args.command == "pack":
            return _pack(args)
        config, _ = _load_config(args.config)
        print(mediapipe_preprocessing_fingerprint(config))
        return 0
    except (ManifestError, OSError, PoseCacheError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _build(args: argparse.Namespace) -> int:
    result = build_kaggle_vsl_manifest(
        args.keypoint_root,
        classes=args.classes,
        min_official_train_samples=args.min_official_train_samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    write_manifest(result.records, args.manifest)
    write_labels(result.labels, args.labels, dataset="kaggle_vsl_mediapipe")
    selection = selection_payload(
        result,
        dataset_handle=args.dataset_handle,
        min_official_train_samples=args.min_official_train_samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        requested_classes=args.classes,
    )
    write_json_atomic(args.selection, selection)
    report = {
        "schema_version": 1,
        "status": "complete",
        "keypoint_root": str(args.keypoint_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "protocol_warning": (
            "The public processed release does not expose signer IDs. The official test split "
            "is preserved, but validation is deterministic class-stratified sample-disjoint."
        ),
        **class_count_summary(result),
    }
    write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _convert(args: argparse.Namespace) -> int:
    config, expected = _load_config(args.config)
    fingerprint = mediapipe_preprocessing_fingerprint(config)
    manifest_path = args.manifest.resolve()
    records = tuple(sorted(read_manifest(manifest_path), key=lambda item: item.sample_id))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        records = records[: args.limit]
    if not records:
        raise ValueError("Manifest contains no records.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must not be negative.")
    if any(record.split not in {"train", "validation", "test"} for record in records):
        raise ValueError("Every manifest row must have a train, validation, or test split.")

    manifest_sha256 = sha256_file(manifest_path)
    expected_manifest = expected.get("manifest_sha256")
    if args.limit is None and expected_manifest and expected_manifest != manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected {expected_manifest}, found {manifest_sha256}."
        )
    if args.limit is None and expected.get("classes") is not None:
        actual_classes = len({record.class_index for record in records})
        if int(expected["classes"]) != actual_classes:
            raise ValueError(f"Expected {expected['classes']} classes, found {actual_classes}.")

    root = args.keypoint_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "source_extractor": "mediapipe_holistic",
        "keypoint_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "output_root": str(output_root),
        "selected": len(records),
        "split_counts": dict(Counter(str(record.split) for record in records)),
        "preprocessing": asdict(config),
        "preprocessing_fingerprint": fingerprint,
        "prepared": 0,
        "resumed": 0,
        "failed": 0,
        "failures": [],
    }
    started_at = time.perf_counter()
    observed_ratios: list[float] = []
    usable_ratios: list[float] = []
    for position, record in enumerate(records, start=1):
        destination = pose_cache_path(output_root, record.sample_id)
        try:
            if destination.is_file() and not args.overwrite and graph_cache_is_current(
                destination,
                sample_id=record.sample_id,
                preprocessing_fingerprint=fingerprint,
            ):
                summary["resumed"] += 1
                _progress(
                    position,
                    len(records),
                    "resume",
                    summary,
                    args.progress_every,
                    started_at,
                )
                continue
            source = root / record.video_path
            if not source.is_file():
                raise ValueError(f"MediaPipe keypoint file not found: {source}")
            array = load_mediapipe_array(source)
            sample = prepare_mediapipe_graph_pose(
                array,
                sample_id=record.sample_id,
                class_index=record.class_index,
                split=str(record.split),
                config=config,
                source_path=record.video_path,
                fps=record.fps or record.metadata_fps,
            )
            write_graph_pose_cache(destination, sample, overwrite=destination.exists())
            summary["prepared"] += 1
            observed_ratios.append(float(sample.metadata["observed_joint_ratio"]))
            usable_ratios.append(float(sample.metadata["usable_joint_ratio"]))
            _progress(position, len(records), "prepare", summary, args.progress_every, started_at)
        except (OSError, PoseCacheError, ValueError) as exc:
            summary["failed"] += 1
            summary["failures"].append({"sample_id": record.sample_id, "error": str(exc)})
            print(f"failed {record.sample_id}: {exc}", file=sys.stderr, flush=True)
            _progress(position, len(records), "failed", summary, args.progress_every, started_at)
            if not args.continue_on_error:
                break

    summary["mean_observed_joint_ratio"] = _mean(observed_ratios)
    summary["mean_usable_joint_ratio"] = _mean(usable_ratios)
    summary["duration_seconds"] = round(time.perf_counter() - started_at, 3)
    summary["status"] = "complete" if summary["failed"] == 0 else "failed"
    write_json_atomic(report_path, summary)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2),
        flush=True,
    )
    return 1 if summary["failed"] else 0


def _pack(args: argparse.Namespace) -> int:
    """Write exactly the manifest-selected canonical arrays into one portable archive."""

    manifest_path = args.manifest.resolve()
    root = args.keypoint_root.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    records = tuple(sorted(read_manifest(manifest_path), key=lambda item: item.sample_id))
    if not records:
        raise ValueError("Manifest contains no records.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must not be negative.")
    manifest_sha256 = sha256_file(manifest_path)
    source_sha256 = _selected_source_sha256(records, root)
    if output_path.is_file() and report_path.is_file() and not args.overwrite:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        reusable = (
            previous.get("status") == "complete"
            and previous.get("manifest_sha256") == manifest_sha256
            and previous.get("dataset_handle") == args.dataset_handle
            and int(previous.get("samples", -1)) == len(records)
            and previous.get("source_sha256") == source_sha256
            and previous.get("output_sha256") == sha256_file(output_path)
        )
        if reusable:
            print(json.dumps(previous, ensure_ascii=False, indent=2), flush=True)
            print("Packed keypoints are current; reusing the Drive artifact.", flush=True)
            return 0
        raise ValueError(
            f"Packed output already exists but does not match this manifest: {output_path}. "
            "Use --overwrite or choose another output path."
        )

    sequences: dict[str, Any] = {}
    total_frames = 0
    started_at = time.perf_counter()
    for position, record in enumerate(records, start=1):
        source = root / record.video_path
        if not source.is_file():
            raise ValueError(f"MediaPipe keypoint file not found: {source}")
        array = validate_raw_keypoints(load_mediapipe_array(source))
        sequences[record.sample_id] = array
        total_frames += int(array.shape[0])
        if args.progress_every and (
            position == 1 or position == len(records) or position % args.progress_every == 0
        ):
            elapsed = time.perf_counter() - started_at
            rate = position / elapsed if elapsed else 0.0
            eta = (len(records) - position) / rate / 60 if rate else 0.0
            print(
                f"[pack] {position}/{len(records)} | frames={total_frames:,} | "
                f"ETA={eta:.1f} min",
                flush=True,
            )

    metadata = {
        "schema_version": 1,
        "format": "vsl_mediapipe_holistic_76",
        "dataset_handle": args.dataset_handle,
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
        "source_root": str(root),
        "samples": len(records),
        "frames": total_frames,
        "joint_shape": ["T", 76, 3],
        "model_layout": "mediapipe_upper68_v1",
    }
    write_packed_keypoints(output_path, sequences, metadata)
    report = {
        "schema_version": 1,
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_handle": args.dataset_handle,
        "source_sha256": source_sha256,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "samples": len(records),
        "frames": total_frames,
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _selected_source_sha256(records: Sequence[ManifestRecord], root: Path) -> str:
    """Fingerprint every selected source file in manifest order before pack reuse."""

    digest = hashlib.sha256()
    for record in records:
        source = root / record.video_path
        if not source.is_file():
            raise ValueError(f"MediaPipe keypoint file not found: {source}")
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.video_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[MediaPipeGraphPreprocessConfig, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("MediaPipe graph config must be a schema_version 1 mapping.")
    preprocessing = payload.get("preprocessing")
    expected = payload.get("expected", {})
    if not isinstance(preprocessing, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("MediaPipe graph config requires preprocessing and expected mappings.")
    config = MediaPipeGraphPreprocessConfig(
        layout_name=str(preprocessing.get("layout_name")),
        target_frames=int(preprocessing.get("target_frames")),
        interpolation_max_gap=int(preprocessing.get("interpolation_max_gap")),
        zero_epsilon=float(preprocessing.get("zero_epsilon", 1e-8)),
    )
    return config, dict(expected)


def _progress(
    position: int,
    total: int,
    action: str,
    summary: Mapping[str, Any],
    every: int,
    started_at: float,
) -> None:
    if not every or (position != 1 and position != total and position % every):
        return
    elapsed = time.perf_counter() - started_at
    rate = position / elapsed if elapsed > 0 else 0.0
    remaining = (total - position) / rate if rate > 0 else 0.0
    print(
        f"[{action}] {position}/{total} | prepared={summary['prepared']} | "
        f"resumed={summary['resumed']} | failed={summary['failed']} | "
        f"ETA={remaining / 60:.1f} min",
        flush=True,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    entrypoint()
