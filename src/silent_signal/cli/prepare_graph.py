"""CLI for resumable conversion of raw pose caches into graph-ready tensors."""

from __future__ import annotations

import argparse
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
from silent_signal.data.manifest import ManifestError, read_manifest
from silent_signal.pose.cache import (
    PoseCacheError,
    pose_cache_path,
    read_pose_cache,
    sha256_file,
    write_json_atomic,
)
from silent_signal.preprocessing.cache import (
    read_graph_pose_cache,
    write_graph_pose_cache,
)
from silent_signal.preprocessing.pose_features import (
    GraphPreprocessConfig,
    prepare_graph_pose,
    preprocessing_fingerprint,
)

_DEFAULT_CONFIG = Path("configs/preprocessing/asl_citizen_graph.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the graph-preparation CLI parser."""

    parser = argparse.ArgumentParser(
        prog="ss-prepare-pose-graph",
        description="Prepare fixed graph tensors from versioned raw pose caches.",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run graph preprocessing and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        graph_config, expected = _load_config(args.config)
        records = tuple(sorted(read_manifest(args.manifest), key=lambda item: item.sample_id))
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be at least 1.")
            records = records[: args.limit]
        if not records:
            raise ValueError("Manifest contains no records.")
        if args.progress_every < 0:
            raise ValueError("--progress-every must not be negative.")
        manifest_sha256 = sha256_file(args.manifest)
        _validate_manifest(records, expected, manifest_sha256, limited=args.limit is not None)
        return _prepare_records(
            records,
            pose_root=args.pose_root.resolve(),
            output_root=args.output_root.resolve(),
            report_path=args.report.resolve(),
            config=graph_config,
            expected=expected,
            manifest_path=args.manifest.resolve(),
            manifest_sha256=manifest_sha256,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
            progress_every=args.progress_every,
        )
    except (ManifestError, OSError, PoseCacheError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    """Console-script entry point."""

    raise SystemExit(main())


def _load_config(path: Path) -> tuple[GraphPreprocessConfig, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Graph config must be a schema_version 1 mapping.")
    preprocessing = payload.get("preprocessing")
    expected = payload.get("expected", {})
    if not isinstance(preprocessing, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("Graph config requires preprocessing and expected mappings.")
    config = GraphPreprocessConfig(
        layout_name=str(preprocessing.get("layout_name")),
        target_frames=int(preprocessing.get("target_frames")),
        confidence_threshold=float(preprocessing.get("confidence_threshold")),
        interpolation_max_gap=int(preprocessing.get("interpolation_max_gap")),
        include_velocity=bool(preprocessing.get("include_velocity", True)),
        include_bones=bool(preprocessing.get("include_bones", True)),
    )
    return config, dict(expected)


def _validate_manifest(
    records: Sequence[ManifestRecord],
    expected: Mapping[str, Any],
    manifest_sha256: str,
    *,
    limited: bool,
) -> None:
    if any(not record.is_valid for record in records):
        raise ValueError("Graph preparation refuses invalid manifest records.")
    if any(record.split not in {"train", "validation", "test"} for record in records):
        raise ValueError("Every graph record must preserve an official split.")
    if limited:
        return
    expected_sha = expected.get("manifest_sha256")
    if expected_sha and expected_sha != manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected {expected_sha}, found {manifest_sha256}."
        )
    expected_clips = expected.get("clips")
    if expected_clips is not None and int(expected_clips) != len(records):
        raise ValueError(f"Expected {expected_clips} clips, found {len(records)}.")
    expected_classes = expected.get("classes")
    classes = {record.class_index for record in records}
    if expected_classes is not None and int(expected_classes) != len(classes):
        raise ValueError(f"Expected {expected_classes} classes, found {len(classes)}.")
    expected_splits = expected.get("splits", {})
    if expected_splits:
        actual_splits = Counter(record.split for record in records)
        if dict(expected_splits) != dict(actual_splits):
            raise ValueError(
                f"Official split counts changed: expected {dict(expected_splits)}, "
                f"found {dict(actual_splits)}."
            )


def _prepare_records(
    records: Sequence[ManifestRecord],
    *,
    pose_root: Path,
    output_root: Path,
    report_path: Path,
    config: GraphPreprocessConfig,
    expected: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    overwrite: bool,
    continue_on_error: bool,
    progress_every: int,
) -> int:
    fingerprint = preprocessing_fingerprint(config)
    expected_extractor = expected.get("extractor_fingerprint")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "pose_root": str(pose_root),
        "output_root": str(output_root),
        "selected": len(records),
        "preprocessing": asdict(config),
        "preprocessing_fingerprint": fingerprint,
        "expected_extractor_fingerprint": expected_extractor,
        "prepared": 0,
        "resumed": 0,
        "failed": 0,
        "failures": [],
        "split_counts": dict(Counter(record.split for record in records)),
    }
    started_at = time.perf_counter()
    expected_names = {pose_cache_path(output_root, record.sample_id).name for record in records}
    existing = sum(path.name in expected_names for path in output_root.glob("*/*.npz"))
    print(
        f"[resume] found {existing}/{len(records)} graph caches; validating each before skip.",
        file=sys.stderr,
        flush=True,
    )
    observed_ratios: list[float] = []
    usable_ratios: list[float] = []
    for position, record in enumerate(records, start=1):
        destination = pose_cache_path(output_root, record.sample_id)
        try:
            if destination.is_file() and not overwrite:
                cached = read_graph_pose_cache(
                    destination,
                    expected_sample_id=record.sample_id,
                    expected_fingerprint=fingerprint,
                )
                if cached.class_index != record.class_index or cached.split != record.split:
                    raise PoseCacheError(
                        f"Graph cache label/split mismatch for {record.sample_id}."
                    )
                summary["resumed"] += 1
                _progress(
                    position,
                    len(records),
                    record.sample_id,
                    "resume",
                    summary,
                    progress_every,
                    started_at,
                )
                continue
            raw = read_pose_cache(
                pose_cache_path(pose_root, record.sample_id),
                expected_sample_id=record.sample_id,
                expected_fingerprint=str(expected_extractor) if expected_extractor else None,
            )
            sample = prepare_graph_pose(
                raw,
                class_index=record.class_index,
                split=str(record.split),
                config=config,
            )
            write_graph_pose_cache(destination, sample, overwrite=overwrite)
            summary["prepared"] += 1
            observed_ratios.append(float(sample.metadata["observed_joint_ratio"]))
            usable_ratios.append(float(sample.metadata["usable_joint_ratio"]))
            _progress(
                position,
                len(records),
                record.sample_id,
                "prepare",
                summary,
                progress_every,
                started_at,
            )
        except (OSError, PoseCacheError, ValueError) as exc:
            summary["failed"] += 1
            summary["failures"].append({"sample_id": record.sample_id, "error": str(exc)})
            print(f"failed {record.sample_id}: {exc}", file=sys.stderr, flush=True)
            _progress(
                position,
                len(records),
                record.sample_id,
                "failed",
                summary,
                progress_every,
                started_at,
            )
            if not continue_on_error:
                break
    summary["new_cache_statistics"] = {
        "mean_observed_joint_ratio": _mean(observed_ratios),
        "mean_usable_joint_ratio": _mean(usable_ratios),
    }
    summary["duration_seconds"] = round(time.perf_counter() - started_at, 3)
    write_json_atomic(report_path, summary)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2),
        flush=True,
    )
    return 1 if summary["failed"] else 0


def _progress(
    position: int,
    total: int,
    sample_id: str,
    action: str,
    summary: Mapping[str, Any],
    every: int,
    started_at: float,
) -> None:
    if every and (position == 1 or position == total or position % every == 0):
        elapsed = time.perf_counter() - started_at
        rate = position / elapsed if elapsed > 0 else 0.0
        remaining = (total - position) / rate if rate > 0 else 0.0
        print(
            f"[{position}/{total}] new={summary['prepared']} resumed={summary['resumed']} "
            f"failed={summary['failed']} action={action} sample={sample_id} "
            f"elapsed={_duration(elapsed)} rate={rate:.2f}/s eta={_duration(remaining)}",
            file=sys.stderr,
            flush=True,
        )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    entrypoint()
