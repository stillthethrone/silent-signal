"""CLI for reproducible RTMPose-L WholeBody extraction from canonical manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from silent_signal.contracts import ManifestRecord, SplitName
from silent_signal.data.manifest import ManifestError, read_manifest
from silent_signal.pose.cache import (
    PoseCacheError,
    cache_is_current,
    pose_cache_path,
    sha256_file,
    write_json_atomic,
    write_pose_cache,
)
from silent_signal.pose.interface import (
    PoseExtractionError,
    RawPoseSequence,
    WholeBodyExtractor,
)
from silent_signal.pose.rtmpose import (
    RTMPoseConfigurationError,
    RTMPoseWholeBodyExtractor,
    load_rtmpose_config,
)

_DEFAULT_CONFIG = Path("configs/pose/rtmpose.yaml")
_DEFAULT_MANIFEST = Path("data/manifests/asl_citizen.parquet")
_DEFAULT_OUTPUT_ROOT = Path("data/processed/pose/asl_citizen/rtmpose_l_coco_wholebody_384x288/raw")
_DEFAULT_REPORT = Path("artifacts/runs/pose-extraction/rtmpose_l_coco_wholebody_384x288.json")


def build_parser() -> argparse.ArgumentParser:
    """Build a parser without importing OpenMMLab or initializing a GPU."""

    parser = argparse.ArgumentParser(
        prog="ss-extract-pose",
        description=(
            "Extract raw COCO-WholeBody keypoints using explicit RTMDet and "
            "RTMPose-L 384x288 artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify",
        help=(
            "Verify artifact paths and print their full SHA-256 provenance without loading models."
        ),
    )
    verify.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    verify.add_argument(
        "--write-lock",
        type=Path,
        help="Optional JSON destination for the verified model/config hashes.",
    )

    extract = subparsers.add_parser(
        "extract",
        help="Extract every selected valid manifest video into an atomic raw cache.",
    )
    extract.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    extract.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    extract.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root; defaults to ASL_CITIZEN_ROOT.",
    )
    extract.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    extract.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    extract.add_argument(
        "--device",
        help="OpenMMLab device override, for example cuda:1 or cpu.",
    )
    extract.add_argument("--split", choices=[name.value for name in SplitName])
    extract.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Extract only this sample ID; repeat for multiple samples.",
    )
    extract.add_argument("--limit", type=int, help="Limit selected records for a pilot run.")
    extract.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically partition selected sample IDs across N workers.",
    )
    extract.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard assigned to this process.",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cache files, even when their provenance matches.",
    )
    extract.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failure and continue with later clips.",
    )
    extract.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Write progress to stderr every N processed records; 0 disables it.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    extractor_factory: Any = RTMPoseWholeBodyExtractor,
) -> int:
    """Run the pose CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        config = load_rtmpose_config(args.config)
        if args.command == "verify":
            provenance = config.verified_provenance(require_pinned_checkpoints=False)
            if args.write_lock:
                write_json_atomic(args.write_lock, provenance)
            _print_json(provenance)
            return 0

        if args.command == "extract":
            if args.device:
                config = replace(config, device=args.device)
            dataset_root = _dataset_root(args.dataset_root)
            records = _select_records(
                read_manifest(args.manifest),
                split=args.split,
                sample_ids=frozenset(args.sample_id),
                limit=args.limit,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )
            extractor: WholeBodyExtractor = extractor_factory(config)
            return _extract_records(
                records,
                dataset_root=dataset_root,
                output_root=args.output_root.resolve(),
                report_path=args.report.resolve(),
                extractor=extractor,
                overwrite=args.overwrite,
                continue_on_error=args.continue_on_error,
                progress_every=args.progress_every,
                hash_source_video=config.hash_source_video,
                selection_metadata={
                    "manifest_path": str(args.manifest.resolve()),
                    "manifest_sha256": sha256_file(args.manifest.resolve()),
                    "split": args.split,
                    "sample_ids": sorted(args.sample_id),
                    "limit": args.limit,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "device": config.device,
                },
            )
    except (
        ManifestError,
        PoseCacheError,
        PoseExtractionError,
        RTMPoseConfigurationError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def entrypoint() -> None:
    """Console-script entry point."""

    raise SystemExit(main())


def _extract_records(
    records: Sequence[ManifestRecord],
    *,
    dataset_root: Path,
    output_root: Path,
    report_path: Path,
    extractor: WholeBodyExtractor,
    overwrite: bool,
    continue_on_error: bool,
    progress_every: int,
    hash_source_video: bool,
    selection_metadata: dict[str, Any],
) -> int:
    if progress_every < 0:
        raise ValueError("--progress-every must not be negative.")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "extractor_fingerprint": extractor.fingerprint,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "selected": len(records),
        "selection": selection_metadata,
        "extracted": 0,
        "resumed": 0,
        "failed": 0,
        "failures": [],
    }
    for position, record in enumerate(records, start=1):
        cache_path = pose_cache_path(output_root, record.sample_id)
        try:
            video_path = _safe_video_path(dataset_root, record.video_path)
            if cache_path.is_file() and not overwrite:
                video_sha256 = sha256_file(video_path) if hash_source_video else None
                if video_sha256 is None:
                    raise PoseCacheError(
                        "Resume requires hash_source_video=true so stale source clips are detected."
                    )
                if cache_is_current(
                    cache_path,
                    sample_id=record.sample_id,
                    extractor_fingerprint=extractor.fingerprint,
                    video_sha256=video_sha256,
                ):
                    summary["resumed"] += 1
                    _progress(position, len(records), record.sample_id, "resume", progress_every)
                    continue
                raise PoseCacheError(
                    f"Stale or corrupt cache exists for {record.sample_id}; "
                    "inspect it or pass --overwrite."
                )

            sequence = extractor.extract_video(video_path, sample_id=record.sample_id)
            _validate_sequence(sequence, record, extractor.fingerprint)
            write_pose_cache(cache_path, sequence, overwrite=overwrite)
            summary["extracted"] += 1
            _progress(position, len(records), record.sample_id, "extract", progress_every)
        except (OSError, PoseCacheError, PoseExtractionError, ValueError) as exc:
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "sample_id": record.sample_id,
                    "video_path": record.video_path,
                    "error": str(exc),
                }
            )
            print(f"failed {record.sample_id}: {exc}", file=sys.stderr)
            if not continue_on_error:
                break

    write_json_atomic(report_path, summary)
    _print_json({key: value for key, value in summary.items() if key != "failures"})
    return 1 if summary["failed"] else 0


def _select_records(
    records: Sequence[ManifestRecord],
    *,
    split: str | None,
    sample_ids: frozenset[str],
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> tuple[ManifestRecord, ...]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in the range [0, num-shards).")
    eligible = tuple(
        sorted(
            (
                record
                for record in records
                if record.is_valid
                and (split is None or record.split == split)
                and (not sample_ids or record.sample_id in sample_ids)
            ),
            key=lambda record: record.sample_id,
        )
    )
    if sample_ids:
        missing = sorted(sample_ids.difference(record.sample_id for record in eligible))
        if missing:
            raise ValueError(f"Requested sample IDs are absent or invalid: {missing}")
    selected = tuple(
        record for index, record in enumerate(eligible) if index % num_shards == shard_index
    )
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No valid manifest records match the requested filters.")
    return selected


def _validate_sequence(
    sequence: RawPoseSequence,
    record: ManifestRecord,
    extractor_fingerprint: str,
) -> None:
    """Prevent cache publication when extraction and manifest identities diverge."""

    if sequence.sample_id != record.sample_id:
        raise PoseExtractionError(
            f"Extractor returned sample_id {sequence.sample_id!r} for {record.sample_id!r}."
        )
    if sequence.metadata.get("extractor_fingerprint") != extractor_fingerprint:
        raise PoseExtractionError("Extractor output does not carry the active fingerprint.")
    if record.frame_count is not None and abs(sequence.frame_count - record.frame_count) > 1:
        raise PoseExtractionError(
            f"Decoded {sequence.frame_count} frames; manifest reports {record.frame_count}."
        )
    if record.height is not None and record.width is not None:
        expected_size = (record.height, record.width)
    else:
        expected_size = None
    if expected_size is not None and sequence.frame_size_hw != expected_size:
        raise PoseExtractionError(
            f"Decoded frame size {sequence.frame_size_hw}; manifest reports {expected_size}."
        )


def _dataset_root(value: Path | None) -> Path:
    if value is None:
        configured = os.environ.get("ASL_CITIZEN_ROOT")
        if not configured:
            raise ValueError("Set ASL_CITIZEN_ROOT or pass --dataset-root.")
        value = Path(configured)
    root = value.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    return root


def _safe_video_path(root: Path, relative_path: str) -> Path:
    value = Path(relative_path)
    if value.is_absolute():
        raise ValueError(f"Manifest video path must be relative: {relative_path}")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest video path escapes dataset root: {relative_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"Manifest video is missing: {candidate}")
    return candidate


def _progress(
    position: int,
    total: int,
    sample_id: str,
    action: str,
    every: int,
) -> None:
    if every and (position == 1 or position == total or position % every == 0):
        print(f"[{position}/{total}] {action}: {sample_id}", file=sys.stderr)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
