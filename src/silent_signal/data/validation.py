"""Shared integrity checks for metadata and physical video files."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from silent_signal.configuration import ExpectedConfig
from silent_signal.contracts import ManifestRecord, SplitName, ValidationIssue, ValidationLevel
from silent_signal.data.manifest import manifest_summary


class ValidationError(RuntimeError):
    """Raised when a requested validation operation cannot be performed."""


class MediaProbeError(RuntimeError):
    """Raised when ffprobe cannot read a video stream."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Video properties returned by ffprobe."""

    frame_count: int | None
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Updated records and a complete machine-readable report."""

    records: tuple[ManifestRecord, ...]
    issues: tuple[ValidationIssue, ...]
    level: ValidationLevel

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "error" for item in self.issues)

    def to_report(self) -> dict[str, Any]:
        by_code: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            by_code[issue.code] += 1
            by_severity[issue.severity] += 1
        return {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "validation_level": self.level.value,
            "passed": not self.has_errors,
            "summary": manifest_summary(self.records),
            "issue_counts": {
                "by_severity": dict(sorted(by_severity.items())),
                "by_code": dict(sorted(by_code.items())),
            },
            "issues": [item.to_dict() for item in self.issues],
        }


ProbeFunction = Callable[[Path], MediaInfo]
DecodeFunction = Callable[[Path], None]


def validate_manifest(
    records: Sequence[ManifestRecord],
    *,
    dataset_root: str | Path,
    expected: ExpectedConfig,
    expected_views: Sequence[str],
    level: ValidationLevel | str = ValidationLevel.METADATA,
    workers: int = 4,
    probe_function: ProbeFunction | None = None,
    decode_function: DecodeFunction | None = None,
) -> ValidationResult:
    """Validate structural invariants, files, and optionally video bitstreams."""

    validation_level = ValidationLevel(level)
    root = Path(dataset_root).resolve()
    if workers < 1:
        raise ValidationError("workers must be at least 1")

    issues: list[ValidationIssue] = []
    errors_by_index: dict[int, set[str]] = defaultdict(set)
    media_by_index: dict[int, MediaInfo] = {}

    def add_issue(
        code: str,
        message: str,
        *,
        severity: str = "error",
        index: int | None = None,
        instance_id: str | None = None,
        path: str | None = None,
    ) -> None:
        record = records[index] if index is not None else None
        issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                sample_id=record.sample_id if record else None,
                instance_id=instance_id or (record.instance_id if record else None),
                path=path or (record.video_path if record else None),
            )
        )
        if severity == "error" and index is not None:
            errors_by_index[index].add(code)

    _validate_expected_counts(records, expected, add_issue)
    _validate_unique_samples(records, add_issue)
    _validate_instance_groups(records, expected_views, add_issue)
    if any(record.split is not None for record in records):
        _validate_split_membership(records, add_issue)

    file_paths: dict[int, Path] = {}
    for index, record in enumerate(records):
        _validate_metadata_fields(record, index, expected, add_issue)
        video_path = _safe_video_path(root, record.video_path)
        if video_path is None:
            add_issue(
                "path_outside_dataset_root",
                "Video path escapes the configured dataset root.",
                index=index,
            )
            continue
        file_paths[index] = video_path
        if not video_path.is_file():
            add_issue("video_missing", "Video file does not exist.", index=index)
            continue
        size = video_path.stat().st_size
        if size <= 0:
            add_issue("video_empty", "Video file is empty.", index=index)

    if validation_level in {ValidationLevel.PROBE, ValidationLevel.DECODE}:
        probe = probe_function or _configured_probe()
        eligible = [
            (index, path) for index, path in file_paths.items() if not errors_by_index.get(index)
        ]
        for index, result in _parallel_media_operation(eligible, probe, workers):
            if isinstance(result, Exception):
                add_issue(
                    "video_probe_failed",
                    f"ffprobe could not read the video: {result}",
                    index=index,
                )
                continue
            media_by_index[index] = result
            _validate_media_info(records[index], result, index, expected, add_issue)

    if validation_level is ValidationLevel.DECODE:
        decode = decode_function or _configured_decoder()
        eligible = [
            (index, path)
            for index, path in file_paths.items()
            if index in media_by_index and not errors_by_index.get(index)
        ]
        for index, result in _parallel_media_operation(eligible, decode, workers):
            if isinstance(result, Exception):
                add_issue(
                    "video_decode_failed",
                    f"ffmpeg could not decode the complete video: {result}",
                    index=index,
                )

    updated: list[ManifestRecord] = []
    for index, record in enumerate(records):
        current_errors = set(errors_by_index.get(index, set()))
        media = media_by_index.get(index)
        path = file_paths.get(index)
        updated.append(
            replace(
                record,
                frame_count=media.frame_count if media else record.frame_count,
                duration_seconds=media.duration_seconds if media else record.duration_seconds,
                fps=media.fps if media else record.fps,
                width=media.width if media else record.width,
                height=media.height if media else record.height,
                codec=media.codec if media else record.codec,
                file_size_bytes=(
                    path.stat().st_size if path is not None and path.is_file() else None
                ),
                is_valid=not current_errors,
                validation_errors=tuple(sorted(current_errors)),
            )
        )
    return ValidationResult(tuple(updated), tuple(issues), validation_level)


def probe_video(path: Path, *, executable: str = "ffprobe") -> MediaInfo:
    """Inspect one video using a JSON ffprobe response."""

    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:"
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise MediaProbeError(detail)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"invalid ffprobe JSON: {exc}") from exc
    return media_info_from_ffprobe(payload)


def media_info_from_ffprobe(payload: dict[str, Any]) -> MediaInfo:
    """Parse the subset of ffprobe JSON used by the data contract."""

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise MediaProbeError("no video stream")
    stream = streams[0]
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        format_info = {}
    fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
        stream.get("r_frame_rate")
    )
    duration = _positive_float(stream.get("duration")) or _positive_float(
        format_info.get("duration")
    )
    return MediaInfo(
        frame_count=_positive_int(stream.get("nb_frames")),
        duration_seconds=duration,
        fps=fps,
        width=_positive_int(stream.get("width")),
        height=_positive_int(stream.get("height")),
        codec=_optional_string(stream.get("codec_name")),
    )


def decode_video(path: Path, *, executable: str = "ffmpeg") -> None:
    """Decode a complete video while discarding frames."""

    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ValidationError(detail)


def write_validation_report(result: ValidationResult, path: str | Path) -> None:
    """Write the complete validation report as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_report(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_expected_counts(
    records: Sequence[ManifestRecord],
    expected: ExpectedConfig,
    add_issue: Callable[..., None],
) -> None:
    severity = "error" if expected.enforce_counts else "warning"
    checks = {
        "clips": (len(records), expected.clips),
        "signers": (len({item.signer_id for item in records}), expected.signers),
        "glosses": (len({item.class_index for item in records}), expected.glosses),
    }
    for label, (actual, target) in checks.items():
        if target is not None and actual != target:
            add_issue(
                f"unexpected_{label}_count",
                f"Expected {target} {label}, found {actual}.",
                severity=severity,
            )
    for split, target in expected.split_clips.items():
        actual = sum(record.split == split for record in records)
        if actual != target:
            add_issue(
                "unexpected_split_clip_count",
                f"Expected {target} clips in {split}, found {actual}.",
                severity=severity,
            )
    for split, target in expected.split_signers.items():
        actual = len({record.signer_id for record in records if record.split == split})
        if actual != target:
            add_issue(
                "unexpected_split_signer_count",
                f"Expected {target} signers in {split}, found {actual}.",
                severity=severity,
            )


def _validate_split_membership(
    records: Sequence[ManifestRecord], add_issue: Callable[..., None]
) -> None:
    valid_splits = {split.value for split in SplitName}
    signer_splits: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        if record.split not in valid_splits:
            add_issue("invalid_split", "Missing or unsupported split membership.", index=index)
        elif record.split is not None:
            signer_splits[record.signer_id].add(record.split)
    for index, record in enumerate(records):
        if len(signer_splits[record.signer_id]) > 1:
            add_issue(
                "signer_split_overlap",
                "Signer occurs in multiple train/validation/test partitions.",
                index=index,
            )


def _validate_unique_samples(
    records: Sequence[ManifestRecord],
    add_issue: Callable[..., None],
) -> None:
    indices_by_id: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_id[record.sample_id].append(index)
    for sample_id, indices in indices_by_id.items():
        if len(indices) <= 1:
            continue
        for index in indices:
            add_issue(
                "duplicate_sample_id",
                f"Sample id {sample_id!r} occurs {len(indices)} times.",
                index=index,
            )


def _validate_instance_groups(
    records: Sequence[ManifestRecord],
    expected_views: Sequence[str],
    add_issue: Callable[..., None],
) -> None:
    expected_set = set(expected_views)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.instance_id].append(index)

    for instance_id, indices in groups.items():
        views = {records[index].view for index in indices}
        if views != expected_set or len(indices) != len(expected_set):
            missing = sorted(expected_set - views)
            extra = sorted(views - expected_set)
            detail = f"expected {sorted(expected_set)}, found {sorted(views)}"
            if missing:
                detail += f"; missing {missing}"
            if extra:
                detail += f"; extra {extra}"
            for index in indices:
                add_issue(
                    (
                        "incomplete_multiview_instance"
                        if len(expected_set) > 1
                        else "invalid_singleview_instance"
                    ),
                    detail,
                    index=index,
                    instance_id=instance_id,
                )

        for field_name in ("signer_id", "gloss_id", "class_index", "gloss_name"):
            values = {getattr(records[index], field_name) for index in indices}
            if len(values) > 1:
                for index in indices:
                    add_issue(
                        f"inconsistent_instance_{field_name}",
                        f"Instance {instance_id!r} has inconsistent {field_name}: "
                        f"{sorted(str(value) for value in values)}.",
                        index=index,
                        instance_id=instance_id,
                    )


def _validate_metadata_fields(
    record: ManifestRecord,
    index: int,
    expected: ExpectedConfig,
    add_issue: Callable[..., None],
) -> None:
    required_strings = {
        "video_id": record.video_id,
        "signer_id": record.signer_id,
        "gloss_name": record.gloss_name,
        "view": record.view,
        "video_path": record.video_path,
    }
    for field_name, value in required_strings.items():
        if not value.strip():
            add_issue(
                f"missing_{field_name}",
                f"Required field {field_name} is empty.",
                index=index,
            )

    positive_fields = {
        "metadata_frame_count": record.metadata_frame_count,
        "metadata_duration_seconds": record.metadata_duration_seconds,
        "metadata_fps": record.metadata_fps,
        "metadata_width": record.metadata_width,
        "metadata_height": record.metadata_height,
    }
    for field_name, metadata_value in positive_fields.items():
        if metadata_value is not None and metadata_value <= 0:
            add_issue(
                f"invalid_{field_name}",
                f"{field_name} must be positive, found {metadata_value}.",
                index=index,
            )

    if (
        expected.fps is not None
        and record.metadata_fps is not None
        and abs(record.metadata_fps - expected.fps) > expected.fps_tolerance
    ):
        add_issue(
            "unexpected_metadata_fps",
            f"Expected metadata fps {expected.fps}, found {record.metadata_fps}.",
            index=index,
        )
    if (
        expected.width is not None
        and record.metadata_width is not None
        and record.metadata_width != expected.width
    ):
        add_issue(
            "unexpected_metadata_width",
            f"Expected metadata width {expected.width}, found {record.metadata_width}.",
            index=index,
        )
    if (
        expected.height is not None
        and record.metadata_height is not None
        and record.metadata_height != expected.height
    ):
        add_issue(
            "unexpected_metadata_height",
            f"Expected metadata height {expected.height}, found {record.metadata_height}.",
            index=index,
        )


def _validate_media_info(
    record: ManifestRecord,
    media: MediaInfo,
    index: int,
    expected: ExpectedConfig,
    add_issue: Callable[..., None],
) -> None:
    if media.fps is None:
        add_issue("video_fps_unavailable", "ffprobe did not return fps.", index=index)
    elif expected.fps is not None and abs(media.fps - expected.fps) > expected.fps_tolerance:
        add_issue(
            "unexpected_video_fps",
            f"Expected video fps {expected.fps}, found {media.fps}.",
            index=index,
        )
    for field_name in ("width", "height", "frame_count", "duration_seconds"):
        if getattr(media, field_name) is None:
            add_issue(
                f"video_{field_name}_unavailable",
                f"ffprobe did not return {field_name}.",
                severity="warning",
                index=index,
            )
    if expected.width is not None and media.width is not None and media.width != expected.width:
        add_issue(
            "unexpected_video_width",
            f"Expected video width {expected.width}, found {media.width}.",
            index=index,
        )
    if expected.height is not None and media.height is not None and media.height != expected.height:
        add_issue(
            "unexpected_video_height",
            f"Expected video height {expected.height}, found {media.height}.",
            index=index,
        )
    if (
        record.metadata_frame_count is not None
        and media.frame_count is not None
        and abs(record.metadata_frame_count - media.frame_count) > 1
    ):
        add_issue(
            "frame_count_mismatch",
            f"Metadata has {record.metadata_frame_count} frames; video has {media.frame_count}.",
            index=index,
        )
    duration_tolerance = max(0.1, 2.0 / media.fps) if media.fps else 0.1
    if (
        record.metadata_duration_seconds is not None
        and media.duration_seconds is not None
        and abs(record.metadata_duration_seconds - media.duration_seconds) > duration_tolerance
    ):
        add_issue(
            "duration_mismatch",
            f"Metadata duration is {record.metadata_duration_seconds}; video duration is "
            f"{media.duration_seconds}.",
            index=index,
        )


def _configured_probe() -> ProbeFunction:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ValidationError(
            "Validation level 'probe' requires ffprobe on PATH. "
            "Install FFmpeg or use --level metadata."
        )
    return lambda path: probe_video(path, executable=executable)


def _configured_decoder() -> DecodeFunction:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ValidationError(
            "Validation level 'decode' requires ffmpeg on PATH. "
            "Install FFmpeg or use --level probe/metadata."
        )
    return lambda path: decode_video(path, executable=executable)


def _parallel_media_operation(
    indexed_paths: Sequence[tuple[int, Path]],
    operation: Callable[[Path], Any],
    workers: int,
) -> list[tuple[int, Any | Exception]]:
    def execute(item: tuple[int, Path]) -> tuple[int, Any | Exception]:
        index, path = item
        try:
            return index, operation(path)
        except Exception as exc:  # Media tooling failures are converted to findings.
            return index, exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(execute, indexed_paths))


def _safe_video_path(root: Path, relative_path: str) -> Path | None:
    untrusted_path = Path(relative_path)
    if untrusted_path.is_absolute():
        return None
    candidate = (root / untrusted_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _parse_fraction(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    text = str(value)
    if "/" not in text:
        return _positive_float(text)
    numerator_text, denominator_text = text.split("/", maxsplit=1)
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError:
        return None
    if denominator == 0:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) and result > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
