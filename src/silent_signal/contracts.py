"""Stable data contracts shared by ingestion, training, and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class View(StrEnum):
    """Camera labels; single makes no synchronized multiview claim."""

    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    SINGLE = "single"


class SplitName(StrEnum):
    """Supported dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ValidationLevel(StrEnum):
    """Increasingly expensive video validation modes."""

    METADATA = "metadata"
    PROBE = "probe"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """One stable mapping between source label and contiguous model index."""

    class_index: int
    gloss_id: str
    gloss_name: str


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One physical video clip in the normalized manifest."""

    sample_id: str
    instance_id: str
    video_id: str
    signer_id: str
    gloss_id: str
    gloss_name: str
    class_index: int
    view: str
    video_path: str
    metadata_frame_count: int | None = None
    metadata_duration_seconds: float | None = None
    metadata_fps: float | None = None
    metadata_width: int | None = None
    metadata_height: int | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    file_size_bytes: int | None = None
    is_valid: bool = True
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    split: str | None = None
    asl_lex_code: str | None = None

    def to_dict(self, *, csv_safe: bool = False) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        value = asdict(self)
        if csv_safe:
            value["validation_errors"] = json.dumps(self.validation_errors, ensure_ascii=False)
        else:
            value["validation_errors"] = list(self.validation_errors)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ManifestRecord:
        """Create a record from CSV, Parquet, or JSON-compatible values."""

        normalized = dict(value)
        errors = normalized.get("validation_errors")
        if isinstance(errors, str):
            try:
                errors = json.loads(errors)
            except json.JSONDecodeError:
                errors = [item for item in errors.split("|") if item]
        normalized["validation_errors"] = tuple(errors or ())

        for key in (
            "metadata_frame_count",
            "metadata_width",
            "metadata_height",
            "frame_count",
            "width",
            "height",
            "file_size_bytes",
        ):
            normalized[key] = _optional_int(normalized.get(key))
        for key in (
            "metadata_duration_seconds",
            "metadata_fps",
            "duration_seconds",
            "fps",
        ):
            normalized[key] = _optional_float(normalized.get(key))
        normalized["class_index"] = int(normalized["class_index"])
        normalized["is_valid"] = _as_bool(normalized.get("is_valid", True))
        normalized["split"] = _optional_str(normalized.get("split"))
        normalized["codec"] = _optional_str(normalized.get("codec"))
        normalized["asl_lex_code"] = _optional_str(normalized.get("asl_lex_code"))
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A machine-readable validation finding."""

    code: str
    severity: str
    message: str
    sample_id: str | None = None
    instance_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    """Reproducible signer allocation and its measured statistics."""

    seed: int | None
    target_ratios: dict[str, float]
    signer_ids: dict[str, tuple[str, ...]]
    signer_counts: dict[str, int]
    instance_counts: dict[str, int]
    clip_counts: dict[str, int]
    gloss_counts: dict[str, int]
    score: float | None
    strategy: str = "signer_disjoint"
    source_files: dict[str, dict[str, str]] = field(default_factory=dict)
    assignments_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = 1
        value["strategy"] = self.strategy
        if self.strategy == "official":
            total = sum(self.clip_counts.values())
            value["observed_ratios"] = {
                split: count / total if total else 0.0 for split, count in self.clip_counts.items()
            }
        if not self.source_files:
            value.pop("source_files")
        if self.assignments_sha256 is None:
            value.pop("assignments_sha256")
        value["signer_ids"] = {split: list(signers) for split, signers in self.signer_ids.items()}
        return value


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
