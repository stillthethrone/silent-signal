"""Build and serialize canonical manifests with dataset-specific ingestion."""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from silent_signal.configuration import DatasetConfig
from silent_signal.contracts import LabelDefinition, ManifestRecord

_VIDEO_ID_KEYS = ("video_id", "clip_id", "id")
_SIGNER_ID_KEYS = ("signer_id", "signer", "performer_id", "subject_id")
_GLOSS_ID_KEYS = ("gloss_id", "label_id", "class_id")
_GLOSS_NAME_KEYS = ("gloss", "gloss_name", "label", "word")
_FRAME_COUNT_KEYS = ("num_of_frames", "frame_count", "num_frames", "frames")
_DURATION_KEYS = ("length", "runtime", "duration", "duration_seconds")
_FPS_KEYS = ("fps", "frame_rate")
_WIDTH_KEYS = ("width", "video_width")
_HEIGHT_KEYS = ("height", "video_height")
_RESOLUTION_KEYS = ("resolution", "video_resolution")
_CONTAINER_KEYS = ("records", "data", "videos", "annotations", "items")
_RESOLUTION_PATTERN = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


class ManifestError(ValueError):
    """Raised when source metadata cannot satisfy the manifest contract."""


@dataclass(frozen=True, slots=True)
class ManifestBuildResult:
    """Manifest records plus the exact class vocabulary used to build them."""

    records: tuple[ManifestRecord, ...]
    labels: tuple[LabelDefinition, ...]
    source_metadata: dict[str, str]
    source_video_directories: dict[str, str]


@dataclass(frozen=True, slots=True)
class _RawRecord:
    video_id: str
    signer_id: str
    gloss_id: str | None
    gloss_name: str
    view: str
    video_path: str
    frame_count: int | None
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None


def build_manifest(config: DatasetConfig) -> ManifestBuildResult:
    """Dispatch source parsing while preserving the shared manifest contract."""

    if config.adapter == "asl_citizen":
        from silent_signal.data.adapters.asl_citizen import build_asl_citizen_manifest

        return build_asl_citizen_manifest(config)
    if config.adapter != "vsl400":
        raise ManifestError(f"Unsupported dataset adapter: {config.adapter!r}.")
    return _build_vsl400_manifest(config)


def _build_vsl400_manifest(config: DatasetConfig) -> ManifestBuildResult:
    """Preserve the existing VSL400 multiview JSON normalization."""

    if not config.root.is_dir():
        raise ManifestError(f"Dataset root does not exist: {config.root}")

    raw_records: list[_RawRecord] = []
    metadata_sources: dict[str, str] = {}
    video_directories: dict[str, str] = {}
    for view, view_config in config.views.items():
        metadata_path = view_config.resolve_metadata(config.root)
        video_directory = view_config.resolve_directory(config.root)
        metadata_sources[view] = metadata_path.relative_to(config.root).as_posix()
        video_directories[view] = video_directory.relative_to(config.root).as_posix()

        rows = _read_json_records(metadata_path)
        for row_index, row in enumerate(rows):
            try:
                raw_records.append(
                    _normalize_row(
                        row,
                        view=view,
                        video_directory=video_directory,
                        dataset_root=config.root,
                        video_extension=config.video_extension,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ManifestError(
                    f"Invalid record {row_index} in {metadata_path}: {exc}"
                ) from exc

    glossary_path = config.root / config.gloss_file if config.gloss_file else None
    glossary = _read_glossary(glossary_path) if glossary_path and glossary_path.exists() else ()
    labels = _build_labels(raw_records, glossary)
    labels_by_name = {_label_key(item.gloss_name): item for item in labels}
    labels_by_id = {item.gloss_id: item for item in labels}

    records: list[ManifestRecord] = []
    for raw in raw_records:
        label = (
            labels_by_id.get(raw.gloss_id)
            if raw.gloss_id is not None
            else labels_by_name.get(_label_key(raw.gloss_name))
        )
        if label is None:
            raise ManifestError(
                f"Gloss {raw.gloss_name!r} ({raw.gloss_id!r}) has no label definition."
            )
        records.append(
            ManifestRecord(
                sample_id=f"{raw.video_id}_{raw.view}",
                instance_id=raw.video_id,
                video_id=raw.video_id,
                signer_id=raw.signer_id,
                gloss_id=label.gloss_id,
                gloss_name=label.gloss_name,
                class_index=label.class_index,
                view=raw.view,
                video_path=raw.video_path,
                metadata_frame_count=raw.frame_count,
                metadata_duration_seconds=raw.duration_seconds,
                metadata_fps=raw.fps,
                metadata_width=raw.width,
                metadata_height=raw.height,
            )
        )

    records.sort(key=lambda item: (item.instance_id, item.view, item.sample_id))
    return ManifestBuildResult(
        records=tuple(records),
        labels=labels,
        source_metadata=metadata_sources,
        source_video_directories=video_directories,
    )


def write_manifest(records: Sequence[ManifestRecord], path: str | Path) -> None:
    """Write a manifest as CSV or Parquet, selected by file suffix."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        _write_csv(records, destination)
        return
    if destination.suffix.lower() in {".parquet", ".pq"}:
        _write_parquet(records, destination)
        return
    raise ManifestError(f"Unsupported manifest format: {destination.suffix}")


def read_manifest(path: str | Path) -> tuple[ManifestRecord, ...]:
    """Read a CSV or Parquet manifest."""

    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"Manifest not found: {source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return tuple(ManifestRecord.from_dict(dict(row)) for row in csv.DictReader(handle))
    if source.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ManifestError(
                "Reading Parquet requires pyarrow; install the project dependencies."
            ) from exc
        return tuple(
            ManifestRecord.from_dict(row) for row in parquet.read_table(source).to_pylist()
        )
    raise ManifestError(f"Unsupported manifest format: {source.suffix}")


def write_labels(
    labels: Sequence[LabelDefinition], path: str | Path, *, dataset: str = "vsl400"
) -> None:
    """Persist both directions of the class vocabulary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "num_classes": len(labels),
        "labels": [asdict(item) for item in labels],
        "gloss_to_class_index": {item.gloss_name: item.class_index for item in labels},
        "class_index_to_gloss": {str(item.class_index): item.gloss_name for item in labels},
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def manifest_summary(records: Sequence[ManifestRecord]) -> dict[str, Any]:
    """Return compact, JSON-safe manifest statistics."""

    return {
        "clips": len(records),
        "instances": len({item.instance_id for item in records}),
        "signers": len({item.signer_id for item in records}),
        "glosses": len({item.class_index for item in records}),
        "views": {
            view: sum(item.view == view for item in records)
            for view in sorted({item.view for item in records})
        },
        "valid_clips": sum(item.is_valid for item in records),
        "invalid_clips": sum(not item.is_valid for item in records),
        "splits": {
            split: sum(item.split == split for item in records)
            for split in sorted({item.split for item in records if item.split})
        },
    }


def _normalize_row(
    row: Mapping[str, Any],
    *,
    view: str,
    video_directory: Path,
    dataset_root: Path,
    video_extension: str,
) -> _RawRecord:
    video_id = _normalize_identifier(_required_value(row, _VIDEO_ID_KEYS, "video_id"), width=6)
    signer_id = _normalize_identifier(_required_value(row, _SIGNER_ID_KEYS, "signer_id"), width=3)
    gloss_name = str(_required_value(row, _GLOSS_NAME_KEYS, "gloss")).strip()
    if not gloss_name:
        raise ValueError("gloss must not be empty")
    raw_gloss_id = _first_value(row, _GLOSS_ID_KEYS)
    gloss_id = _normalize_identifier(raw_gloss_id) if raw_gloss_id is not None else None
    width, height = _parse_resolution(row)
    video_path = (video_directory / f"{video_id}{video_extension}").relative_to(dataset_root)
    return _RawRecord(
        video_id=video_id,
        signer_id=signer_id,
        gloss_id=gloss_id,
        gloss_name=gloss_name,
        view=view,
        video_path=video_path.as_posix(),
        frame_count=_optional_int(_first_value(row, _FRAME_COUNT_KEYS)),
        duration_seconds=_optional_float(_first_value(row, _DURATION_KEYS)),
        fps=_optional_float(_first_value(row, _FPS_KEYS)),
        width=width,
        height=height,
    )


def _build_labels(
    records: Sequence[_RawRecord],
    glossary: Sequence[tuple[str, str]],
) -> tuple[LabelDefinition, ...]:
    names_by_id: dict[str, str] = {}
    ids_by_name: dict[str, str] = {}

    for gloss_id, gloss_name in glossary:
        _register_label(names_by_id, ids_by_name, gloss_id, gloss_name, "glossary")
    for record in records:
        resolved_id = record.gloss_id or ids_by_name.get(_label_key(record.gloss_name))
        if resolved_id is None:
            resolved_id = record.gloss_name
        _register_label(
            names_by_id,
            ids_by_name,
            resolved_id,
            record.gloss_name,
            f"video {record.video_id}",
        )

    ordered = sorted(
        names_by_id.items(),
        key=lambda item: _label_sort_key(item[0], item[1]),
    )
    return tuple(
        LabelDefinition(class_index=index, gloss_id=gloss_id, gloss_name=gloss_name)
        for index, (gloss_id, gloss_name) in enumerate(ordered)
    )


def _register_label(
    names_by_id: dict[str, str],
    ids_by_name: dict[str, str],
    gloss_id: str,
    gloss_name: str,
    source: str,
) -> None:
    gloss_id = _normalize_identifier(gloss_id)
    gloss_name = gloss_name.strip()
    name_key = _label_key(gloss_name)
    previous_name = names_by_id.get(gloss_id)
    previous_id = ids_by_name.get(name_key)
    if previous_name is not None and _label_key(previous_name) != name_key:
        raise ManifestError(
            f"Gloss id {gloss_id!r} maps to both {previous_name!r} and {gloss_name!r} ({source})."
        )
    if previous_id is not None and previous_id != gloss_id:
        raise ManifestError(
            f"Gloss {gloss_name!r} maps to both {previous_id!r} and {gloss_id!r} ({source})."
        )
    names_by_id[gloss_id] = gloss_name
    ids_by_name[name_key] = gloss_id


def _read_glossary(path: Path) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ManifestError(f"Glossary row {index + 1} has fewer than two columns.")
            first, second = row[0].strip(), row[1].strip()
            if index == 0 and first.casefold() in {"id", "gloss_id", "class_id"}:
                continue
            rows.append((_normalize_identifier(first), second))
    if not rows:
        raise ManifestError(f"Glossary is empty: {path}")
    return tuple(rows)


def _read_json_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(payload, list):
        return _ensure_mappings(payload, path)
    if not isinstance(payload, Mapping):
        raise ManifestError(f"Metadata root must be an object or list: {path}")

    for key in _CONTAINER_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return _ensure_mappings(candidate, path)

    if any(key in payload for key in _VIDEO_ID_KEYS):
        columnar = _column_oriented_records(payload)
        if columnar:
            return columnar
        return (payload,)

    if payload and all(isinstance(value, Mapping) for value in payload.values()):
        rows: list[Mapping[str, Any]] = []
        for key, value in payload.items():
            row = dict(value)
            row.setdefault("video_id", key)
            rows.append(row)
        return tuple(rows)
    raise ManifestError(f"Unsupported JSON metadata shape: {path}")


def _column_oriented_records(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    columns = {key: value for key, value in payload.items() if isinstance(value, Mapping)}
    if not columns:
        list_columns = {key: value for key, value in payload.items() if isinstance(value, list)}
        if not list_columns:
            return ()
        lengths = {len(value) for value in list_columns.values()}
        if len(lengths) != 1:
            raise ManifestError("Column-oriented JSON arrays have inconsistent lengths.")
        size = lengths.pop()
        return tuple(
            {key: values[index] for key, values in list_columns.items()} for index in range(size)
        )

    indices: set[str] = set()
    for values in columns.values():
        indices.update(str(index) for index in values)
    return tuple(
        {key: values.get(index, values.get(_maybe_int(index))) for key, values in columns.items()}
        for index in sorted(indices, key=_index_sort_key)
    )


def _ensure_mappings(values: Iterable[Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(values)
    if not all(isinstance(value, Mapping) for value in rows):
        raise ManifestError(f"Every metadata record must be an object: {path}")
    return tuple(value for value in rows if isinstance(value, Mapping))


def _parse_resolution(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    width = _optional_int(_first_value(row, _WIDTH_KEYS))
    height = _optional_int(_first_value(row, _HEIGHT_KEYS))
    if width is not None or height is not None:
        return width, height
    resolution = _first_value(row, _RESOLUTION_KEYS)
    if resolution is None:
        return None, None
    if isinstance(resolution, (int, float)):
        side = int(resolution)
        return side, side
    if (
        isinstance(resolution, Sequence)
        and not isinstance(resolution, str)
        and len(resolution) >= 2
    ):
        return int(resolution[0]), int(resolution[1])
    match = _RESOLUTION_PATTERN.match(str(resolution))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _write_csv(records: Sequence[ManifestRecord], destination: Path) -> None:
    fieldnames = list(ManifestRecord.__dataclass_fields__)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict(csv_safe=True) for item in records)


def _write_parquet(records: Sequence[ManifestRecord], destination: Path) -> None:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ManifestError(
            "Writing Parquet requires pyarrow; install the project dependencies."
        ) from exc
    table = arrow.Table.from_pylist([item.to_dict() for item in records])
    parquet.write_table(table, destination, compression="zstd")


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _required_value(row: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    value = _first_value(row, keys)
    if value is None:
        raise ValueError(f"missing required field {label}; accepted aliases: {', '.join(keys)}")
    return value


def _normalize_identifier(value: Any, *, width: int | None = None) -> str:
    if value is None:
        raise ValueError("identifier must not be null")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identifier must be finite")
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("identifier must not be empty")
    if width is not None and text.isdecimal():
        return text.zfill(width)
    return text


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _label_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.casefold().split())


def _label_sort_key(gloss_id: str, gloss_name: str) -> tuple[int, int | str, str]:
    if gloss_id.lstrip("+-").isdigit():
        return (0, int(gloss_id), _label_key(gloss_name))
    return (1, gloss_id.casefold(), _label_key(gloss_name))


def _maybe_int(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def _index_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value)
