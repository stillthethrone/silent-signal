"""Import ASL Citizen videos and their authoritative CSV split membership."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from silent_signal.configuration import DatasetConfig
from silent_signal.contracts import LabelDefinition, ManifestRecord, SplitName
from silent_signal.data.manifest import ManifestBuildResult, ManifestError

# Verified against ASL_Citizen/splits/{train,val,test}.csv in the official ZIP.
_REQUIRED_COLUMNS = ("Participant ID", "Video file", "Gloss", "ASL-LEX Code")


@dataclass(frozen=True, slots=True)
class _SourceRow:
    signer_id: str
    filename: str
    gloss: str
    split: str
    origin: str
    asl_lex_code: str | None


def build_asl_citizen_manifest(config: DatasetConfig) -> ManifestBuildResult:
    """Read the official splits without relabeling identities or reallocating videos."""

    root = config.root.resolve()
    if not root.is_dir():
        raise ManifestError(f"ASL Citizen dataset root does not exist: {root}")
    if set(config.views) != {"single"}:
        raise ManifestError("ASL Citizen requires exactly one configured view named 'single'.")
    video_directory = _inside_root(root, config.views["single"].directory, "video directory")
    if not video_directory.is_dir():
        raise ManifestError(f"ASL Citizen video directory does not exist: {video_directory}")

    expected_splits = {split.value for split in SplitName}
    if set(config.metadata_splits) != expected_splits:
        raise ManifestError(
            "ASL Citizen metadata_splits must specify train, validation, and test CSVs; "
            f"found {sorted(config.metadata_splits)}."
        )

    rows: list[_SourceRow] = []
    metadata_sources: dict[str, str] = {}
    filenames: dict[str, _SourceRow] = {}
    signer_splits: dict[str, _SourceRow] = {}
    for split_name in SplitName:
        split = split_name.value
        metadata_path = _inside_root(root, config.metadata_splits[split], f"{split} metadata")
        metadata_sources[split] = metadata_path.relative_to(root).as_posix()
        for row in _read_split(metadata_path, split):
            previous = filenames.get(row.filename)
            if previous is not None:
                raise ManifestError(
                    f"Duplicate ASL Citizen filename {row.filename!r} at {row.origin}; "
                    f"already present at {previous.origin} ({previous.split}). "
                    "Use the unmodified official split CSVs."
                )
            previous_signer = signer_splits.get(row.signer_id)
            if previous_signer is not None and previous_signer.split != row.split:
                raise ManifestError(
                    f"ASL Citizen signer leakage: {row.signer_id!r} occurs in "
                    f"{previous_signer.split} ({previous_signer.origin}) and "
                    f"{row.split} ({row.origin}). Check the official split CSVs."
                )
            # A symbolic link must not turn an otherwise valid basename into an escape.
            _inside_root(root, str(video_directory / row.filename), "video", absolute=True)
            filenames[row.filename] = row
            signer_splits.setdefault(row.signer_id, row)
            rows.append(row)

    vocabulary = sorted({row.gloss for row in rows if row.split == SplitName.TRAIN.value})
    labels = tuple(
        LabelDefinition(class_index=index, gloss_id=gloss, gloss_name=gloss)
        for index, gloss in enumerate(vocabulary)
    )
    labels_by_gloss = {label.gloss_name: label for label in labels}
    records: list[ManifestRecord] = []
    for row in rows:
        label = labels_by_gloss.get(row.gloss)
        if label is None:
            raise ManifestError(
                f"ASL Citizen {row.split} gloss {row.gloss!r} at {row.origin} "
                "is absent from the training vocabulary. Check release consistency; "
                "evaluation labels must not silently expand the classifier vocabulary."
            )
        identifier = f"asl_citizen:{row.filename}"
        records.append(
            ManifestRecord(
                sample_id=identifier,
                instance_id=identifier,
                video_id=row.filename,
                signer_id=row.signer_id,
                gloss_id=label.gloss_id,
                gloss_name=label.gloss_name,
                class_index=label.class_index,
                view="single",
                video_path=(video_directory / row.filename).relative_to(root).as_posix(),
                split=row.split,
                asl_lex_code=row.asl_lex_code,
            )
        )
    return ManifestBuildResult(
        records=tuple(records),
        labels=labels,
        source_metadata=metadata_sources,
        source_video_directories={"single": video_directory.relative_to(root).as_posix()},
    )


def _read_split(path: Path, split: str) -> tuple[_SourceRow, ...]:
    if not path.is_file():
        raise ManifestError(f"ASL Citizen {split} split CSV not found: {path}")
    rows: list[_SourceRow] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = reader.fieldnames or []
            if len(header) != len(set(header)):
                raise ManifestError(f"Duplicate CSV column names in {path}: {header!r}")
            missing = sorted(set(_REQUIRED_COLUMNS) - set(header))
            if missing:
                raise ManifestError(
                    f"Invalid ASL Citizen CSV header in {path}; missing {missing}; "
                    f"found {header!r}. Use the official split CSVs."
                )
            user_column, filename_column, gloss_column, lex_column = _REQUIRED_COLUMNS
            for values in reader:
                origin = f"{path}:{reader.line_num}"
                if None in values or any(value is None for value in values.values()):
                    raise ManifestError(f"Malformed CSV row at {origin}: field count mismatch.")
                user = _required_value(values.get(user_column), user_column, origin)
                filename = _required_value(values.get(filename_column), filename_column, origin)
                gloss = _required_value(values.get(gloss_column), gloss_column, origin)
                _validate_filename(filename, origin)
                # ASL-LEX codes are annotations, not unique gloss/class identities.
                rows.append(
                    _SourceRow(user, filename, gloss, split, origin, values[lex_column] or None)
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"Cannot read ASL Citizen {split} CSV {path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"ASL Citizen {split} split CSV contains no records: {path}")
    return tuple(rows)


def _required_value(value: str | None, field: str, origin: str) -> str:
    if value is None or not value.strip():
        raise ManifestError(f"Missing ASL Citizen {field!r} at {origin}.")
    if any(ord(character) < 32 for character in value):
        raise ManifestError(f"Control character in ASL Citizen {field!r} at {origin}.")
    return value


def _validate_filename(filename: str, origin: str) -> None:
    if (
        filename in {".", ".."}
        or any(character in filename for character in ("/", "\\", ":"))
        or PureWindowsPath(filename).drive
        or Path(filename).is_absolute()
    ):
        raise ManifestError(
            f"Unsafe ASL Citizen video filename {filename!r} at {origin}; "
            "expected a plain filename without a directory or drive."
        )


def _inside_root(root: Path, value: str, label: str, *, absolute: bool = False) -> Path:
    path = Path(value)
    if not absolute and (path.is_absolute() or PureWindowsPath(value).drive):
        raise ManifestError(f"ASL Citizen {label} must be relative to the dataset root: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ManifestError(f"ASL Citizen {label} escapes the dataset root: {value}")
    return resolved
