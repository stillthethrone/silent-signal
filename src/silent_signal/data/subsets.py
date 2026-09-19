"""Reproducible ASL Citizen subsets ranked by ASL-LEX sign frequency."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord
from silent_signal.data.manifest import ManifestError, manifest_summary

_CODE_COLUMN = "Code"
_ENTRY_COLUMN = "EntryID"
_LEMMA_COLUMN = "LemmaID"
_FREQUENCY_COLUMN = "SignFrequency(M)"
_MIN_FREQUENCY = 1.0
_MAX_FREQUENCY = 7.0


@dataclass(frozen=True, slots=True)
class ASLLexFrequency:
    """One ASL-LEX 2.0 subjective-frequency observation."""

    code: str
    frequency: float
    entry_id: str | None = None
    lemma_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedASLClass:
    """One source ASL Citizen class selected into a contiguous subset."""

    rank: int
    subset_class_index: int
    source_class_index: int
    gloss_id: str
    gloss_name: str
    asl_lex_code: str
    asl_lex_entry_id: str | None
    asl_lex_lemma_id: str | None
    sign_frequency_mean: float
    clip_count: int
    split_clip_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "rank": self.rank,
            "subset_class_index": self.subset_class_index,
            "source_class_index": self.source_class_index,
            "gloss_id": self.gloss_id,
            "gloss_name": self.gloss_name,
            "asl_lex_code": self.asl_lex_code,
            "asl_lex_entry_id": self.asl_lex_entry_id,
            "asl_lex_lemma_id": self.asl_lex_lemma_id,
            "sign_frequency_mean": self.sign_frequency_mean,
            "clip_count": self.clip_count,
            "split_clip_counts": self.split_clip_counts,
        }


@dataclass(frozen=True, slots=True)
class ASLSubsetResult:
    """Selected records, labels, ranking, and diagnostic metadata."""

    records: tuple[ManifestRecord, ...]
    labels: tuple[LabelDefinition, ...]
    classes: tuple[SelectedASLClass, ...]
    source_class_count: int
    excluded_class_counts: dict[str, int]

    def report(self) -> dict[str, Any]:
        """Build the data-dependent part of the selection report."""

        classes_by_code: dict[str, list[str]] = {}
        for item in self.classes:
            classes_by_code.setdefault(item.asl_lex_code, []).append(item.gloss_name)
        shared_codes = {
            code: names for code, names in sorted(classes_by_code.items()) if len(names) > 1
        }
        return {
            "classes_requested": len(self.classes),
            "classes_selected": len(self.classes),
            "unique_asl_lex_codes_selected": len(classes_by_code),
            "shared_asl_lex_codes": shared_codes,
            "source_classes": self.source_class_count,
            "excluded_class_counts": self.excluded_class_counts,
            "subset_manifest": manifest_summary(self.records),
            "classes": [item.to_dict() for item in self.classes],
        }


@dataclass(slots=True)
class _ClassAccumulator:
    source_class_index: int
    gloss_id: str
    gloss_name: str
    asl_lex_codes: set[str]
    clip_count: int
    split_clip_counts: Counter[str]


def read_asl_lex_frequencies(path: str | Path) -> dict[str, ASLLexFrequency]:
    """Read ASL-LEX 2.0 ``signdata.csv`` using its published Latin-1 encoding."""

    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"ASL-LEX frequency CSV not found: {source}")

    frequencies: dict[str, ASLLexFrequency] = {}
    with source.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {_CODE_COLUMN, _FREQUENCY_COLUMN} - fields
        if missing:
            raise ManifestError(
                "ASL-LEX CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            code = (row.get(_CODE_COLUMN) or "").strip()
            if not code:
                continue
            raw_frequency = (row.get(_FREQUENCY_COLUMN) or "").strip()
            if not raw_frequency:
                raise ManifestError(
                    f"ASL-LEX row {row_number} has no {_FREQUENCY_COLUMN} for code {code!r}."
                )
            try:
                frequency = float(raw_frequency)
            except ValueError as exc:
                raise ManifestError(
                    f"ASL-LEX row {row_number} has invalid frequency {raw_frequency!r}."
                ) from exc
            if not _MIN_FREQUENCY <= frequency <= _MAX_FREQUENCY:
                raise ManifestError(
                    f"ASL-LEX row {row_number} frequency {frequency} is outside the 1-7 scale."
                )
            if code in frequencies:
                raise ManifestError(f"Duplicate ASL-LEX Code {code!r} at row {row_number}.")
            frequencies[code] = ASLLexFrequency(
                code=code,
                frequency=frequency,
                entry_id=_optional_text(row.get(_ENTRY_COLUMN)),
                lemma_id=_optional_text(row.get(_LEMMA_COLUMN)),
            )

    if not frequencies:
        raise ManifestError(f"ASL-LEX CSV contains no usable frequency rows: {source}")
    return frequencies


def select_asl_citizen_top_classes(
    records: Sequence[ManifestRecord],
    frequencies: Mapping[str, ASLLexFrequency],
    *,
    class_count: int = 200,
) -> ASLSubsetResult:
    """Select the most frequent ASL Citizen classes and preserve official splits.

    ASL-LEX codes are annotations, not class identifiers. Two ASL Citizen
    glosses that share a code therefore remain two independent classes.
    """

    if class_count < 1:
        raise ManifestError("class_count must be at least 1.")
    if not records:
        raise ManifestError("Cannot select a subset from an empty manifest.")

    accumulators = _accumulate_classes(records)
    eligible: list[tuple[_ClassAccumulator, ASLLexFrequency]] = []
    excluded = Counter[str]()
    for item in accumulators.values():
        if not item.asl_lex_codes:
            excluded["missing_asl_lex_code"] += 1
            continue
        if len(item.asl_lex_codes) > 1:
            codes = ", ".join(sorted(item.asl_lex_codes))
            raise ManifestError(
                f"ASL Citizen class {item.source_class_index} ({item.gloss_name!r}) "
                f"has conflicting ASL-LEX codes: {codes}."
            )
        code = next(iter(item.asl_lex_codes))
        frequency = frequencies.get(code)
        if frequency is None:
            excluded["code_not_in_asl_lex"] += 1
            continue
        if item.split_clip_counts.get("train", 0) == 0:
            excluded["no_train_clips"] += 1
            continue
        eligible.append((item, frequency))

    eligible.sort(
        key=lambda pair: (
            -pair[1].frequency,
            pair[0].gloss_name.casefold(),
            pair[0].gloss_name,
            pair[0].source_class_index,
        )
    )
    if len(eligible) < class_count:
        raise ManifestError(
            f"Requested {class_count} classes, but only {len(eligible)} ASL Citizen "
            "classes have a usable ASL-LEX frequency and at least one train clip."
        )

    selected = eligible[:class_count]
    remapping = {
        item.source_class_index: subset_index
        for subset_index, (item, _frequency) in enumerate(selected)
    }
    subset_records = tuple(
        replace(record, class_index=remapping[record.class_index])
        for record in records
        if record.class_index in remapping
    )

    labels: list[LabelDefinition] = []
    classes: list[SelectedASLClass] = []
    for subset_index, (item, frequency) in enumerate(selected):
        labels.append(
            LabelDefinition(
                class_index=subset_index,
                gloss_id=item.gloss_id,
                gloss_name=item.gloss_name,
            )
        )
        classes.append(
            SelectedASLClass(
                rank=subset_index + 1,
                subset_class_index=subset_index,
                source_class_index=item.source_class_index,
                gloss_id=item.gloss_id,
                gloss_name=item.gloss_name,
                asl_lex_code=frequency.code,
                asl_lex_entry_id=frequency.entry_id,
                asl_lex_lemma_id=frequency.lemma_id,
                sign_frequency_mean=frequency.frequency,
                clip_count=item.clip_count,
                split_clip_counts=dict(sorted(item.split_clip_counts.items())),
            )
        )

    return ASLSubsetResult(
        records=subset_records,
        labels=tuple(labels),
        classes=tuple(classes),
        source_class_count=len(accumulators),
        excluded_class_counts=dict(sorted(excluded.items())),
    )


def _accumulate_classes(records: Sequence[ManifestRecord]) -> dict[int, _ClassAccumulator]:
    accumulators: dict[int, _ClassAccumulator] = {}
    for record in records:
        item = accumulators.get(record.class_index)
        if item is None:
            item = _ClassAccumulator(
                source_class_index=record.class_index,
                gloss_id=record.gloss_id,
                gloss_name=record.gloss_name,
                asl_lex_codes=set(),
                clip_count=0,
                split_clip_counts=Counter(),
            )
            accumulators[record.class_index] = item
        elif (item.gloss_id, item.gloss_name) != (record.gloss_id, record.gloss_name):
            raise ManifestError(
                f"Class index {record.class_index} maps to multiple ASL Citizen glosses."
            )
        if record.asl_lex_code and record.asl_lex_code.strip():
            item.asl_lex_codes.add(record.asl_lex_code.strip())
        item.clip_count += 1
        item.split_clip_counts[record.split or "unspecified"] += 1
    return accumulators


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
