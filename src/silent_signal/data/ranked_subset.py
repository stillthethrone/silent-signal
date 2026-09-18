"""Derive a smaller frozen ASL subset from an existing ranked subset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord
from silent_signal.data.manifest import ManifestError

_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class RankedSubsetResult:
    """A contiguous class subset that preserves every official split assignment."""

    records: tuple[ManifestRecord, ...]
    labels: tuple[LabelDefinition, ...]
    classes: tuple[dict[str, Any], ...]
    split_counts: dict[str, int]
    signer_counts: dict[str, int]

    def report(self) -> dict[str, Any]:
        """Return a JSON-safe description of the frozen selection."""

        total = len(self.records)
        return {
            "classes_selected": len(self.classes),
            "clips": total,
            "split_counts": self.split_counts,
            "split_percentages": {
                split: round(self.split_counts[split] / total * 100.0, 4)
                for split in _SPLITS
            },
            "signer_counts": self.signer_counts,
            "classes": list(self.classes),
        }


def select_ranked_asl_subset(
    records: Sequence[ManifestRecord],
    selection_report: Mapping[str, Any],
    *,
    class_count: int = 50,
) -> RankedSubsetResult:
    """Select ranked classes with coverage in every official ASL Citizen split.

    ``records`` must be the manifest produced by the ranked source subset, such
    as the frozen ASL-LEX top-200 manifest. Class labels are remapped to
    ``0..class_count-1`` while sample, signer, and official split identities are
    preserved exactly.
    """

    if class_count < 2:
        raise ManifestError("class_count must be at least 2.")
    if not records:
        raise ManifestError("Cannot select from an empty source manifest.")
    ranked_payload = selection_report.get("classes")
    if not isinstance(ranked_payload, list) or not ranked_payload:
        raise ManifestError("Selection report must contain a non-empty classes list.")

    _validate_source_records(records)
    counts = Counter((record.class_index, str(record.split)) for record in records)
    records_by_class: dict[int, list[ManifestRecord]] = {}
    for record in records:
        records_by_class.setdefault(record.class_index, []).append(record)

    ranked = sorted(
        (_normalize_ranked_item(item) for item in ranked_payload),
        key=lambda item: (item["source_rank"], item["source_top200_class_index"]),
    )
    seen_source_classes: set[int] = set()
    eligible: list[dict[str, Any]] = []
    for item in ranked:
        source_index = int(item["source_top200_class_index"])
        if source_index in seen_source_classes:
            raise ManifestError(f"Duplicate ranked source class index {source_index}.")
        seen_source_classes.add(source_index)
        if source_index not in records_by_class:
            raise ManifestError(
                f"Ranked source class {source_index} is absent from the source manifest."
            )
        if all(counts[(source_index, split)] > 0 for split in _SPLITS):
            eligible.append(item)

    if len(eligible) < class_count:
        raise ManifestError(
            f"Only {len(eligible)} ranked classes contain clips in train, validation, "
            f"and test; {class_count} are required."
        )

    selected = eligible[:class_count]
    remapping = {
        int(item["source_top200_class_index"]): class_index
        for class_index, item in enumerate(selected)
    }
    subset_records = tuple(
        replace(record, class_index=remapping[record.class_index])
        for record in records
        if record.class_index in remapping
    )

    classes: list[dict[str, Any]] = []
    labels: list[LabelDefinition] = []
    for class_index, item in enumerate(selected):
        source_index = int(item["source_top200_class_index"])
        source_records = records_by_class[source_index]
        exemplar = source_records[0]
        split_clip_counts = {
            split: counts[(source_index, split)] for split in _SPLITS
        }
        clip_count = sum(split_clip_counts.values())
        classes.append(
            {
                **item,
                "rank": class_index + 1,
                "class_index": class_index,
                "source_rank": int(item["source_rank"]),
                "source_top200_class_index": source_index,
                "source_subset_class_index": source_index,
                "gloss_id": exemplar.gloss_id,
                "gloss_name": exemplar.gloss_name,
                "clip_count": clip_count,
                "split_clip_counts": split_clip_counts,
                "split_percentages": {
                    split: round(split_clip_counts[split] / clip_count * 100.0, 4)
                    for split in _SPLITS
                },
            }
        )
        labels.append(
            LabelDefinition(
                class_index=class_index,
                gloss_id=exemplar.gloss_id,
                gloss_name=exemplar.gloss_name,
            )
        )

    _validate_subset(subset_records, class_count=class_count)
    split_counts = Counter(str(record.split) for record in subset_records)
    signer_counts = {
        split: len(
            {record.signer_id for record in subset_records if record.split == split}
        )
        for split in _SPLITS
    }
    return RankedSubsetResult(
        records=subset_records,
        labels=tuple(labels),
        classes=tuple(classes),
        split_counts={split: split_counts[split] for split in _SPLITS},
        signer_counts=signer_counts,
    )


def _normalize_ranked_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError("Every ranked class must be an object.")
    source_index = value.get("subset_class_index", value.get("class_index"))
    source_rank = value.get("rank")
    if source_index is None or source_rank is None:
        raise ManifestError("Ranked classes require rank and subset_class_index/class_index.")
    try:
        normalized = {
            key: item
            for key, item in value.items()
            if key not in {"rank", "class_index", "subset_class_index", "source_class_index"}
        }
        normalized["source_top200_class_index"] = int(source_index)
        normalized["source_rank"] = int(source_rank)
        if "source_class_index" in value:
            normalized["source_asl_citizen_class_index"] = int(
                value["source_class_index"]
            )
    except (TypeError, ValueError) as exc:
        raise ManifestError("Ranked class indices and ranks must be integers.") from exc
    return normalized


def _validate_source_records(records: Sequence[ManifestRecord]) -> None:
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError("Source manifest contains duplicate sample_id values.")
    if {record.split for record in records} != set(_SPLITS):
        raise ManifestError("Source manifest must contain official train/validation/test splits.")
    _validate_signer_isolation(records)


def _validate_subset(records: Sequence[ManifestRecord], *, class_count: int) -> None:
    classes = {record.class_index for record in records}
    if classes != set(range(class_count)):
        raise ManifestError("Selected class indices are not contiguous from zero.")
    for class_index in range(class_count):
        present = {
            str(record.split) for record in records if record.class_index == class_index
        }
        if present != set(_SPLITS):
            raise ManifestError(
                f"Selected class {class_index} does not cover all official splits."
            )
    _validate_signer_isolation(records)


def _validate_signer_isolation(records: Sequence[ManifestRecord]) -> None:
    signers = {
        split: {record.signer_id for record in records if record.split == split}
        for split in _SPLITS
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = signers[first] & signers[second]
        if overlap:
            raise ManifestError(
                f"Signer leakage between {first}/{second}: {sorted(overlap)}."
            )
