"""Select a class subset from a split manifest without re-splitting it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord, SplitName
from silent_signal.data.manifest import ManifestError

_SPLITS = tuple(item.value for item in SplitName)
SELECTION_RULE = (
    "classes present in every split, ranked by training recordings (instances) only; "
    "ties keep the source class order"
)


@dataclass(frozen=True, slots=True)
class ClassSubset:
    """Records remapped to contiguous class indices, plus the ranked class table."""

    records: tuple[ManifestRecord, ...]
    labels: tuple[LabelDefinition, ...]
    classes: tuple[dict[str, Any], ...]
    rule: str


def select_classes(
    records: Sequence[ManifestRecord],
    *,
    class_count: int = 50,
    gloss_ids: Sequence[str] = (),
) -> ClassSubset:
    """Keep whole classes and their existing split membership.

    By default classes are ranked by the number of training instances (recordings), so
    validation and test only need to contain the class. ``gloss_ids`` instead keeps exactly
    those classes, in the given order. The new ``class_index`` is the rank.
    """

    if not records:
        raise ManifestError("Cannot select classes from an empty manifest.")
    if any(record.split not in _SPLITS for record in records):
        raise ManifestError("Every record needs a train/validation/test split; run create-splits.")
    if any(not record.is_valid for record in records):
        raise ManifestError("The manifest contains invalid clips; fix validation errors first.")

    by_class: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_index].append(record)
    instances = {
        class_index: {
            split: len({item.instance_id for item in items if item.split == split})
            for split in _SPLITS
        }
        for class_index, items in by_class.items()
    }
    eligible = [
        class_index
        for class_index, counts in instances.items()
        if all(counts[split] > 0 for split in _SPLITS)
    ]

    if gloss_ids:
        by_gloss = {items[0].gloss_id: class_index for class_index, items in by_class.items()}
        unknown = [gloss for gloss in gloss_ids if gloss not in by_gloss]
        if unknown:
            raise ManifestError(f"Unknown gloss IDs: {unknown}.")
        if len(set(gloss_ids)) != len(gloss_ids):
            raise ManifestError("Requested gloss IDs contain duplicates.")
        chosen = [by_gloss[gloss] for gloss in gloss_ids]
        missing_split = [by_class[index][0].gloss_id for index in chosen if index not in eligible]
        if missing_split:
            raise ManifestError(f"Gloss IDs absent from at least one split: {missing_split}.")
        rule = "classes chosen explicitly by gloss ID, in the given order"
    else:
        if class_count < 2:
            raise ManifestError("class_count must be at least 2.")
        ranked = sorted(eligible, key=lambda index: (-instances[index]["train"], index))
        if len(ranked) < class_count:
            raise ManifestError(
                f"Only {len(ranked)} classes occur in every split; {class_count} requested."
            )
        chosen = ranked[:class_count]
        rule = SELECTION_RULE

    remap = {source: rank for rank, source in enumerate(chosen)}
    subset = tuple(
        replace(record, class_index=remap[record.class_index])
        for record in sorted(
            records, key=lambda item: (remap.get(item.class_index, -1), item.sample_id)
        )
        if record.class_index in remap
    )
    labels = tuple(
        LabelDefinition(
            class_index=rank,
            gloss_id=by_class[source][0].gloss_id,
            gloss_name=by_class[source][0].gloss_name,
        )
        for rank, source in enumerate(chosen)
    )
    classes = tuple(
        {
            "rank": rank + 1,
            "class_index": rank,
            "source_class_index": source,
            "gloss_id": by_class[source][0].gloss_id,
            "gloss_name": by_class[source][0].gloss_name,
            "instances": instances[source],
            "clips": {
                split: sum(item.split == split for item in by_class[source]) for split in _SPLITS
            },
        }
        for rank, source in enumerate(chosen)
    )
    return ClassSubset(records=subset, labels=labels, classes=classes, rule=rule)
