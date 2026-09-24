from __future__ import annotations

from dataclasses import replace

import pytest

from silent_signal.contracts import ManifestRecord
from silent_signal.data.class_subset import select_classes
from silent_signal.data.manifest import ManifestError

# class_index -> training recordings; validation/test get one recording each, except class 3.
_TRAIN_RECORDINGS = {0: 2, 1: 4, 2: 4, 3: 5, 4: 1}


def _records() -> tuple[ManifestRecord, ...]:
    records = []
    for class_index, train in _TRAIN_RECORDINGS.items():
        splits = ["train"] * train + ([] if class_index == 3 else ["validation", "test"])
        for number, split in enumerate(splits):
            instance = f"{class_index}-{number}"
            for view in ("front", "left", "right"):
                records.append(
                    ManifestRecord(
                        sample_id=f"{instance}_{view}",
                        instance_id=instance,
                        video_id=instance,
                        signer_id={"train": "001", "validation": "002", "test": "003"}[split],
                        gloss_id=str(10 + class_index),
                        gloss_name=f"từ {class_index}",
                        class_index=class_index,
                        view=view,
                        video_path=f"{view}_view/{instance}.mp4",
                        split=split,
                    )
                )
    return tuple(records)


def test_ranks_by_training_recordings_and_keeps_split() -> None:
    subset = select_classes(_records(), class_count=3)

    # Class 3 has most training data but no validation/test recording, so it is ineligible.
    assert [item["source_class_index"] for item in subset.classes] == [1, 2, 0]
    assert [label.gloss_name for label in subset.labels] == ["từ 1", "từ 2", "từ 0"]
    assert subset.classes[0]["instances"] == {"train": 4, "validation": 1, "test": 1}
    assert subset.classes[0]["clips"] == {"train": 12, "validation": 3, "test": 3}
    source = {record.sample_id: record for record in _records()}
    for record in subset.records:
        original = source[record.sample_id]
        assert record.split == original.split
        assert record.class_index == [1, 2, 0].index(original.class_index)


def test_explicit_gloss_ids_keep_requested_order() -> None:
    subset = select_classes(_records(), gloss_ids=["14", "10"])

    assert [label.gloss_id for label in subset.labels] == ["14", "10"]
    assert {record.class_index for record in subset.records} == {0, 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"class_count": 5}, "Only 4 classes"),
        ({"gloss_ids": ["99"]}, "Unknown gloss"),
        ({"gloss_ids": ["13"]}, "absent from at least one split"),
    ],
)
def test_rejects_impossible_selections(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        select_classes(_records(), **kwargs)  # type: ignore[arg-type]


def test_requires_split_and_valid_records() -> None:
    records = list(_records())
    with pytest.raises(ManifestError, match="split"):
        select_classes([replace(records[0], split=None), *records[1:]])
    with pytest.raises(ManifestError, match="invalid"):
        select_classes([replace(records[0], is_valid=False), *records[1:]])
