from __future__ import annotations

from silent_signal.contracts import ManifestRecord
from silent_signal.data.ranked_subset import select_ranked_asl_subset


def test_ranked_subset_preserves_official_split_and_skips_incomplete_class() -> None:
    records = [
        _record(class_index, split)
        for class_index in range(4)
        for split in ("train", "validation", "test")
        if not (class_index == 0 and split == "test")
    ]
    report = {
        "classes": [
            {
                "rank": class_index + 1,
                "subset_class_index": class_index,
                "source_class_index": class_index + 100,
                "gloss_name": f"WORD_{class_index}",
                "sign_frequency_mean": 7.0 - class_index,
            }
            for class_index in range(4)
        ]
    }

    result = select_ranked_asl_subset(records, report, class_count=3)

    assert {record.class_index for record in result.records} == {0, 1, 2}
    assert {record.split for record in result.records} == {
        "train",
        "validation",
        "test",
    }
    assert result.split_counts == {"train": 3, "validation": 3, "test": 3}
    assert [item["source_top200_class_index"] for item in result.classes] == [1, 2, 3]
    assert [item["class_index"] for item in result.classes] == [0, 1, 2]
    assert all(item["split_clip_counts"]["test"] == 1 for item in result.classes)


def _record(class_index: int, split: str) -> ManifestRecord:
    return ManifestRecord(
        sample_id=f"{split}-{class_index}",
        instance_id=f"{split}-{class_index}",
        video_id=f"{split}-{class_index}.mp4",
        signer_id=f"signer-{split}",
        gloss_id=f"g-{class_index}",
        gloss_name=f"WORD_{class_index}",
        class_index=class_index,
        view="single",
        video_path=f"videos/{split}-{class_index}.mp4",
        split=split,
    )
