from __future__ import annotations

import json
from pathlib import Path

from silent_signal.cli.select_ranked_subset import main
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import read_manifest, write_manifest


def test_ranked_subset_cli_writes_split_manifests_and_frozen_mapping(
    tmp_path: Path,
) -> None:
    records = tuple(
        _record(class_index, split)
        for class_index in range(3)
        for split in ("train", "validation", "test")
    )
    manifest = tmp_path / "source.csv"
    selection = tmp_path / "selection.json"
    output = tmp_path / "top2"
    write_manifest(records, manifest)
    selection.write_text(
        json.dumps(
            {
                "classes": [
                    {
                        "rank": index + 1,
                        "subset_class_index": index,
                        "source_class_index": index + 100,
                        "gloss_name": f"WORD_{index}",
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--selection-report",
            str(selection),
            "--output-root",
            str(output),
            "--classes",
            "2",
        ]
    )

    payload = json.loads((output / "selected_2_words.json").read_text(encoding="utf-8"))
    selected = read_manifest(output / "manifest.csv")
    assert exit_code == 0
    assert len(selected) == 6
    assert {record.class_index for record in selected} == {0, 1}
    assert payload["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert payload["classes"][0]["class_index"] == 0
    assert payload["classes"][0]["source_subset_class_index"] == 0
    assert "subset_class_index" not in payload["classes"][0]
    for split in ("train", "validation", "test"):
        split_records = read_manifest(output / "manifests" / f"{split}.csv")
        assert len(split_records) == 2
        assert {record.split for record in split_records} == {split}


def _record(class_index: int, split: str) -> ManifestRecord:
    sample_id = f"{split}-{class_index}"
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=f"{sample_id}.mp4",
        signer_id=f"signer-{split}",
        gloss_id=f"g-{class_index}",
        gloss_name=f"WORD_{class_index}",
        class_index=class_index,
        view="single",
        video_path=f"videos/{sample_id}.mp4",
        split=split,
    )
