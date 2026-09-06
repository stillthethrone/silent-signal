from __future__ import annotations

import json
from pathlib import Path

from silent_signal.configuration import DatasetConfig
from silent_signal.data.manifest import build_manifest, read_manifest, write_manifest


def test_builds_canonical_three_view_manifest(dataset_config: DatasetConfig) -> None:
    result = build_manifest(dataset_config)

    assert len(result.records) == 96
    assert len(result.labels) == 2
    assert {record.view for record in result.records} == {"front", "left", "right"}
    assert len({record.instance_id for record in result.records}) == 32
    assert all(len(record.video_id) == 6 for record in result.records)
    assert all(len(record.signer_id) == 3 for record in result.records)
    grouped: dict[str, set[str]] = {}
    for record in result.records:
        grouped.setdefault(record.instance_id, set()).add(record.view)
    assert all(views == {"front", "left", "right"} for views in grouped.values())


def test_csv_and_parquet_round_trip(dataset_config: DatasetConfig, tmp_path: Path) -> None:
    records = build_manifest(dataset_config).records
    for suffix in (".csv", ".parquet"):
        path = tmp_path / f"manifest{suffix}"
        write_manifest(records, path)
        assert read_manifest(path) == records


def test_accepts_pandas_column_oriented_json(dataset_config: DatasetConfig) -> None:
    path = dataset_config.root / "right_view.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = {key: {str(index): row[key] for index, row in enumerate(rows)} for key in rows[0]}
    path.write_text(json.dumps(columns, ensure_ascii=False), encoding="utf-8")

    result = build_manifest(dataset_config)

    assert len(result.records) == 96


def test_discovers_legacy_camera_aliases(dataset_config: DatasetConfig) -> None:
    for index, view in enumerate(("front", "left", "right"), start=1):
        (dataset_config.root / f"{view}_view").rename(dataset_config.root / f"cam_{index}")
        (dataset_config.root / f"{view}_view.json").rename(
            dataset_config.root / f"cam_{index}.json"
        )

    result = build_manifest(dataset_config)

    assert result.source_video_directories == {
        "front": "cam_1",
        "left": "cam_2",
        "right": "cam_3",
    }
