from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from silent_signal.configuration import (
    DatasetConfig,
    ExpectedConfig,
    OutputConfig,
    SplitConfig,
    ViewConfig,
)


@pytest.fixture
def vsl400_root(tmp_path: Path) -> Path:
    root = tmp_path / "VSL400"
    views = ("front", "left", "right")
    metadata: dict[str, list[dict[str, object]]] = {view: [] for view in views}
    video_id = 0
    for signer_number in range(1, 9):
        for repetition in range(2):
            for gloss_id, gloss in (("0", "xin chào"), ("1", "cảm ơn")):
                identifier = f"{video_id:06d}"
                row = {
                    "video_id": identifier,
                    "signer_id": f"{signer_number:03d}",
                    "fps": 25,
                    "resolution": 1080,
                    "num_of_frames": 25 + repetition,
                    "length": 1.0 + repetition / 25,
                    "gloss": gloss,
                    "gloss_id": gloss_id,
                }
                for view in views:
                    metadata[view].append(dict(row))
                video_id += 1

    for view in views:
        directory = root / f"{view}_view"
        directory.mkdir(parents=True)
        for row in metadata[view]:
            (directory / f"{row['video_id']}.mp4").write_bytes(b"synthetic-video")
        (root / f"{view}_view.json").write_text(
            json.dumps(metadata[view], ensure_ascii=False),
            encoding="utf-8",
        )
    (root / "gloss.csv").write_text("0,xin chào\n1,cảm ơn\n", encoding="utf-8")
    return root


@pytest.fixture
def dataset_config(vsl400_root: Path, tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        name="vsl400",
        root=vsl400_root,
        video_extension=".mp4",
        gloss_file="gloss.csv",
        views={
            "front": ViewConfig("front_view", "front_view.json", ("cam_1",), ("cam_1.json",)),
            "left": ViewConfig("left_view", "left_view.json", ("cam_2",), ("cam_2.json",)),
            "right": ViewConfig("right_view", "right_view.json", ("cam_3",), ("cam_3.json",)),
        },
        expected=ExpectedConfig(
            clips=96,
            glosses=2,
            signers=8,
            views_per_instance=3,
            fps=25.0,
            width=1080,
            height=1080,
            enforce_counts=True,
        ),
        split=SplitConfig(
            ratios={"train": 0.75, "validation": 0.125, "test": 0.125},
            seed=42,
            search_trials=200,
            official_file=None,
        ),
        outputs=OutputConfig(
            manifest_csv=tmp_path / "outputs" / "vsl400.csv",
            manifest_parquet=tmp_path / "outputs" / "vsl400.parquet",
            labels=tmp_path / "outputs" / "labels.json",
            split=tmp_path / "outputs" / "split.json",
            report=tmp_path / "outputs" / "report.json",
            invalid_records=tmp_path / "outputs" / "invalid.csv",
        ),
    )


@pytest.fixture
def config_file(dataset_config: DatasetConfig, tmp_path: Path) -> Path:
    config = {
        "schema_version": 1,
        "dataset": {
            "name": "vsl400",
            "root": str(dataset_config.root),
            "video_extension": ".mp4",
            "gloss_file": "gloss.csv",
            "views": {
                name: {
                    "directory": value.directory,
                    "metadata": value.metadata,
                    "directory_aliases": list(value.directory_aliases),
                    "metadata_aliases": list(value.metadata_aliases),
                }
                for name, value in dataset_config.views.items()
            },
        },
        "expected": {
            "clips": 96,
            "glosses": 2,
            "signers": 8,
            "views_per_instance": 3,
            "fps": 25.0,
            "width": 1080,
            "height": 1080,
            "enforce_counts": True,
        },
        "split": {
            "strategy": "signer_disjoint",
            "ratios": {"train": 0.75, "validation": 0.125, "test": 0.125},
            "seed": 42,
            "search_trials": 200,
        },
        "outputs": {
            "manifest_csv": str(dataset_config.outputs.manifest_csv),
            "manifest_parquet": str(dataset_config.outputs.manifest_parquet),
            "labels": str(dataset_config.outputs.labels),
            "split": str(dataset_config.outputs.split),
            "report": str(dataset_config.outputs.report),
            "invalid_records": str(dataset_config.outputs.invalid_records),
        },
    }
    path = tmp_path / "vsl400.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path
