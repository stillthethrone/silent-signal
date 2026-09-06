from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from silent_signal.configuration import DatasetConfig, ExpectedConfig, ViewConfig
from silent_signal.data.adapters.asl_citizen import build_asl_citizen_manifest
from silent_signal.data.manifest import ManifestError, build_manifest, read_manifest, write_manifest

_COLUMNS = ("Participant ID", "Video file", "Gloss", "ASL-LEX Code")
_SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}


def _write_csv(path: Path, rows: list[list[str]], *, columns: tuple[str, ...] = _COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))[1:]


@pytest.fixture
def asl_config(dataset_config: DatasetConfig, tmp_path: Path) -> DatasetConfig:
    root = tmp_path / "ASL_Citizen"
    (root / "videos").mkdir(parents=True)
    (root / "splits").mkdir()
    splits = {
        "train": [
            ["7", "clip A.MP4", "HELLO", "shared-code"],
            ["7", "2_uuid+clip.mp4", "BOOK", "shared-code"],
            ["7", "clip-C.mp4", "hello", ""],
        ],
        "validation": [
            ["007", "val-1.mp4", "BOOK", "shared-code"],
            ["007", "val-2.mp4", "HELLO", "shared-code"],
        ],
        "test": [["u-test", "test-1.mp4", "hello", ""]],
    }
    for split, rows in splits.items():
        _write_csv(root / "splits" / _SPLIT_FILES[split], rows)
        for _, filename, _, _ in rows:
            (root / "videos" / filename).write_bytes(b"synthetic-video")
    return replace(
        dataset_config,
        name="asl_citizen",
        adapter="asl_citizen",
        root=root,
        gloss_file=None,
        views={"single": ViewConfig(directory="videos")},
        metadata_splits={split: f"splits/{name}" for split, name in _SPLIT_FILES.items()},
        expected=ExpectedConfig(clips=6, glosses=3, signers=3, views_per_instance=1),
        split=replace(dataset_config.split, strategy="official", ratios={}),
    )


def test_imports_exact_ids_vocabulary_and_official_membership(asl_config: DatasetConfig) -> None:
    result = build_asl_citizen_manifest(asl_config)

    assert [(label.gloss_name, label.class_index) for label in result.labels] == [
        ("BOOK", 0),
        ("HELLO", 1),
        ("hello", 2),
    ]
    assert all(label.gloss_id == label.gloss_name for label in result.labels)
    by_filename = {record.video_id: record for record in result.records}
    assert by_filename["clip A.MP4"].video_path == "videos/clip A.MP4"
    assert by_filename["clip A.MP4"].signer_id == "7"
    assert by_filename["val-1.mp4"].signer_id == "007"
    assert by_filename["clip A.MP4"].split == "train"
    assert by_filename["val-1.mp4"].split == "validation"
    assert by_filename["test-1.mp4"].split == "test"
    assert by_filename["clip A.MP4"].sample_id == "asl_citizen:clip A.MP4"
    assert all(record.instance_id == record.sample_id for record in result.records)
    assert {record.view for record in result.records} == {"single"}
    assert result.source_metadata == asl_config.metadata_splits
    assert result.source_video_directories == {"single": "videos"}


def test_preserves_lex_annotation_without_merging_distinct_glosses(
    asl_config: DatasetConfig,
) -> None:
    result = build_asl_citizen_manifest(asl_config)
    by_filename = {record.video_id: record for record in result.records}

    hello = by_filename["clip A.MP4"]
    book = by_filename["2_uuid+clip.mp4"]
    assert hello.asl_lex_code == book.asl_lex_code == "shared-code"
    assert hello.class_index != book.class_index
    assert by_filename["clip-C.mp4"].asl_lex_code is None


def test_shared_dispatch_and_manifest_round_trip(asl_config: DatasetConfig, tmp_path: Path) -> None:
    result = build_manifest(asl_config)

    assert result == build_asl_citizen_manifest(asl_config)
    for suffix in (".csv", ".parquet"):
        path = tmp_path / f"asl-manifest{suffix}"
        write_manifest(result.records, path)
        assert read_manifest(path) == result.records


def test_rejects_eval_gloss_absent_from_training(asl_config: DatasetConfig) -> None:
    path = asl_config.root / "splits/val.csv"
    rows = _read_rows(path)
    rows[0][2] = "UNSEEN"
    _write_csv(path, rows)

    with pytest.raises(ManifestError, match="absent from the training vocabulary"):
        build_asl_citizen_manifest(asl_config)


@pytest.mark.parametrize("field", [0, 1, 2])
def test_rejects_empty_required_cell(asl_config: DatasetConfig, field: int) -> None:
    row = ["7", "clip.mp4", "BOOK", "book-code"]
    row[field] = "  "
    _write_csv(asl_config.root / "splits/train.csv", [row])

    with pytest.raises(ManifestError, match="Missing ASL Citizen"):
        build_asl_citizen_manifest(asl_config)


@pytest.mark.parametrize(
    "row", [["7", "clip.mp4", "BOOK"], ["7", "clip.mp4", "BOOK", "book-code", "extra"]]
)
def test_rejects_malformed_row(asl_config: DatasetConfig, row: list[str]) -> None:
    _write_csv(asl_config.root / "splits/train.csv", [row])

    with pytest.raises(ManifestError, match="field count mismatch"):
        build_asl_citizen_manifest(asl_config)


def test_rejects_unrecognized_header(asl_config: DatasetConfig) -> None:
    _write_csv(
        asl_config.root / "splits/train.csv",
        [["7", "clip.mp4", "BOOK"]],
        columns=("signer_id", "filename", "gloss"),
    )

    with pytest.raises(ManifestError, match="Invalid ASL Citizen CSV header"):
        build_asl_citizen_manifest(asl_config)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_rejects_missing_split_csv(asl_config: DatasetConfig, split: str) -> None:
    (asl_config.root / asl_config.metadata_splits[split]).unlink()

    with pytest.raises(ManifestError, match=f"{split} split CSV not found"):
        build_asl_citizen_manifest(asl_config)


def test_rejects_missing_split_configuration(asl_config: DatasetConfig) -> None:
    sources = dict(asl_config.metadata_splits)
    del sources["test"]

    with pytest.raises(ManifestError, match="metadata_splits must specify"):
        build_asl_citizen_manifest(replace(asl_config, metadata_splits=sources))


def test_rejects_empty_split_csv(asl_config: DatasetConfig) -> None:
    _write_csv(asl_config.root / "splits/test.csv", [])

    with pytest.raises(ManifestError, match="contains no records"):
        build_asl_citizen_manifest(asl_config)


@pytest.mark.parametrize("destination", ["train", "validation"])
def test_rejects_duplicate_filename(asl_config: DatasetConfig, destination: str) -> None:
    path = asl_config.root / asl_config.metadata_splits[destination]
    rows = _read_rows(path)
    rows.append([rows[0][0], "clip A.MP4", "HELLO", "shared-code"])
    _write_csv(path, rows)

    with pytest.raises(ManifestError, match="Duplicate ASL Citizen filename"):
        build_asl_citizen_manifest(asl_config)


def test_rejects_cross_split_signer_leakage(asl_config: DatasetConfig) -> None:
    path = asl_config.root / "splits/val.csv"
    rows = _read_rows(path)
    rows[0][0] = "7"
    _write_csv(path, rows)

    with pytest.raises(ManifestError, match="signer leakage"):
        build_asl_citizen_manifest(asl_config)


@pytest.mark.parametrize(
    "filename",
    ["../clip.mp4", "folder/clip.mp4", "folder\\clip.mp4", "C:\\clip.mp4", "/clip.mp4", ".."],
)
def test_rejects_unsafe_video_paths(asl_config: DatasetConfig, filename: str) -> None:
    _write_csv(asl_config.root / "splits/train.csv", [["7", filename, "BOOK", "book-code"]])

    with pytest.raises(ManifestError, match="Unsafe ASL Citizen video filename"):
        build_asl_citizen_manifest(asl_config)


def test_rejects_metadata_path_escape(asl_config: DatasetConfig) -> None:
    sources = {**asl_config.metadata_splits, "train": "../train.csv"}

    with pytest.raises(ManifestError, match="metadata escapes the dataset root"):
        build_asl_citizen_manifest(replace(asl_config, metadata_splits=sources))


def test_rejects_video_directory_escape(asl_config: DatasetConfig) -> None:
    views = {"single": ViewConfig(directory="../videos")}

    with pytest.raises(ManifestError, match="video directory escapes the dataset root"):
        build_asl_citizen_manifest(replace(asl_config, views=views))


def test_leaves_missing_video_detection_to_validation(asl_config: DatasetConfig) -> None:
    (asl_config.root / "videos/test-1.mp4").unlink()

    assert len(build_asl_citizen_manifest(asl_config).records) == 6
