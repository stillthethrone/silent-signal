from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from silent_signal.cli import select_subset
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import read_manifest, write_manifest


def _record(gloss: str, class_index: int, code: str, split: str) -> ManifestRecord:
    suffix = f"{gloss.lower()}-{split}"
    return ManifestRecord(
        sample_id=f"asl_citizen:{suffix}",
        instance_id=f"asl_citizen:{suffix}",
        video_id=f"{suffix}.mp4",
        signer_id=f"signer-{split}",
        gloss_id=gloss,
        gloss_name=gloss,
        class_index=class_index,
        view="single",
        video_path=f"videos/{suffix}.mp4",
        split=split,
        asl_lex_code=code,
    )


def test_cli_writes_reproducible_subset_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "source.csv"
    write_manifest(
        (
            _record("FIRST", 10, "first-code", "train"),
            _record("FIRST", 10, "first-code", "test"),
            _record("SECOND", 20, "second-code", "train"),
        ),
        manifest,
    )
    asl_lex = tmp_path / "signdata.csv"
    with asl_lex.open("w", encoding="latin-1", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EntryID", "LemmaID", "Code", "SignFrequency(M)"])
        writer.writerow(["first", "first", "first-code", "6.5"])
        writer.writerow(["second", "second", "second-code", "4.0"])

    output_root = tmp_path / "subsets"
    real_write_manifest = select_subset.write_manifest

    def write_manifest_without_optional_pyarrow(
        records: Sequence[ManifestRecord], path: str | Path
    ) -> None:
        destination = Path(path)
        if destination.suffix == ".parquet":
            destination.write_bytes(b"synthetic-parquet-for-cli-test")
        else:
            real_write_manifest(records, destination)

    monkeypatch.setattr(select_subset, "write_manifest", write_manifest_without_optional_pyarrow)
    assert (
        select_subset.main(
            [
                "--manifest",
                str(manifest),
                "--asl-lex-csv",
                str(asl_lex),
                "--classes",
                "1",
                "--output-root",
                str(output_root),
                "--project-commit",
                "test-commit",
            ]
        )
        == 0
    )

    destination = output_root / "asl_citizen_asllex_top1"
    assert {item.gloss_name for item in read_manifest(destination / "manifest.csv")} == {"FIRST"}
    assert (destination / "manifest.parquet").read_bytes() == b"synthetic-parquet-for-cli-test"
    labels = json.loads((destination / "labels.json").read_text(encoding="utf-8"))
    report = json.loads((destination / "selection_report.json").read_text(encoding="utf-8"))
    assert labels["num_classes"] == 1
    assert report["classes_selected"] == 1
    assert report["project_commit"] == "test-commit"
    assert report["classes"][0]["sign_frequency_mean"] == 6.5
    assert len(report["sources"]["manifest"]["sha256"]) == 64
