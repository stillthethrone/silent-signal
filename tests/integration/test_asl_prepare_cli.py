from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from silent_signal.cli.prepare import main
from silent_signal.data.manifest import read_manifest, write_manifest


@pytest.fixture
def asl_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "ASL_Citizen"
    (root / "videos").mkdir(parents=True)
    (root / "splits").mkdir()
    for split, signer in (("train", "7"), ("val", "007"), ("test", "heldout")):
        with (root / "splits" / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
            for gloss in ("BOOK", "HELLO"):
                filename = f"{split} {gloss}.mp4"
                writer.writerow([signer, filename, gloss, "shared-code"])
                (root / "videos" / filename).write_bytes(b"synthetic-video")
    payload = yaml.safe_load(Path("configs/dataset/asl_citizen.yaml").read_text(encoding="utf-8"))
    payload["dataset"]["root"] = str(root)
    payload["expected"].update(
        clips=6,
        glosses=2,
        signers=3,
        split_clips={"train": 2, "validation": 2, "test": 2},
        split_signers={"train": 1, "validation": 1, "test": 1},
    )
    outputs = tmp_path / "outputs"
    payload["outputs"] = {
        "manifest_csv": str(outputs / "asl_citizen.csv"),
        "manifest_parquet": str(outputs / "asl_citizen.parquet"),
        "labels": str(outputs / "labels.json"),
        "split": str(outputs / "split.json"),
        "report": str(outputs / "report.json"),
        "invalid_records": str(outputs / "invalid.csv"),
    }
    config_path = tmp_path / "asl.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path, root, outputs


def test_asl_cli_all_and_subcommands_preserve_official_assignments(
    asl_project: tuple[Path, Path, Path],
) -> None:
    config_path, _, outputs = asl_project
    assert main(["all", "--config", str(config_path)]) == 0
    manifest = read_manifest(outputs / "asl_citizen.parquet")
    assert {row.signer_id: row.split for row in manifest} == {
        "7": "train",
        "007": "validation",
        "heldout": "test",
    }
    assert all(row.asl_lex_code == "shared-code" for row in manifest)
    labels = json.loads((outputs / "labels.json").read_text(encoding="utf-8"))
    assert labels["dataset"] == "asl_citizen"
    assert labels["gloss_to_class_index"] == {"BOOK": 0, "HELLO": 1}
    split = json.loads((outputs / "split.json").read_text(encoding="utf-8"))
    assert split["strategy"] == "official"
    assert split["seed"] is None
    assert split["source_files"]["validation"]["path"] == "splits/val.csv"
    assert len(split["source_files"]["test"]["sha256"]) == 64
    assert json.loads((outputs / "report.json").read_text(encoding="utf-8"))["passed"]
    for command in ("validate", "create-splits", "summarize"):
        assert main([command, "--config", str(config_path)]) == 0
    assert read_manifest(outputs / "asl_citizen.parquet") == manifest


def test_asl_cli_rejects_repartitioning_a_saved_manifest(
    asl_project: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataclasses import replace

    config_path, _, outputs = asl_project
    assert main(["all", "--config", str(config_path)]) == 0
    records = read_manifest(outputs / "asl_citizen.parquet")
    write_manifest(
        [
            replace(record, split="test") if record.split == "train" else record
            for record in records
        ],
        outputs / "asl_citizen.parquet",
    )
    assert main(["create-splits", "--config", str(config_path)]) == 2
    assert "differs from its official CSV" in capsys.readouterr().err


def test_asl_cli_missing_video_reports_error_and_does_not_finalize_split(
    asl_project: tuple[Path, Path, Path],
) -> None:
    config_path, root, outputs = asl_project
    (root / "videos" / "train BOOK.mp4").unlink()
    assert main(["all", "--config", str(config_path)]) == 1
    report = json.loads((outputs / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"]
    assert report["issue_counts"]["by_code"]["video_missing"] == 1
    assert not (outputs / "split.json").exists()


def test_asl_cli_count_mismatch_is_a_global_error(asl_project: tuple[Path, Path, Path]) -> None:
    config_path, _, outputs = asl_project
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["expected"]["split_clips"]["test"] = 3
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert main(["all", "--config", str(config_path)]) == 1
    report = json.loads((outputs / "report.json").read_text(encoding="utf-8"))
    assert "unexpected_split_clip_count" in report["issue_counts"]["by_code"]
    assert not (outputs / "split.json").exists()


def test_asl_cli_rejects_json_override_before_writing_outputs(
    asl_project: tuple[Path, Path, Path],
) -> None:
    config_path, _, outputs = asl_project
    assert main(["all", "--config", str(config_path), "--official-split", "unused.json"]) == 2
    assert not outputs.exists()
