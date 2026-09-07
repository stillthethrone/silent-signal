from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from silent_signal.data.manifest import read_manifest
from silent_signal.data.validation import MediaInfo

_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks/00_asl_citizen_colab_preparation.ipynb"


def _code_cells() -> dict[str, str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def test_notebook_is_valid_cleared_python() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(_NOTEBOOK), "exec")
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize(
    ("extract", "expected_reserve"),
    [
        (False, 2 * 1024**3),
        (True, 49_604_368_459 + 2 * 1024**3),
    ],
)
def test_notebook_reserves_only_requested_space_before_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extract: bool,
    expected_reserve: int,
) -> None:
    calls: list[int] = []

    def download(destination: Path, *, reserve_bytes: int) -> Path:
        calls.append(reserve_bytes)
        return destination

    monkeypatch.setattr("silent_signal.data.asl_download.download_archive", download)
    namespace = {
        "ARCHIVE_PATH": tmp_path / "release/dataset.zip",
        "DATASET_ROOT": tmp_path / "dataset",
        "DOWNLOAD": True,
        "EXTRACT": extract,
        "STREAM_EXTRACT": False,
        "shutil": shutil,
    }
    exec(_code_cells()["download"], namespace)
    assert calls == [expected_reserve]


def test_notebook_stream_extract_dispatches_without_local_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "ASL_Citizen"
    calls: list[Path] = []

    def extract(root: Path) -> Path:
        calls.append(root)
        (root / "videos").mkdir(parents=True)
        (root / "splits").mkdir()
        for split in ("train", "val", "test"):
            (root / "splits" / f"{split}.csv").write_text("header\n", encoding="utf-8")
        return root

    monkeypatch.setattr("silent_signal.data.asl_download.extract_remote_archive", extract)
    namespace = {
        "ARCHIVE_PATH": tmp_path / "missing.zip",
        "DATASET_ROOT": destination,
        "STREAM_EXTRACT": True,
        "EXTRACT": False,
    }
    exec(_code_cells()["extract"], namespace)
    assert calls == [destination]


def test_notebook_persists_outputs_and_resumes_partial_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute preparation/media cells offline against a tiny real CLI dataset."""

    root = tmp_path / "ASL_Citizen"
    (root / "videos").mkdir(parents=True)
    (root / "splits").mkdir()
    for index, split in enumerate(("train", "val", "test")):
        filename = f"clip-{index}.mp4"
        (root / "videos" / filename).write_bytes(b"synthetic-video")
        with (root / "splits" / f"{split}.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
            writer.writerow([f"person-{index}", filename, "HELLO", "hello"])
    source = yaml.safe_load(
        (_ROOT / "configs/dataset/asl_citizen.yaml").read_text(encoding="utf-8")
    )
    source["expected"] = {
        "clips": 3,
        "glosses": 1,
        "signers": 3,
        "views_per_instance": 1,
        "enforce_counts": True,
        "split_clips": {"train": 1, "validation": 1, "test": 1},
        "split_signers": {"train": 1, "validation": 1, "test": 1},
    }
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    result_root = tmp_path / "drive/asl_citizen"
    result_root.mkdir(parents=True)
    probes: list[Path] = []
    decodes: list[Path] = []

    def probe(path: Path) -> MediaInfo:
        probes.append(path)
        return MediaInfo(30, 1.0, 30.0, 640, 480, "h264")

    monkeypatch.setattr("silent_signal.data.validation.probe_video", probe)
    monkeypatch.setattr("silent_signal.data.validation.decode_video", decodes.append)
    namespace: dict[str, Any] = {
        "Path": Path,
        "json": json,
        "subprocess": subprocess,
        "sys": sys,
        "PROJECT_ROOT": _ROOT,
        "PROJECT_COMMIT": "test-commit",
        "SOURCE_CONFIG": source_path,
        "DATASET_ROOT": root,
        "RESULT_ROOT": result_root,
        "MEDIA_WORKERS": 1,
        "MEDIA_BATCH_SIZE": 2,
        "CACHE_MODE": "keep",
        "RETRY_MEDIA_ERRORS": False,
        "RUN_PROBE": True,
        "RUN_FULL_DECODE": True,
        "DECODE_LIMIT": 1,
    }
    cells = _code_cells()
    for cell_id in ("prepare", "summary", "cache-functions", "probe", "decode", "final-status"):
        exec(cells[cell_id], namespace)
    assert len(probes) == 3
    assert len(decodes) == 1
    status_path = result_root / "reports/preparation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["probe"]["state"] == "passed"
    assert status["decode"]["state"] == "partial"
    assert not (result_root / "manifests/asl_citizen.decoded.parquet").exists()
    assert all(
        Path(path).is_relative_to(result_root)
        for path in namespace["config_payload"]["outputs"].values()
    )

    namespace["DECODE_LIMIT"] = 0
    exec(cells["decode"], namespace)
    assert len(decodes) == 3
    decoded = read_manifest(result_root / "manifests/asl_citizen.decoded.parquet")
    assert len(decoded) == 3
    assert all(record.is_valid and record.fps == 30 for record in decoded)
    assert {record.split for record in decoded} == {"train", "validation", "test"}
    assert namespace["status"]["decode"]["state"] == "passed"

    exec(cells["probe"], namespace)
    exec(cells["decode"], namespace)
    assert len(probes) == len(decodes) == 3
    (root / "videos/clip-0.mp4").write_bytes(b"changed-video")
    namespace["RUN_PROBE"] = False
    exec(cells["probe"], namespace)
    assert namespace["status"]["probe"]["state"] == "partial"
    assert namespace["status"]["probe"]["remaining"] == 1

    prior_cache = namespace["CACHE_ROOT"]
    namespace["CACHE_MODE"] = "restart"
    exec(cells["cache-functions"], namespace)
    assert namespace["CACHE_ROOT"] != prior_cache
    assert (prior_cache / "probe.jsonl").is_file()
