from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _ROOT / "notebooks/03_rtmpose_wholebody_colab_check.ipynb"


def _code_cells() -> dict[str, str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def test_pose_notebook_is_valid_cleared_python() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(_NOTEBOOK), "exec")
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_pose_notebook_stream_extracts_official_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "ASL_Citizen"
    calls: list[tuple[Path, int]] = []

    def extract(root: Path, *, reserve_bytes: int) -> Path:
        calls.append((root, reserve_bytes))
        (root / "videos").mkdir(parents=True)
        (root / "splits").mkdir()
        for split in ("train", "val", "test"):
            (root / "splits" / f"{split}.csv").write_text("header\n", encoding="utf-8")
        return root

    monkeypatch.setattr("silent_signal.data.asl_download.extract_remote_archive", extract)
    namespace = {
        "PROJECT_ROOT": _ROOT,
        "DATASET_ROOT": destination,
        "ARCHIVE_PATH": tmp_path / "release/ASL_Citizen.zip",
        "STREAM_EXTRACT_DATASET": True,
        "DOWNLOAD_DATASET_ARCHIVE": False,
        "EXTRACT_DATASET_ARCHIVE": False,
    }
    exec(_code_cells()["prepare-dataset"], namespace)
    assert calls == [(destination, 2 * 1024**3)]
