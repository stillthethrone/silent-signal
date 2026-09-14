from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/04_asl_citizen_top200_pose_extraction.ipynb"


def _load_notebook() -> dict[str, object]:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def test_top200_notebook_is_clean_and_has_expected_sequence() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")

    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "Không cần chạy notebook `00` hay `03` trước" in source
    assert "STREAM_EXTRACT_DATASET = True" in source
    assert "ACCEPT_ASL_CITIZEN_LICENSE = False" in source
    assert "PREPARE_MANIFEST = True" in source
    assert "extract_remote_archive" in source
    assert "numpy==1.26.4" in source
    assert "torch==2.1.0" in source
    assert "mmcv==2.1.0" in source
    assert "DOWNLOAD_MODELS = True" in source
    assert "SignFrequency(M)" in source
    assert "ss-select-asl-subset" in source
    assert "ss-extract-pose" in source
    assert "RUN_FULL_EXTRACTION = True" in source
    assert "--num-shards" in source
    assert "--continue-on-error" in source


def test_top200_notebook_avoids_kernel_binary_data_imports() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "exec(environment_check)" not in source
    assert "run([POSE_PY, '-c', environment_check])" in source
    assert "str(POSE_ENV_ROOT / 'bin/ss-select-asl-subset')" in source
