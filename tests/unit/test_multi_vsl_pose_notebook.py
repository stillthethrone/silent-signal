from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/10_multi_vsl_rtmpose_pose_extraction.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_pose_notebook_cells_are_valid_python() -> None:
    for source in _code_cells():
        ast.parse(source)


def test_pose_notebook_keeps_official_split_and_videos_out_of_drive() -> None:
    source = "\n".join(_code_cells())

    assert "Etdihatthoc/Multi-VSL_WACV_2025" in source
    assert "'list-videos'" in source
    assert "'build'" in source
    assert "VIDEO_ROOT = RUNTIME_ROOT / 'videos'" in source
    assert "VIEW_MODE = 'three_view'" in source
    assert "'--views', VIEW_MODE" in source
    assert "RESULTS_ROOT = Path('/content/drive/MyDrive/" in source
    assert "md5Checksum" in source
    assert "'--split', 'train', '--limit', PILOT_LIMIT" in source
    assert "'--num-shards', NUM_SHARDS" in source


def test_pose_notebook_pins_rtmpose_environment_and_provenance() -> None:
    source = "\n".join(_code_cells())

    for pin in ("numpy==1.26.4", "torch==2.1.0", "mmcv==2.1.0", "mmdet==3.2.0", "v1.3.2"):
        assert pin in source
    assert "checkpoint_sha256" in source
    assert "'extractor_fingerprint'" in source
    assert "coco_wholebody_75_t64.yaml" in source
