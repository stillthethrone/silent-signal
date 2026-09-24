from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/11_vsl400_rtmpose_pose_extraction.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_vsl400_notebook_cells_are_valid_python() -> None:
    for source in _code_cells():
        ast.parse(source)


def test_vsl400_notebook_splits_full_dataset_before_selecting_classes() -> None:
    source = "\n".join(_code_cells())

    prepare = source.index("[PREPARE_CLI, 'all'")
    select = source.index("[SELECT_CLI, '--manifest', FULL_MANIFEST")
    validate = source.index("[PREPARE_CLI, 'validate'")
    extract = source.index("[POSE_CLI, 'extract'")
    assert prepare < select < validate < extract
    assert "configs/dataset/vsl400.yaml" in source
    assert "LOCAL_DATASET_ROOT = Path('/content/VSL400_subset')" in source
    assert "'--dataset-root', LOCAL_DATASET_ROOT" in source
    assert "RESULTS_ROOT / 'splits/vsl400_signer_split.json'" in source


def test_vsl400_notebook_pins_rtmpose_environment_and_provenance() -> None:
    source = "\n".join(_code_cells())

    for pin in ("numpy==1.26.4", "torch==2.1.0", "mmcv==2.1.0", "mmdet==3.2.0", "v1.3.2"):
        assert pin in source
    assert "checkpoint_sha256" in source
    assert "'extractor_fingerprint'" in source
    assert "coco_wholebody_75_t64.yaml" in source
