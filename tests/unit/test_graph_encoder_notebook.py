from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/06_asl_citizen_top200_graph_encoder_check.ipynb"


def test_graph_encoder_notebook_is_clean_and_checks_preprocessing() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell["source"]) for cell in cells)

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")
    assert "feat/asl-citizen-graph-encoder" in source
    assert "EXPECTED_CLIPS = 6146" in source
    assert "silent_signal.cli.check_graph_encoder" in source
    assert "RUN_SMOKE_TRAINING = True" in source
    assert "OVERWRITE_CHECKPOINT = True" in source
    assert "full training" in source
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
