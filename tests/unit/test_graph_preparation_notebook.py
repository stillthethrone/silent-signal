from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/05_asl_citizen_top200_graph_preparation.ipynb"


def test_graph_preparation_notebook_is_clean_and_pins_completed_pose_run() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell["source"]) for cell in cells)

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")
    assert "feat/asl-citizen-graph-preprocessing" in source
    assert "EXPECTED_CLIPS = 6146" in source
    assert "7e3dc11bc6fb25ff6b4311c9fe4f2175841ce6ad5eab89da4e5c8af0de596dfe" in source
    assert "ss-prepare-pose-graph" not in source
    assert "silent_signal.cli.prepare_graph" in source
    assert "--continue-on-error" in source
    assert "--progress-every" in source
    assert "OVERWRITE = False" in source
    assert "Graph cache trên Drive" in source
