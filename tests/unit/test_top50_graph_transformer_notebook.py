from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/09_asl_citizen_top50_graph_transformer_training.ipynb"


def test_top50_graph_transformer_notebook_is_clean_and_preserves_official_split() -> None:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join("".join(cell["source"]) for cell in cells)

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(
        cell.get("execution_count") is None
        for cell in cells
        if cell["cell_type"] == "code"
    )
    assert "silent_signal.cli.select_ranked_subset" in source
    assert "silent_signal.cli.prepare_graph" in source
    assert "silent_signal.cli.check_graph_encoder" in source
    assert "silent_signal.cli.train_graph_transformer" in source
    assert "set(range(50))" in source
    assert "Split isolation: PASS" in source
    assert "--resume" in source
    assert "test_top1" in source
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
