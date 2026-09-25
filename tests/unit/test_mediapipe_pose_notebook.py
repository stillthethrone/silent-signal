from __future__ import annotations

import ast
import json
import re
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/12_vsl400_mediapipe_pose_transformer.ipynb"


def _code() -> str:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in cells:
        ast.parse(source)
    return "\n".join(cells)


def test_notebook_runs_the_full_pose_pipeline_in_order() -> None:
    source = _code()

    steps = [
        "cli('fetch_vsl400_kaggle', 'metadata'",
        "cli('prepare', 'create-splits'",
        "cli('select_classes', *command)",
        "cli('fetch_vsl400_kaggle', 'keypoints'",
        "cli('train_pose_transformer', *arguments)",
        "history, report = load_run(RUN_ROOT)",
        "plot_training_curves(history, FIGURES_ROOT / 'training_curves.png'",
        "(RUN_ROOT / 'report.json')",
    ]
    positions = [source.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "CLASS_COUNT = 70" in source
    assert "'--graph-blocks', 2, '--spatial-layers', 2, '--spatial-heads', 4" in source
    assert "'--temporal-layers', 3, '--temporal-heads', 4" in source


def test_notebook_keeps_results_on_drive_and_guards_test_and_credentials() -> None:
    source = _code()

    assert "RESULTS_ROOT = Path('/content/drive/MyDrive/silent-signal-results')" in source
    assert "RUN_ROOT = SUBSET_ROOT / 'runs' / RUN_NAME" in source
    assert "RUN_TEST = False" in source and "arguments.append('--run-test')" in source
    assert "CONFIRM_KAGGLE_VSL400_PERMISSION = False" in source
    assert not re.search(r"print\([^\n]*KAGGLE_KEY", source)
    assert "sys.path.insert(0, str(PROJECT_ROOT / 'src'))" in source
