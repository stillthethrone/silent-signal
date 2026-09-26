from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).parents[2]
_REFERENCE = _ROOT / "notebooks/12_vsl400_mediapipe_pose_transformer.ipynb"
_REFERENCE_CLASSES = 70
# Notebook 12 and its copies for more glosses (14: 200, 15: 400), each on its own branch;
# every copy present in the checkout is tested.
_NOTEBOOKS = sorted(_ROOT.glob("notebooks/*_vsl400_mediapipe_pose_transformer*.ipynb"))


def _cells(path: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return cells


def _code(path: Path) -> str:
    cells = _cells(path)
    assert all(cell.get("outputs", []) == [] for cell in cells)
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    return "\n".join(sources)


def _class_count(path: Path) -> int:
    match = re.search(r"^CLASS_COUNT = (\d+)  #", _code(path), re.MULTILINE)
    assert match, f"{path.name} has no CLASS_COUNT parameter"
    return int(match.group(1))


def _number(path: Path) -> str:
    return path.name.split("_", 1)[0]


def test_notebook_12_is_the_70_gloss_reference() -> None:
    assert _REFERENCE in _NOTEBOOKS
    assert _class_count(_REFERENCE) == _REFERENCE_CLASSES


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_number)
def test_notebook_runs_the_full_pose_pipeline_in_order(path: Path) -> None:
    source = _code(path)

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
    assert "'--graph-blocks', 2, '--spatial-layers', 2, '--spatial-heads', 4" in source
    assert "'--temporal-layers', 3, '--temporal-heads', 4" in source


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_number)
def test_notebook_keeps_results_on_drive_and_guards_test_and_credentials(path: Path) -> None:
    source = _code(path)

    assert "RESULTS_ROOT = Path('/content/drive/MyDrive/silent-signal-results')" in source
    assert "RUN_ROOT = SUBSET_ROOT / 'runs' / RUN_NAME" in source
    assert "RUN_TEST = False" in source and "arguments.append('--run-test')" in source
    assert "CONFIRM_KAGGLE_VSL400_PERMISSION = False" in source
    assert not re.search(r"print\([^\n]*KAGGLE_KEY", source)
    assert "sys.path.insert(0, str(PROJECT_ROOT / 'src'))" in source


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_number)
def test_notebook_differs_from_notebook_12_only_in_its_class_count(path: Path) -> None:
    reference, variant = _cells(_REFERENCE), _cells(path)
    classes = _class_count(path)

    assert [cell["cell_type"] for cell in reference] == [cell["cell_type"] for cell in variant]
    for expected, actual in zip(reference, variant, strict=True):
        # Compare digit runs one by one: "400" also appears in "VSL400" and "400 từ".
        expected_parts = re.split(r"(\d+)", _without_title_number("".join(expected["source"])))
        actual_parts = re.split(r"(\d+)", _without_title_number("".join(actual["source"])))
        assert len(expected_parts) == len(actual_parts)
        for left, right in zip(expected_parts, actual_parts, strict=True):
            assert left == right or (left, right) == (str(_REFERENCE_CLASSES), str(classes))


def _without_title_number(source: str) -> str:
    return re.sub(r"\A# \d+ —", "# NB —", source)
