from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_TRAIN_NOTEBOOK = _ROOT / "notebooks/07_asl_citizen_top20_videomaev2_rgb_baseline.ipynb"
_ANALYSIS_NOTEBOOK = _ROOT / "notebooks/08_asl_citizen_top20_videomaev2_error_analysis.ipynb"


def _source(path: Path) -> tuple[dict[str, object], str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    return notebook, "\n".join("".join(cell["source"]) for cell in cells)


def _assert_clean(notebook: dict[str, object]) -> None:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")


def test_videomaev2_demo_notebook_is_bounded_reproducible_and_leak_free() -> None:
    notebook, source = _source(_TRAIN_NOTEBOOK)
    _assert_clean(notebook)
    assert "feat/asl-citizen-videomaev2-demo-baseline" in source
    assert "OpenGVLab/VideoMAEv2-Base" in source
    assert "CLASS_COUNT = 20" in source
    assert "selection['classes'][:CLASS_COUNT]" in source
    assert "SignFrequency(M)" in source
    assert "train', 'validation', 'test" in source
    assert "Split isolation: PASS" in source
    assert "train_test_split" not in source
    assert "MAX_TRAIN_BATCHES" in source
    assert "MAX_EVAL_BATCHES" in source
    assert "CHECKPOINT_EVERY" in source
    assert "RESUME = True" in source
    assert "RUN_TEST = False" in source
    assert "include_paths=selected_video_paths" in source
    assert "silent_signal.cli.train_videomaev2_demo" in source


def test_videomaev2_error_analysis_defaults_to_validation_and_visualizes_errors() -> None:
    notebook, source = _source(_ANALYSIS_NOTEBOOK)
    _assert_clean(notebook)
    assert "ANALYSIS_SPLIT = 'validation'" in source
    assert "ALLOW_TEST_ANALYSIS = False" in source
    assert "silent_signal.cli.analyze_videomaev2_demo" in source
    assert "confusion_matrices.png" in source
    assert "per_class_metrics.png" in source
    assert "top_confusions.png" in source
    assert "confidence_histogram.png" in source
    assert "selected_20_official_split_counts.png" in source
