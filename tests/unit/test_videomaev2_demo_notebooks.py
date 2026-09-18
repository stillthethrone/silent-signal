from __future__ import annotations

import ast
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_TRAIN_NOTEBOOK = _ROOT / "notebooks/07_asl_citizen_top30_videomaev2_rgb_transformer_baseline.ipynb"
_ANALYSIS_NOTEBOOK = _ROOT / "notebooks/08_asl_citizen_top30_videomaev2_error_analysis.ipynb"


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
    assert "PROJECT_GIT_REF = 'dev'" in source
    assert "OpenGVLab/VideoMAEv2-Base" in source
    assert "CLASS_COUNT = 50" in source
    assert "MAX_EPOCHS = 50" in source
    assert "classes = list(selection['classes'])" in source
    assert "không chọn lại dữ liệu" in source
    assert "train', 'validation', 'test" in source
    assert "Split isolation: PASS" in source
    assert "train_test_split" not in source
    assert "MAX_TRAIN_BATCHES" in source
    assert "MAX_EVAL_BATCHES" in source
    assert "MAX_TRAIN_BATCHES = 0" in source
    assert "MAX_EVAL_BATCHES = 0" in source
    assert "RGB_LAYERS = 2" in source
    assert "RGB_HEADS = 8" in source
    assert "RGB Transformer" in source
    assert "numpy==2.1.3" in source
    assert "USE_TF'] = '0'" in source
    assert "from transformers import PreTrainedModel" in source
    assert "Import check PASS" in source
    assert "CHECKPOINT_EVERY" in source
    assert "RESUME = True" in source
    assert "RUN_TEST = False" in source
    assert "PERSIST_VIDEO_CACHE = True" in source
    assert "datasets/asl_citizen_top30" in source
    assert "tái sử dụng cache top-30" in source
    assert "videomaev2_rgb_transformer_demo50_e50_early_stopping" in source
    assert "SOURCE_BASELINE_ROOT" in source
    assert "SOURCE_MANIFEST_ROOT" in source
    assert "SOURCE_SPLIT_MANIFESTS" in source
    assert "SOURCE_MANIFEST_ROOT / 'all.csv'" in source
    assert "selected_50_words.json" in source
    assert "split_ids != ids[split]" in source
    assert "EARLY_STOPPING_MIN_EPOCHS = 10" in source
    assert "EARLY_STOPPING_PATIENCE = 8" in source
    assert "EARLY_STOPPING_MIN_DELTA = 0.001" in source
    assert "LR_PLATEAU_PATIENCE = 3" in source
    assert "LR_PLATEAU_FACTOR = 0.5" in source
    assert "LABEL_SMOOTHING = 0.05" in source
    assert "--early-stopping-patience" in source
    assert "--lr-plateau-factor" in source
    assert "persistent_video_cache.json" in source
    assert "include_paths=selected_video_paths" in source
    assert "silent_signal.cli.train_videomaev2_demo" in source


def test_videomaev2_error_analysis_defaults_to_validation_and_visualizes_errors() -> None:
    notebook, source = _source(_ANALYSIS_NOTEBOOK)
    _assert_clean(notebook)
    assert "ANALYSIS_SPLIT = 'validation'" in source
    assert "videomaev2_rgb_transformer_demo50_e50_early_stopping" in source
    assert "ALLOW_TEST_ANALYSIS = False" in source
    assert "silent_signal.cli.analyze_videomaev2_demo" in source
    assert "confusion_matrices.png" in source
    assert "per_class_metrics.png" in source
    assert "top_confusions.png" in source
    assert "confidence_histogram.png" in source
    assert "selected_50_official_split_counts.png" in source
