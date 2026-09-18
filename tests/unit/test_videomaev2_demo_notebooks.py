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
    assert "feat/asl-citizen-videomaev2-demo-baseline" in source
    assert "OpenGVLab/VideoMAEv2-Base" in source
    assert "CLASS_COUNT = 50" in source
    assert "MAX_EPOCHS = 100" in source
    assert "classes = eligible[:CLASS_COUNT]" in source
    assert "SignFrequency(M)" in source
    assert "train', 'validation', 'test" in source
    assert "Official split isolation: PASS" in source
    assert "train_test_split" not in source
    assert "MAX_TRAIN_BATCHES" in source
    assert "MAX_EVAL_BATCHES" in source
    assert "MAX_TRAIN_BATCHES = 0" in source
    assert "MAX_EVAL_BATCHES = 0" in source
    assert "RGB_EMBEDDING_DIM = 64" in source
    assert "RGB_LAYERS = 1" in source
    assert "RGB_HEADS = 2" in source
    assert "RGB Transformer" in source
    assert "numpy==2.1.3" in source
    assert "USE_TF'] = '0'" in source
    assert "from transformers import PreTrainedModel" in source
    assert "Import check PASS" in source
    assert "PROJECT_SRC = str(PROJECT_ROOT / 'src')" in source
    assert source.index("PROJECT_SRC = str(PROJECT_ROOT / 'src')") < source.index(
        "import silent_signal, torch"
    )
    assert "Project import PASS" in source
    assert "CHECKPOINT_EVERY" in source
    assert "EARLY_STOPPING_PATIENCE = 10" in source
    assert "EARLY_STOPPING_MIN_DELTA = 0.001" in source
    assert "RGB_DROPOUT = 0.5" in source
    assert "LABEL_SMOOTHING = 0.15" in source
    assert "WEIGHT_DECAY = 0.04" in source
    assert "RANDOM_CROP_SCALE_MIN = 0.85" in source
    assert "COLOR_JITTER = 0.1" in source
    assert "GRADIENT_CLIP_NORM = 1.0" in source
    assert "--early-stopping-patience" in source
    assert "best_checkpoint.pt" in source
    assert "Macro-F1 is the unweighted mean of per-word F1 scores" in source
    assert "RESUME = False" in source
    assert "RUN_TEST = True" in source
    assert "CUSTOM_SIGNER_SPLIT" not in source
    assert "--resplit-by-signer" not in source
    assert "PERSIST_VIDEO_CACHE = True" in source
    assert "datasets/asl_citizen_top30" in source
    assert "reuses the existing cache" in source
    assert "videomaev2_rgb_transformer_demo50_official_compact64_v1" in source
    assert "persistent_video_cache.json" in source
    assert "include_paths=selected_video_paths" in source
    assert "silent_signal.cli.train_videomaev2_demo" in source


def test_videomaev2_error_analysis_defaults_to_validation_and_visualizes_errors() -> None:
    notebook, source = _source(_ANALYSIS_NOTEBOOK)
    _assert_clean(notebook)
    assert "ANALYSIS_SPLIT = 'validation'" in source
    assert "videomaev2_rgb_transformer_demo50_official_compact64_v1" in source
    assert "ALLOW_TEST_ANALYSIS = False" in source
    assert "Artifact guard PASS: official split, RGB embedding=64" in source
    assert "training_report.get('model', {}).get('rgb_embedding_dim') != 64" in source
    assert "drive_ready" in source
    assert "force_remount=True" in source
    assert "timeout_ms=120_000" in source
    assert "Environment PASS" in source
    assert "silent-signal-analysis-site-py313-v1" in source
    assert "numpy==2.2.2" in source
    assert "scipy==1.15.1" in source
    assert "--no-cache-dir" in source
    assert "--target" in source
    assert "ANALYSIS_ENV['PYTHONPATH']" in source
    assert "pip', 'install', '-q', '-e'" not in source
    assert "silent_signal.cli.analyze_videomaev2_demo" in source
    assert "confusion_matrices.png" in source
    assert "per_class_metrics.png" in source
    assert "f1_scores.png" in source
    assert "top_confusions.png" in source
    assert "confidence_histogram.png" in source
    assert "selected_50_official_split_counts.png" in source
    assert "training_curves.png" in source
    assert "generalization" in source
    assert "class_support" in source
    assert "Macro-F1" in source
