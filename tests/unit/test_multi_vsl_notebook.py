from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_NOTEBOOK = _ROOT / "notebooks/09_multi_vsl_top50_videomaev2_rgb_transformer_baseline.ipynb"


def _source() -> str:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_multi_vsl_notebook_keeps_source_video_out_of_drive() -> None:
    source = _source()

    assert "VIDEO_ROOT = RUNTIME_ROOT / 'videos'" in source
    assert "RESULTS_ROOT = Path('/content/drive/MyDrive/" in source
    assert "--video-root', VIDEO_ROOT" in source
    assert "--output-root', RUN_ROOT" in source


def test_multi_vsl_notebook_uses_compact_regularized_baseline() -> None:
    source = _source()

    assert "RGB_EMBEDDING_DIM = 64" in source
    assert "RGB_LAYERS = 1" in source
    assert "RGB_HEADS = 2" in source
    assert "RGB_DROPOUT = 0.50" in source
    assert "WEIGHT_DECAY = 0.04" in source
    assert "LABEL_SMOOTHING = 0.15" in source
    assert "EARLY_STOPPING_PATIENCE = 6" in source
    assert "RUN_TEST = False" in source


def test_multi_vsl_notebook_uses_official_metadata_and_validates_gpu() -> None:
    source = _source()

    assert "Etdihatthoc/Multi-VSL_WACV_2025" in source
    assert "label_1_200" in source
    assert "prepare_multi_vsl_demo" in source
    assert "torch.cuda.get_device_name(0)" in source
