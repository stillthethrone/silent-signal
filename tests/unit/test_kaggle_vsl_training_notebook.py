from __future__ import annotations

import json
from pathlib import Path


def test_notebook_12_uses_small_model_progress_logging_and_releases_preview() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/12_kaggle_vsl_mediapipe_graph_training.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "mediapipe_upper68_pose_transformer_small_top70_v3" in source
    assert "GRAPH_DIM = 96" in source
    assert "TEMPORAL_DIM = 192" in source
    assert "TEMPORAL_LAYERS = 2" in source
    assert "DROPOUT = 0.30" in source
    assert "--progress-every-batches" in source
    assert "del packed, sample_features, sample_joint_mask, sample_frame_mask" in source
