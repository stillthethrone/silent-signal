from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "notebooks/13_kaggle_vsl_allclass_rgb_preparation.ipynb"
TRAINING = ROOT / "notebooks/14_kaggle_vsl_allclass_rgb_pose_fusion_training.ipynb"


def _sources(path: Path) -> tuple[str, list[str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code):
        compile(source, f"{path.name}:cell-{index}", "exec")
    return "\n".join(code), code


def test_notebook_13_builds_all_eligible_and_pins_rgb_extraction() -> None:
    source, _cells = _sources(PREPARATION)

    assert "'--classes', '0'" in source
    assert "Actual eligible classes" in source
    assert "OpenGVLab/VideoMAEv2-Base" in source
    assert "0e826d7e85e39f9d951e331cd91c5c2d8142d385" in source
    assert "transformers==4.57.6" in source
    assert "completed_output_is_current" in source
    assert "silent_signal.cli.fetch_kaggle_vsl_rgb" in source
    assert "RGB_SOURCE_READY" in source
    assert "all_eligible_{VERSION_TAG}_min" in source
    assert "processed_augmented" not in source


def test_notebook_14_consumes_same_contract_and_runs_test_after_selection() -> None:
    source, _cells = _sources(TRAINING)

    assert "all_eligible_latest_min2_val0p2_seed42" in source
    assert "CLASS_COUNT = len({row.class_index for row in records})" in source
    assert "silent_signal.cli.train_rgb_pose_fusion" in source
    assert "command.append('--run-test')" in source
    assert "Pose pack failed SHA-256 verification" in source
    assert "validation_macro_f1" in source
    assert "confusion_test" not in source  # plotting is split-driven, not test-selection logic
    assert "top70" not in source.lower()
