from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from silent_signal.configuration import ConfigurationError, load_dataset_config


def test_root_override_does_not_require_environment(vsl400_root: Path) -> None:
    config = load_dataset_config(
        "configs/dataset/vsl400.yaml",
        root_override=vsl400_root,
    )

    assert config.root == vsl400_root.resolve()
    assert config.split.ratios == {"train": 0.8, "validation": 0.1, "test": 0.1}


def test_missing_root_environment_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSL400_ROOT", raising=False)

    with pytest.raises(ConfigurationError, match="VSL400_ROOT"):
        load_dataset_config("configs/dataset/vsl400.yaml")


def test_rejects_non_disjoint_split_strategy(
    config_file: Path,
) -> None:
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    payload["split"]["strategy"] = "random"
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="signer_disjoint"):
        load_dataset_config(config_file)
