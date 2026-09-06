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


def test_asl_config_uses_official_singleview_release(tmp_path: Path) -> None:
    config = load_dataset_config("configs/dataset/asl_citizen.yaml", root_override=tmp_path)

    assert config.adapter == "asl_citizen"
    assert tuple(config.views) == ("single",)
    assert config.split.strategy == "official"
    assert config.split.ratios == {}
    assert config.metadata_splits == {
        "train": "splits/train.csv",
        "validation": "splits/val.csv",
        "test": "splits/test.csv",
    }
    assert config.expected.clips == 83399
    assert config.expected.glosses == 2731
    assert config.expected.signers == 52
    assert config.expected.fps is None
    assert config.expected.width is None
    assert config.expected.height is None
    assert config.expected.split_clips == {"train": 40154, "validation": 10304, "test": 32941}
    assert config.expected.split_signers == {"train": 35, "validation": 6, "test": 11}


@pytest.mark.parametrize("change", ["random_split", "ratio", "missing_csv", "same_csv", "views"])
def test_asl_rejects_configs_that_change_official_protocol(tmp_path: Path, change: str) -> None:
    payload = yaml.safe_load(Path("configs/dataset/asl_citizen.yaml").read_text(encoding="utf-8"))
    if change == "random_split":
        payload["split"]["strategy"] = "signer_disjoint"
    elif change == "ratio":
        payload["split"]["ratios"] = {"train": 0.8, "validation": 0.1, "test": 0.1}
    elif change == "missing_csv":
        payload["dataset"]["metadata_splits"].pop("test")
    elif change == "same_csv":
        payload["dataset"]["metadata_splits"]["test"] = "splits/train.csv"
    else:
        payload["dataset"]["views"] = {"front": {"directory": "videos"}}
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_dataset_config(config_path, root_override=tmp_path)
