from __future__ import annotations

import json
from pathlib import Path

from silent_signal.cli.prepare import main
from silent_signal.configuration import DatasetConfig
from silent_signal.data.manifest import read_manifest


def test_all_command_writes_validated_split_artifacts(
    config_file: Path,
    dataset_config: DatasetConfig,
) -> None:
    exit_code = main(["all", "--config", str(config_file), "--level", "metadata"])

    assert exit_code == 0
    assert dataset_config.outputs.manifest_csv.is_file()
    assert dataset_config.outputs.manifest_parquet.is_file()
    assert dataset_config.outputs.labels.is_file()
    assert dataset_config.outputs.split.is_file()
    assert dataset_config.outputs.report.is_file()
    records = read_manifest(dataset_config.outputs.manifest_parquet)
    assert {record.split for record in records} == {"train", "validation", "test"}
    report = json.loads(dataset_config.outputs.report.read_text(encoding="utf-8"))
    assert report["passed"] is True
