from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from silent_signal.cli.check_graph_encoder import main  # noqa: E402
from silent_signal.contracts import ManifestRecord  # noqa: E402
from silent_signal.data.manifest import write_manifest  # noqa: E402
from silent_signal.pose.cache import pose_cache_path, sha256_file  # noqa: E402
from silent_signal.preprocessing.cache import write_graph_pose_cache  # noqa: E402
from silent_signal.preprocessing.pose_features import GraphPoseSample  # noqa: E402


def test_graph_encoder_cli_writes_checkpoint_and_report(tmp_path: Path) -> None:
    records = tuple(_record(index) for index in range(3))
    manifest = tmp_path / "manifest.csv"
    graph_root = tmp_path / "graph"
    config = tmp_path / "encoder.yaml"
    checkpoint = tmp_path / "smoke.pt"
    report = tmp_path / "smoke.json"
    write_manifest(records, manifest)
    for record in records:
        write_graph_pose_cache(
            pose_cache_path(graph_root, record.sample_id),
            _sample(record),
        )
    config.write_text(
        f"""schema_version: 1
model:
  input_dim: 7
  num_nodes: 5
  max_frames: 4
  hidden_dim: 8
  embedding_dim: 12
  num_blocks: 1
  temporal_kernel: 3
  dropout: 0.0
  num_classes: 3
smoke:
  batch_size: 2
  train_steps: 1
  sample_limit: 3
  learning_rate: 0.001
  weight_decay: 0.0
  seed: 7
expected:
  manifest_sha256: {sha256_file(manifest)}
  preprocessing_fingerprint: {'f' * 64}
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--graph-root",
            str(graph_root),
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert checkpoint.is_file()
    assert payload["state"] == "passed"
    assert payload["logits_shape"] == [2, 3]
    assert payload["mask_invariance_max_error"] <= 1e-5


def _record(index: int) -> ManifestRecord:
    return ManifestRecord(
        sample_id=f"sample-{index}",
        instance_id=f"instance-{index}",
        video_id=f"video-{index}",
        signer_id=f"signer-{index}",
        gloss_id=str(index),
        gloss_name=f"word-{index}",
        class_index=index,
        view="single",
        video_path=f"videos/{index}.mp4",
        split="train",
    )


def _sample(record: ManifestRecord) -> GraphPoseSample:
    return GraphPoseSample(
        sample_id=record.sample_id,
        class_index=record.class_index,
        split="train",
        features=np.random.default_rng(record.class_index).normal(
            size=(4, 5, 7)
        ).astype(np.float32),
        joint_mask=np.ones((4, 5), dtype=np.bool_),
        observed_mask=np.ones((4, 5), dtype=np.bool_),
        frame_mask=np.ones((4,), dtype=np.bool_),
        source_frame_indices=np.arange(4, dtype=np.int64),
        timestamps_seconds=np.arange(4, dtype=np.float64),
        adjacency=np.eye(5, dtype=np.float32),
        metadata={"preprocessing_fingerprint": "f" * 64},
    )
