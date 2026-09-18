from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from silent_signal.cli.train_graph_transformer import main  # noqa: E402
from silent_signal.contracts import ManifestRecord  # noqa: E402
from silent_signal.data.manifest import write_manifest  # noqa: E402
from silent_signal.pose.cache import pose_cache_path  # noqa: E402
from silent_signal.preprocessing.cache import write_graph_pose_cache  # noqa: E402
from silent_signal.preprocessing.pose_features import GraphPoseSample  # noqa: E402


def test_full_graph_transformer_training_writes_best_and_test_report(
    tmp_path: Path,
) -> None:
    records = tuple(
        _record(class_index, split, copy)
        for split, copies in (("train", 2), ("validation", 1), ("test", 1))
        for class_index in range(3)
        for copy in range(copies)
    )
    manifest = tmp_path / "manifest.csv"
    graph_root = tmp_path / "graph"
    output_root = tmp_path / "training"
    config = tmp_path / "config.yaml"
    write_manifest(records, manifest)
    for record in records:
        write_graph_pose_cache(
            pose_cache_path(graph_root, record.sample_id),
            _sample(record),
        )
    config.write_text(
        f"""schema_version: 1
model:
  architecture: graph_spatial_temporal_transformer_v1
  input_dim: 7
  num_nodes: 5
  max_frames: 4
  hidden_dim: 8
  embedding_dim: 12
  num_blocks: 1
  temporal_kernel: 3
  spatial_layers: 1
  temporal_layers: 1
  num_heads: 2
  ffn_dim: 16
  dropout: 0.0
  num_classes: 3
smoke:
  batch_size: 2
  train_steps: 1
  sample_limit: 3
  learning_rate: 0.001
  weight_decay: 0.0
  seed: 7
training:
  batch_size: 3
  max_epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0
  label_smoothing: 0.0
  gradient_clip: 1.0
  early_stopping_min_epochs: 1
  early_stopping_patience: 1
  early_stopping_min_delta: 0.0
  num_workers: 0
  seed: 7
expected:
  manifest_sha256: auto
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
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
        ]
    )

    report = json.loads((output_root / "evaluation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert (output_root / "best.pt").is_file()
    assert (output_root / "last.pt").is_file()
    assert report["state"] == "completed"
    assert report["split"]["clip_counts"] == {
        "train": 6,
        "validation": 3,
        "test": 3,
    }
    assert report["test"]["samples"] == 3
    assert len(report["test"]["per_class"]) == 3


def _record(class_index: int, split: str, copy: int) -> ManifestRecord:
    sample_id = f"{split}-{class_index}-{copy}"
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=f"{sample_id}.mp4",
        signer_id=f"signer-{split}",
        gloss_id=f"g-{class_index}",
        gloss_name=f"WORD_{class_index}",
        class_index=class_index,
        view="single",
        video_path=f"videos/{sample_id}.mp4",
        split=split,
    )


def _sample(record: ManifestRecord) -> GraphPoseSample:
    rng = np.random.default_rng(abs(hash(record.sample_id)) % (2**32))
    return GraphPoseSample(
        sample_id=record.sample_id,
        class_index=record.class_index,
        split=str(record.split),
        features=rng.normal(size=(4, 5, 7)).astype(np.float32),
        joint_mask=np.ones((4, 5), dtype=np.bool_),
        observed_mask=np.ones((4, 5), dtype=np.bool_),
        frame_mask=np.ones((4,), dtype=np.bool_),
        source_frame_indices=np.arange(4, dtype=np.int64),
        timestamps_seconds=np.arange(4, dtype=np.float64),
        adjacency=np.eye(5, dtype=np.float32),
        metadata={"preprocessing_fingerprint": "f" * 64},
    )
