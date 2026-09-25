from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from silent_signal.cli.train_pose_graph import main  # noqa: E402
from silent_signal.contracts import LabelDefinition, ManifestRecord  # noqa: E402
from silent_signal.data.manifest import write_labels, write_manifest  # noqa: E402
from silent_signal.pose.cache import pose_cache_path, sha256_file  # noqa: E402
from silent_signal.preprocessing.cache import write_graph_pose_cache  # noqa: E402
from silent_signal.preprocessing.pose_features import GraphPoseSample  # noqa: E402


def test_train_pose_graph_writes_metrics_checkpoints_and_plots(tmp_path: Path) -> None:
    records = tuple(
        _record(split, class_index, item)
        for split, count in (("train", 4), ("validation", 2), ("test", 2))
        for class_index in range(2)
        for item in range(count)
    )
    manifest = tmp_path / "manifest.csv"
    labels = tmp_path / "labels.json"
    graph_root = tmp_path / "graph"
    output_root = tmp_path / "run"
    config = tmp_path / "experiment.yaml"
    write_manifest(records, manifest)
    write_labels(
        tuple(
            LabelDefinition(index, f"label-{index}", f"word-{index}") for index in range(2)
        ),
        labels,
        dataset="test",
    )
    for record in records:
        write_graph_pose_cache(
            pose_cache_path(graph_root, record.sample_id),
            _sample(record),
        )
    config.write_text(
        f"""schema_version: 1
evaluation_protocol: provided_test_plus_sample_validation
model:
  architecture: graph_spatial_temporal_transformer_v1
  input_dim: 7
  num_nodes: 5
  max_frames: 4
  hidden_dim: 8
  embedding_dim: 8
  num_blocks: 1
  temporal_kernel: 3
  spatial_layers: 1
  temporal_layers: 1
  num_heads: 2
  ffn_dim: 16
  dropout: 0.1
  num_classes: 2
smoke:
  batch_size: 2
  train_steps: 1
  sample_limit: 4
  learning_rate: 0.001
  weight_decay: 0.0
  seed: 3
training:
  batch_size: 2
  max_epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0
  label_smoothing: 0.0
  gradient_clip: 1.0
  early_stopping_min_epochs: 1
  early_stopping_patience: 2
  early_stopping_min_delta: 0.0
  num_workers: 0
  seed: 3
  coordinate_scale_jitter: 0.0
  coordinate_translation_jitter: 0.0
  joint_dropout: 0.0
expected:
  manifest_sha256: {sha256_file(manifest)}
  preprocessing_fingerprint: {"f" * 64}
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--labels",
            str(labels),
            "--graph-root",
            str(graph_root),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--progress-every",
            "0",
            "--run-test",
        ]
    )

    report = json.loads((output_root / "training_report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["architecture"] == "graph_spatial_temporal_transformer_v1"
    assert report["completed_epochs"] == 2
    assert report["test_was_run"] is True
    assert set(report["evaluation"]) == {"train", "validation", "test"}
    for artifact in (
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "history.json",
        "training_curves.png",
        "split_metrics_comparison.png",
        "test_confusion_matrix.png",
        "test_per_class_f1.png",
        "test_predictions.csv",
    ):
        assert (output_root / artifact).is_file()


def _record(split: str, class_index: int, item: int) -> ManifestRecord:
    sample_id = f"{split}-{class_index}-{item}"
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=sample_id,
        signer_id="unknown",
        gloss_id=f"label-{class_index}",
        gloss_name=f"word-{class_index}",
        class_index=class_index,
        view="front",
        video_path=f"{split}/{class_index}/{item}.npy",
        split=split,
    )


def _sample(record: ManifestRecord) -> GraphPoseSample:
    rng = np.random.default_rng(abs(hash(record.sample_id)) % (2**32))
    features = rng.normal(0.0, 0.05, size=(4, 5, 7)).astype(np.float32)
    features[..., 0] += float(record.class_index) * 0.75
    return GraphPoseSample(
        sample_id=record.sample_id,
        class_index=record.class_index,
        split=str(record.split),
        features=features,
        joint_mask=np.ones((4, 5), dtype=np.bool_),
        observed_mask=np.ones((4, 5), dtype=np.bool_),
        frame_mask=np.ones((4,), dtype=np.bool_),
        source_frame_indices=np.arange(4, dtype=np.int64),
        timestamps_seconds=np.arange(4, dtype=np.float64),
        adjacency=np.eye(5, dtype=np.float32),
        metadata={"preprocessing_fingerprint": "f" * 64},
    )
