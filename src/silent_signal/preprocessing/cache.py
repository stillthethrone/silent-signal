"""Atomic storage for graph-ready pose tensors."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from silent_signal.pose.cache import PoseCacheError
from silent_signal.preprocessing.pose_features import GraphPoseSample


def write_graph_pose_cache(
    path: str | Path,
    sample: GraphPoseSample,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save one graph sample without pickle serialization."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise PoseCacheError(f"Graph pose cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "class_index": sample.class_index,
        "split": sample.split,
        "metadata": sample.metadata,
    }
    metadata_json = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.frombuffer(metadata_json, dtype=np.uint8),
                features=sample.features,
                joint_mask=sample.joint_mask,
                observed_mask=sample.observed_mask,
                frame_mask=sample.frame_mask,
                source_frame_indices=sample.source_frame_indices,
                timestamps_seconds=sample.timestamps_seconds,
                adjacency=sample.adjacency,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PoseCacheError(f"Cannot write graph pose cache {destination}: {exc}") from exc


def read_graph_pose_cache(
    path: str | Path,
    *,
    expected_sample_id: str | None = None,
    expected_fingerprint: str | None = None,
) -> GraphPoseSample:
    """Load and validate one graph-ready cache."""

    source = Path(path)
    required = {
        "metadata_json",
        "features",
        "joint_mask",
        "observed_mask",
        "frame_mask",
        "source_frame_indices",
        "timestamps_seconds",
        "adjacency",
    }
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise PoseCacheError(f"Graph pose cache is missing arrays: {sorted(missing)}")
            envelope = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if envelope.get("schema_version") != 1:
                raise PoseCacheError("Unsupported graph pose cache schema_version.")
            sample = GraphPoseSample(
                sample_id=str(envelope["sample_id"]),
                class_index=int(envelope["class_index"]),
                split=str(envelope["split"]),
                features=archive["features"].astype(np.float32, copy=False),
                joint_mask=archive["joint_mask"].astype(np.bool_, copy=False),
                observed_mask=archive["observed_mask"].astype(np.bool_, copy=False),
                frame_mask=archive["frame_mask"].astype(np.bool_, copy=False),
                source_frame_indices=archive["source_frame_indices"].astype(
                    np.int64, copy=False
                ),
                timestamps_seconds=archive["timestamps_seconds"].astype(
                    np.float64, copy=False
                ),
                adjacency=archive["adjacency"].astype(np.float32, copy=False),
                metadata=dict(envelope.get("metadata", {})),
            )
    except PoseCacheError:
        raise
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PoseCacheError(f"Cannot read graph pose cache {source}: {exc}") from exc
    _validate_graph_sample(sample)
    if expected_sample_id is not None and sample.sample_id != expected_sample_id:
        raise PoseCacheError(
            f"Graph cache sample mismatch: expected {expected_sample_id!r}, "
            f"found {sample.sample_id!r}."
        )
    if expected_fingerprint is not None:
        actual = sample.metadata.get("preprocessing_fingerprint")
        if actual != expected_fingerprint:
            raise PoseCacheError(
                f"Graph cache fingerprint mismatch: expected {expected_fingerprint!r}, "
                f"found {actual!r}."
            )
    return sample


def graph_cache_is_current(
    path: str | Path,
    *,
    sample_id: str,
    preprocessing_fingerprint: str,
) -> bool:
    """Return false for missing, corrupt, or stale graph caches."""

    try:
        read_graph_pose_cache(
            path,
            expected_sample_id=sample_id,
            expected_fingerprint=preprocessing_fingerprint,
        )
    except PoseCacheError:
        return False
    return True


def _validate_graph_sample(sample: GraphPoseSample) -> None:
    if sample.features.ndim != 3:
        raise PoseCacheError("Graph features must have shape [T, V, C].")
    target_frames, joints, _ = sample.features.shape
    if sample.joint_mask.shape != (target_frames, joints):
        raise PoseCacheError("joint_mask shape does not match graph features.")
    if sample.observed_mask.shape != (target_frames, joints):
        raise PoseCacheError("observed_mask shape does not match graph features.")
    if sample.frame_mask.shape != (target_frames,):
        raise PoseCacheError("frame_mask shape does not match graph features.")
    if sample.adjacency.shape != (joints, joints):
        raise PoseCacheError("adjacency shape does not match graph joints.")
    if not np.isfinite(sample.features).all() or not np.isfinite(sample.adjacency).all():
        raise PoseCacheError("Graph cache contains non-finite values.")
