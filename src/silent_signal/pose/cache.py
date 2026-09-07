"""Atomic, versioned storage for raw RTMPose whole-body predictions."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from silent_signal.pose.interface import RawPoseSequence

POSE_CACHE_SCHEMA_VERSION = 1
_SOFTWARE_PACKAGES = (
    "mmpose",
    "mmdet",
    "mmcv",
    "mmengine",
    "torch",
    "torchvision",
    "opencv-python",
    "opencv-python-headless",
    "numpy",
)


class PoseCacheError(RuntimeError):
    """Raised when a pose cache is corrupt, stale, or cannot be written."""


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the full SHA-256 digest of a local artifact."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise PoseCacheError(f"Cannot hash artifact {source}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping independent of key insertion order."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PoseCacheError(f"Fingerprint payload is not JSON serializable: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def installed_software_versions(
    packages: Sequence[str] = _SOFTWARE_PACKAGES,
) -> dict[str, str]:
    """Collect installed runtime versions without importing heavyweight modules."""

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def runtime_environment() -> dict[str, Any]:
    """Capture Python, package, PyTorch CUDA, and GPU runtime provenance."""

    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": installed_software_versions(),
        "cuda": {
            "torch_compiled_version": None,
            "available": False,
            "device_count": 0,
            "devices": [],
        },
    }
    if "torch" not in environment["packages"]:
        return environment
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
        environment["cuda"] = {
            "torch_compiled_version": getattr(torch.version, "cuda", None),
            "available": available,
            "device_count": device_count,
            "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        }
    except Exception as exc:  # Provenance collection must not hide artifact verification.
        environment["cuda"]["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return environment


def pose_cache_path(root: str | Path, sample_id: str) -> Path:
    """Map an untrusted sample ID to a traversal-safe, evenly distributed path."""

    if not sample_id:
        raise PoseCacheError("sample_id must not be empty.")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return Path(root) / digest[:2] / f"{digest}.npz"


def write_pose_cache(
    path: str | Path,
    sequence: RawPoseSequence,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write one raw sequence without pickle-based serialization."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise PoseCacheError(f"Pose cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": POSE_CACHE_SCHEMA_VERSION,
        "sample_id": sequence.sample_id,
        "source_video": sequence.source_video,
        "frame_size_hw": list(sequence.frame_size_hw),
        "metadata": sequence.metadata,
    }
    try:
        metadata_json = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PoseCacheError(f"Pose metadata is not JSON serializable: {exc}") from exc

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
                frame_indices=sequence.frame_indices,
                timestamps_seconds=sequence.timestamps_seconds,
                keypoints_xy=sequence.keypoints_xy,
                keypoint_scores=sequence.keypoint_scores,
                bboxes_xyxy=sequence.bboxes_xyxy,
                bbox_scores=sequence.bbox_scores,
                person_detected=sequence.person_detected,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except (OSError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PoseCacheError(f"Cannot write pose cache {destination}: {exc}") from exc


def read_pose_cache(
    path: str | Path,
    *,
    expected_sample_id: str | None = None,
    expected_fingerprint: str | None = None,
    expected_video_sha256: str | None = None,
) -> RawPoseSequence:
    """Read and validate one cache, optionally enforcing provenance identity."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "frame_indices",
                "timestamps_seconds",
                "keypoints_xy",
                "keypoint_scores",
                "bboxes_xyxy",
                "bbox_scores",
                "person_detected",
            }
            missing = required.difference(archive.files)
            if missing:
                raise PoseCacheError(f"Pose cache is missing arrays: {sorted(missing)}")
            envelope = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if envelope.get("schema_version") != POSE_CACHE_SCHEMA_VERSION:
                raise PoseCacheError(
                    f"Unsupported pose cache schema_version: {envelope.get('schema_version')!r}."
                )
            sequence = RawPoseSequence(
                sample_id=str(envelope["sample_id"]),
                source_video=str(envelope["source_video"]),
                frame_indices=archive["frame_indices"],
                timestamps_seconds=archive["timestamps_seconds"],
                frame_size_hw=tuple(int(value) for value in envelope["frame_size_hw"]),
                keypoints_xy=archive["keypoints_xy"],
                keypoint_scores=archive["keypoint_scores"],
                bboxes_xyxy=archive["bboxes_xyxy"],
                bbox_scores=archive["bbox_scores"],
                person_detected=archive["person_detected"],
                metadata=dict(envelope.get("metadata", {})),
            )
    except PoseCacheError:
        raise
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PoseCacheError(f"Cannot read pose cache {source}: {exc}") from exc

    if expected_sample_id is not None and sequence.sample_id != expected_sample_id:
        raise PoseCacheError(
            f"Cache sample mismatch: expected {expected_sample_id!r}, found {sequence.sample_id!r}."
        )
    _check_metadata_identity(sequence, "extractor_fingerprint", expected_fingerprint)
    _check_metadata_identity(sequence, "video_sha256", expected_video_sha256)
    return sequence


def cache_is_current(
    path: str | Path,
    *,
    sample_id: str,
    extractor_fingerprint: str,
    video_sha256: str,
) -> bool:
    """Return false for missing, corrupt, or provenance-mismatched cache files."""

    try:
        read_pose_cache(
            path,
            expected_sample_id=sample_id,
            expected_fingerprint=extractor_fingerprint,
            expected_video_sha256=video_sha256,
        )
    except PoseCacheError:
        return False
    return True


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a human-readable report atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PoseCacheError(f"Cannot write JSON report {destination}: {exc}") from exc


def _check_metadata_identity(
    sequence: RawPoseSequence,
    key: str,
    expected: str | None,
) -> None:
    if expected is None:
        return
    actual = sequence.metadata.get(key)
    if actual != expected:
        raise PoseCacheError(f"Cache {key} mismatch: expected {expected!r}, found {actual!r}.")
