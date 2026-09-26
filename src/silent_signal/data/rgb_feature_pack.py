"""Portable, pickle-free storage for frozen RGB temporal features."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

RGB_FEATURE_PACK_SCHEMA_VERSION = 1


class RGBFeaturePackError(RuntimeError):
    """Raised when an RGB feature pack is corrupt or has the wrong identity."""


@dataclass(frozen=True, slots=True)
class RGBFeaturePack:
    """Frozen per-video temporal tokens indexed by stable manifest sample IDs."""

    sample_ids: tuple[str, ...]
    features: NDArray[np.float16]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [samples, temporal_tokens, channels].")
        if self.features.shape[0] != len(self.sample_ids):
            raise ValueError("The number of feature rows must equal the number of sample IDs.")
        if self.features.dtype != np.float16:
            raise ValueError("RGB features must be stored as float16.")
        if not self.sample_ids or any(not sample_id for sample_id in self.sample_ids):
            raise ValueError("sample_ids must be non-empty strings.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique.")
        if not np.isfinite(self.features).all():
            raise ValueError("RGB features contain non-finite values.")

    @property
    def index(self) -> dict[str, int]:
        """Map each stable sample ID to its row in ``features``."""

        return {sample_id: position for position, sample_id in enumerate(self.sample_ids)}


def write_rgb_feature_pack(
    path: str | Path,
    sample_ids: Sequence[str],
    features: NDArray[np.generic],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write frozen RGB tokens without pickle-based serialization."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise RGBFeaturePackError(f"RGB feature pack already exists: {destination}")
    normalized_ids = tuple(str(sample_id) for sample_id in sample_ids)
    normalized_features = np.asarray(features, dtype=np.float16)
    envelope = {
        "schema_version": RGB_FEATURE_PACK_SCHEMA_VERSION,
        "metadata": dict(metadata),
    }
    try:
        payload = RGBFeaturePack(normalized_ids, normalized_features, dict(metadata))
        encoded = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RGBFeaturePackError(f"Invalid RGB feature pack: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                sample_ids=np.asarray(payload.sample_ids, dtype=np.str_),
                features=payload.features,
                metadata_json=np.frombuffer(encoded, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, ValueError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise RGBFeaturePackError(f"Cannot write RGB feature pack {destination}: {exc}") from exc


def read_rgb_feature_pack(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> RGBFeaturePack:
    """Read and validate an RGB feature pack and optional extraction identity."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {"sample_ids", "features", "metadata_json"}
            missing = required.difference(archive.files)
            if missing:
                raise RGBFeaturePackError(
                    f"RGB feature pack is missing arrays: {sorted(missing)}"
                )
            envelope = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if envelope.get("schema_version") != RGB_FEATURE_PACK_SCHEMA_VERSION:
                raise RGBFeaturePackError(
                    "Unsupported RGB feature pack schema_version: "
                    f"{envelope.get('schema_version')!r}."
                )
            metadata_payload = envelope.get("metadata", {})
            if not isinstance(metadata_payload, dict):
                raise RGBFeaturePackError("RGB feature pack metadata must be a JSON object.")
            metadata = dict(metadata_payload)
            pack = RGBFeaturePack(
                sample_ids=tuple(str(value) for value in archive["sample_ids"]),
                features=archive["features"].astype(np.float16, copy=False),
                metadata=metadata,
            )
    except RGBFeaturePackError:
        raise
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RGBFeaturePackError(f"Cannot read RGB feature pack {source}: {exc}") from exc

    if expected_fingerprint is not None:
        actual = pack.metadata.get("extraction_fingerprint")
        if actual != expected_fingerprint:
            raise RGBFeaturePackError(
                "RGB feature extraction fingerprint mismatch: "
                f"expected {expected_fingerprint!r}, found {actual!r}."
            )
    return pack
