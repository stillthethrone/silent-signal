"""One pickle-free NPZ holding many variable-length keypoint sequences."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PackedKeypoints:
    """Sequences concatenated along time; sequence ``i`` is ``frames[offsets[i]:offsets[i+1]]``."""

    frames: NDArray[np.float32]
    offsets: NDArray[np.int64]
    sample_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.offsets.ndim != 1 or len(self.offsets) != len(self.sample_ids) + 1:
            raise ValueError("offsets must have one entry more than sample_ids.")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.frames):
            raise ValueError("offsets must start at 0 and end at the frame count.")
        if np.any(np.diff(self.offsets) < 1):
            raise ValueError("Every packed sequence needs at least one frame.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Packed sample_ids must be unique.")

    def sequence(self, index: int) -> NDArray[np.float32]:
        return self.frames[self.offsets[index] : self.offsets[index + 1]]

    @property
    def index(self) -> dict[str, int]:
        return {sample_id: position for position, sample_id in enumerate(self.sample_ids)}


def write_packed_keypoints(
    path: Path,
    sequences: Mapping[str, NDArray[np.float32]],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically write sequences (sorted by sample ID) and JSON metadata."""

    if not sequences:
        raise ValueError("Nothing to pack.")
    sample_ids = sorted(sequences)
    lengths = [len(sequences[sample_id]) for sample_id in sample_ids]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    frames = np.concatenate([sequences[sample_id] for sample_id in sample_ids]).astype(np.float32)
    encoded = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.savez_compressed(
            handle,
            frames=frames,
            offsets=offsets,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            metadata_json=np.frombuffer(encoded, dtype=np.uint8),
        )
        temporary = Path(handle.name)
    temporary.replace(path)


def read_packed_keypoints(path: Path) -> PackedKeypoints:
    with np.load(path, allow_pickle=False) as archive:
        return PackedKeypoints(
            frames=archive["frames"].astype(np.float32, copy=False),
            offsets=archive["offsets"].astype(np.int64, copy=False),
            sample_ids=tuple(str(value) for value in archive["sample_ids"]),
            metadata=json.loads(archive["metadata_json"].tobytes().decode("utf-8")),
        )


def select_sequences(
    packed: PackedKeypoints, sample_ids: Sequence[str]
) -> list[NDArray[np.float32]]:
    index = packed.index
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise KeyError(f"{len(missing)} samples are not packed; first: {missing[0]}")
    return [packed.sequence(index[sample_id]) for sample_id in sample_ids]
