"""Atomic PyTorch checkpoint publication."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def write_torch_checkpoint_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write a checkpoint atomically so interrupted Drive writes are not accepted."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {destination}")
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
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, RuntimeError, TypeError, ValueError):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
