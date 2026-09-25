"""Ingest the public cropped VSL MediaPipe keypoint release."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord
from silent_signal.data.manifest import ManifestError


@dataclass(frozen=True, slots=True)
class KaggleVSLBuildResult:
    records: tuple[ManifestRecord, ...]
    labels: tuple[LabelDefinition, ...]
    selection: tuple[dict[str, Any], ...]
    source_counts: dict[str, dict[str, int]]
    split_counts: dict[str, int]


def build_kaggle_vsl_manifest(
    keypoint_root: str | Path,
    *,
    classes: int = 50,
    min_official_train_samples: int = 40,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> KaggleVSLBuildResult:
    """Build a contiguous top-class manifest while preserving the provided test set."""

    root = Path(keypoint_root).resolve()
    if not root.is_dir():
        raise ManifestError(f"MediaPipe keypoint root does not exist: {root}")
    if classes < 0:
        raise ValueError("classes must be zero (all eligible) or positive.")
    if min_official_train_samples < 2:
        raise ValueError("min_official_train_samples must be at least 2.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    by_split = {
        split: _scan_split(root, split)
        for split in ("train", "test")
    }
    train_by_gloss = by_split["train"]
    test_by_gloss = by_split["test"]
    eligible = [
        gloss
        for gloss, paths in train_by_gloss.items()
        if len(paths) >= min_official_train_samples and test_by_gloss.get(gloss)
    ]
    eligible.sort(key=lambda gloss: (-len(train_by_gloss[gloss]), _gloss_key(gloss)))
    if classes:
        eligible = eligible[:classes]
    if not eligible:
        raise ManifestError("No gloss satisfies the class-selection requirements.")
    if classes and len(eligible) != classes:
        raise ManifestError(f"Requested {classes} classes but only {len(eligible)} are eligible.")

    labels = tuple(
        LabelDefinition(
            class_index=class_index,
            gloss_id=f"kvsl:{unicodedata.normalize('NFC', gloss)}",
            gloss_name=unicodedata.normalize("NFC", gloss),
        )
        for class_index, gloss in enumerate(eligible)
    )
    class_by_gloss = {item.gloss_name: item for item in labels}

    records: list[ManifestRecord] = []
    selection: list[dict[str, Any]] = []
    for gloss in eligible:
        label = class_by_gloss[gloss]
        official_train = train_by_gloss[gloss]
        validation_paths = _validation_paths(
            official_train,
            root=root,
            fraction=validation_fraction,
            seed=seed,
        )
        for path in official_train:
            split = "validation" if path in validation_paths else "train"
            records.append(_manifest_record(path, root=root, label=label, split=split))
        for path in test_by_gloss[gloss]:
            records.append(_manifest_record(path, root=root, label=label, split="test"))
        selection.append(
            {
                "class_index": label.class_index,
                "gloss_id": label.gloss_id,
                "gloss_name": label.gloss_name,
                "official_train": len(official_train),
                "train": len(official_train) - len(validation_paths),
                "validation": len(validation_paths),
                "test": len(test_by_gloss[gloss]),
            }
        )

    records.sort(key=lambda item: (item.class_index, str(item.split), item.sample_id))
    sample_ids = [item.sample_id for item in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError("Generated sample identifiers are not unique.")
    split_counts = dict(Counter(str(item.split) for item in records))
    source_counts = {
        gloss: {
            "official_train": len(train_by_gloss[gloss]),
            "official_test": len(test_by_gloss[gloss]),
        }
        for gloss in eligible
    }
    return KaggleVSLBuildResult(
        records=tuple(records),
        labels=labels,
        selection=tuple(selection),
        source_counts=source_counts,
        split_counts=split_counts,
    )


def class_count_summary(result: KaggleVSLBuildResult) -> dict[str, Any]:
    return {
        "clips": len(result.records),
        "classes": len(result.labels),
        "splits": result.split_counts,
        "minimum_per_class": {
            split: min(
                sum(
                    record.split == split and record.class_index == label.class_index
                    for record in result.records
                )
                for label in result.labels
            )
            for split in ("train", "validation", "test")
        },
    }


def _scan_split(root: Path, split: str) -> dict[str, tuple[Path, ...]]:
    split_root = root / split
    if not split_root.is_dir():
        raise ManifestError(f"Expected MediaPipe split directory: {split_root}")
    grouped: dict[str, list[Path]] = {}
    display_by_key: dict[str, str] = {}
    for path in sorted(split_root.rglob("*.npy")):
        relative_parent = path.parent.relative_to(split_root)
        if relative_parent == Path("."):
            raise ManifestError(f"Keypoint file is not inside a gloss directory: {path}")
        display = unicodedata.normalize("NFC", relative_parent.as_posix())
        key = _gloss_key(display)
        previous = display_by_key.setdefault(key, display)
        if previous != display:
            raise ManifestError(
                f"Unicode-equivalent gloss directories disagree: {previous}, {display}"
            )
        grouped.setdefault(display, []).append(path)
    if not grouped:
        raise ManifestError(f"No .npy keypoint files found under {split_root}")
    return {gloss: tuple(paths) for gloss, paths in grouped.items()}


def _validation_paths(
    paths: Sequence[Path],
    *,
    root: Path,
    fraction: float,
    seed: int,
) -> frozenset[Path]:
    ranked = sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root).as_posix()}".encode()
        ).digest(),
    )
    count = max(1, round(len(ranked) * fraction))
    count = min(count, len(ranked) - 1)
    return frozenset(ranked[:count])


def _manifest_record(
    path: Path,
    *,
    root: Path,
    label: LabelDefinition,
    split: str,
) -> ManifestRecord:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    sample_id = f"kvsl-{digest[:20]}"
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=path.stem,
        signer_id="unknown",
        gloss_id=label.gloss_id,
        gloss_name=label.gloss_name,
        class_index=label.class_index,
        view="front",
        video_path=relative,
        is_valid=True,
        split=split,
    )


def _gloss_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def selection_payload(
    result: KaggleVSLBuildResult,
    *,
    dataset_handle: str,
    min_official_train_samples: int,
    validation_fraction: float,
    seed: int,
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "dataset_handle": dataset_handle,
        "source_extractor": "mediapipe_holistic",
        "source_joint_shape": ["T", 76, 3],
        "graph_layout": "mediapipe_upper68_v1",
        "ignored_source_indices": list(range(25, 33)),
        "selection_strategy": "descending_official_train_count_then_gloss",
        "min_official_train_samples": min_official_train_samples,
        "split_protocol": "provided_test_plus_stratified_sample_validation",
        "validation_fraction_of_official_train": validation_fraction,
        "signer_disjoint_validation": False,
        "seed": seed,
        "summary": class_count_summary(result),
        "classes": list(result.selection),
    }
