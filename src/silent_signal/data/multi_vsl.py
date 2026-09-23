"""Prepare an official signer-disjoint Multi-VSL RGB baseline manifest."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPLIT_FILES = {
    "train": "train_1_200_center_ord1.csv",
    "validation": "val_1_200_center_ord1.csv",
    "test": "test_1_200_center_ord1.csv",
}
_SIGNER_PATTERN = re.compile(r"_signer(\d+)_", re.IGNORECASE)


@dataclass(frozen=True)
class MultiVSLPreparationResult:
    """Paths and counts produced by :func:`prepare_multi_vsl`."""

    manifest_path: Path
    selection_path: Path
    summary_path: Path
    selected_classes: tuple[int, ...]
    split_counts: dict[str, int]


def _find_metadata_file(metadata_root: Path, filename: str) -> Path:
    matches = sorted(metadata_root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r} under {metadata_root}, found {len(matches)}."
        )
    return matches[0]


def _video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(video_root.rglob("*.mp4")):
        if path.name in index:
            duplicates.add(path.name)
        index[path.name] = path
    if duplicates:
        first = sorted(duplicates)[0]
        raise RuntimeError(f"Duplicate Multi-VSL video basename found: {first}")
    if not index:
        raise RuntimeError(f"No MP4 videos found under {video_root}.")
    return index


def _read_official_rows(metadata_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, filename in _SPLIT_FILES.items():
        path = _find_metadata_file(metadata_root, filename)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"name", "label", "video_lb_id"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise RuntimeError(f"Unexpected Multi-VSL metadata columns in {path}.")
            for row in reader:
                name = str(row["name"]).strip()
                signer_match = _SIGNER_PATTERN.search(name)
                if signer_match is None:
                    raise RuntimeError(f"Cannot parse signer from Multi-VSL filename: {name}")
                rows.append(
                    {
                        "name": name,
                        "source_class_index": int(row["label"]),
                        "video_lb_id": str(row["video_lb_id"]).strip(),
                        "signer_id": signer_match.group(1),
                        "split": split,
                    }
                )
    if not rows:
        raise RuntimeError("Official Multi-VSL metadata is empty.")
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("The official Multi-VSL metadata contains duplicate filenames.")
    return rows


def _select_classes(rows: list[dict[str, Any]], class_count: int) -> tuple[int, ...]:
    if class_count < 2:
        raise ValueError("class_count must be at least 2.")
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[int(row["source_class_index"])][str(row["split"])] += 1
    eligible = [
        class_index
        for class_index, split_counts in counts.items()
        if all(split_counts[split] > 0 for split in _SPLIT_FILES)
    ]
    ranked = sorted(eligible, key=lambda value: (-counts[value]["train"], value))
    if len(ranked) < class_count:
        raise RuntimeError(
            f"Only {len(ranked)} classes occur in every official split; {class_count} requested."
        )
    return tuple(ranked[:class_count])


def _validate_signer_isolation(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    signers = {
        split: {str(row["signer_id"]) for row in rows if row["split"] == split}
        for split in _SPLIT_FILES
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = signers[left] & signers[right]
        if overlap:
            raise RuntimeError(f"Signer leakage between {left} and {right}: {sorted(overlap)}")
    return {split: sorted(values) for split, values in signers.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_multi_vsl(
    *,
    metadata_root: Path,
    video_root: Path,
    output_root: Path,
    class_count: int = 50,
) -> MultiVSLPreparationResult:
    """Create a compact center-view manifest without changing official signer splits.

    Classes are ranked only by training-set clip count. Validation and test labels
    are used solely to require coverage, never to choose a better-performing subset.
    Every official clip from each selected class must exist locally.
    """

    source_rows = _read_official_rows(metadata_root.resolve())
    selected_classes = _select_classes(source_rows, class_count)
    selected_set = set(selected_classes)
    selected_source_rows = [
        row for row in source_rows if int(row["source_class_index"]) in selected_set
    ]
    signer_splits = _validate_signer_isolation(selected_source_rows)
    videos = _video_index(video_root.resolve())
    missing = [row["name"] for row in selected_source_rows if row["name"] not in videos]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} official videos for the selected classes are missing; "
            f"first: {missing[0]}"
        )

    class_to_rank = {class_index: rank for rank, class_index in enumerate(selected_classes)}
    manifest_rows: list[dict[str, Any]] = []
    for row in selected_source_rows:
        source_class_index = int(row["source_class_index"])
        path = videos[str(row["name"])]
        manifest_rows.append(
            {
                "sample_id": f"multi-vsl-{row['video_lb_id']}-{path.stem}",
                "video_path": path.relative_to(video_root.resolve()).as_posix(),
                "gloss_name": f"VSL_{source_class_index + 1:03d}",
                "class_index": source_class_index,
                "split": row["split"],
                "signer_id": row["signer_id"],
                "view": "center",
                "source_video_label_id": row["video_lb_id"],
            }
        )
    manifest_rows.sort(
        key=lambda row: (str(row["split"]), int(row["class_index"]), row["sample_id"])
    )

    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    class_split_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in manifest_rows:
        class_split_counts[int(row["class_index"])][str(row["split"])] += 1

    output_root = output_root.resolve()
    manifest_path = output_root / "multi_vsl_200_center_top50.csv"
    selection_path = output_root / "multi_vsl_200_center_top50_selection.json"
    summary_path = output_root / "multi_vsl_200_center_top50_summary.json"
    _write_csv(manifest_path, manifest_rows)
    classes = [
        {
            "rank": rank + 1,
            "subset_class_index": class_index,
            "gloss_name": f"VSL_{class_index + 1:03d}",
            "source_label": class_index,
            "counts": dict(class_split_counts[class_index]),
        }
        for class_index, rank in sorted(class_to_rank.items(), key=lambda item: item[1])
    ]
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "dataset_name": "Multi-VSL M-VSL200",
            "study_stage": f"center-view {class_count}-class RGB baseline",
            "selection": "classes ranked by official training clip count only",
            "split_policy": "official signer-disjoint train/validation/test split; never re-split",
            "class_name_note": (
                "The public metadata contains numeric labels but no Vietnamese gloss text; "
                "VSL_NNN is a stable display label for source label NNN-1."
            ),
            "classes": classes,
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "dataset_name": "Multi-VSL M-VSL200",
            "view": "center",
            "class_count": class_count,
            "selected_source_labels": list(selected_classes),
            "clips": dict(split_counts),
            "signers": signer_splits,
            "signer_isolation": "passed",
            "video_root": str(video_root.resolve()),
            "manifest": str(manifest_path),
            "selection": str(selection_path),
        },
    )
    return MultiVSLPreparationResult(
        manifest_path=manifest_path,
        selection_path=selection_path,
        summary_path=summary_path,
        selected_classes=selected_classes,
        split_counts=dict(split_counts),
    )
