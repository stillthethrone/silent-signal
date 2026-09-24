"""Prepare official signer-disjoint Multi-VSL manifests for the RGB and pose pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from silent_signal.configuration import ExpectedConfig
from silent_signal.contracts import LabelDefinition, ManifestRecord, SplitDefinition
from silent_signal.data.manifest import write_labels, write_manifest
from silent_signal.data.splits import write_split_definition
from silent_signal.data.validation import (
    ValidationResult,
    validate_manifest,
    write_validation_report,
)

_SPLIT_FILES = {
    "train": "train_1_200_center_ord1.csv",
    "validation": "val_1_200_center_ord1.csv",
    "test": "test_1_200_center_ord1.csv",
}
_SIGNER_PATTERN = re.compile(r"_signer(\d+)_", re.IGNORECASE)
_THREE_VIEW_FILES = {
    "train": "train_1_200_three_view_ord1.csv",
    "validation": "val_1_200_three_view_ord1.csv",
    "test": "test_1_200_three_view_ord1.csv",
}
_VIEW = "center"
_VIEW_MODES = {"center": (_VIEW,), "three_view": ("center", "left", "right")}
_DATASET_NAME = "Multi-VSL M-VSL200"
_SELECTION_RULE = "classes ranked by official training clip count only"
_SPLIT_POLICY = "official signer-disjoint train/validation/test split; never re-split"
_CLASS_NAME_NOTE = (
    "The public metadata contains numeric labels but no Vietnamese gloss text; "
    "VSL_NNN is a stable display label for source label NNN-1."
)


@dataclass(frozen=True)
class MultiVSLPreparationResult:
    """Paths and counts produced by :func:`prepare_multi_vsl`."""

    manifest_path: Path
    selection_path: Path
    summary_path: Path
    selected_classes: tuple[int, ...]
    split_counts: dict[str, int]


def _sample_id(row: dict[str, Any]) -> str:
    """Shared RGB/pose sample key, so both branches can later be joined per clip."""

    return f"multi-vsl-{row['video_lb_id']}-{Path(str(row['name'])).stem}"


def _gloss_name(source_label: int) -> str:
    return f"VSL_{source_label + 1:03d}"


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
    """Rank classes by training clips; ``class_count=0`` keeps every eligible class."""

    if class_count == 1 or class_count < 0:
        raise ValueError("class_count must be 0 (all classes) or at least 2.")
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[int(row["source_class_index"])][str(row["split"])] += 1
    eligible = [
        class_index
        for class_index, split_counts in counts.items()
        if all(split_counts[split] > 0 for split in _SPLIT_FILES)
    ]
    ranked = sorted(eligible, key=lambda value: (-counts[value]["train"], value))
    if class_count == 0:
        class_count = len(ranked)
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
                "sample_id": _sample_id(row),
                "video_path": path.relative_to(video_root.resolve()).as_posix(),
                "gloss_name": _gloss_name(source_class_index),
                "class_index": source_class_index,
                "split": row["split"],
                "signer_id": row["signer_id"],
                "view": _VIEW,
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
            "gloss_name": _gloss_name(class_index),
            "source_label": class_index,
            "counts": dict(class_split_counts[class_index]),
        }
        for class_index, rank in sorted(class_to_rank.items(), key=lambda item: item[1])
    ]
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "dataset_name": _DATASET_NAME,
            "study_stage": f"center-view {class_count}-class RGB baseline",
            "selection": _SELECTION_RULE,
            "split_policy": _SPLIT_POLICY,
            "class_name_note": _CLASS_NAME_NOTE,
            "classes": classes,
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "dataset_name": _DATASET_NAME,
            "view": _VIEW,
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


@dataclass(frozen=True)
class MultiVSLSelection:
    """Official rows of the ranked classes, one row per clip, before any video is read."""

    rows: tuple[dict[str, Any], ...]
    classes: tuple[int, ...]
    views: tuple[str, ...]
    signer_splits: dict[str, list[str]]
    source_files: dict[str, dict[str, str]]

    @property
    def required_videos(self) -> tuple[str, ...]:
        """Official filenames the selection needs, sorted for stable fetching."""

        return tuple(sorted(str(row["name"]) for row in self.rows))


@dataclass(frozen=True)
class MultiVSLPoseDataset:
    """Artifacts written by :func:`build_multi_vsl_pose_dataset`."""

    manifest_csv: Path
    manifest_parquet: Path
    labels_path: Path
    split_path: Path
    selection_path: Path
    report_path: Path
    validation: ValidationResult


def select_multi_vsl(
    metadata_root: Path, class_count: int = 50, *, view_mode: str = "center"
) -> MultiVSLSelection:
    """Rank classes by center training clips and keep their official split rows unchanged.

    The class ranking always uses the center-view files, so every view mode (and the RGB
    baseline) shares one class list. ``view_mode="three_view"`` reads the authors'
    synchronized center/left/right triplets for those classes. ``class_count=0`` keeps
    every class present in all three official splits.
    """

    if view_mode not in _VIEW_MODES:
        raise ValueError(f"view_mode must be one of {sorted(_VIEW_MODES)}.")
    root = metadata_root.resolve()
    center_rows = _read_official_rows(root)
    classes = _select_classes(center_rows, class_count)
    selected = set(classes)
    files = dict(_SPLIT_FILES)
    if view_mode == "center":
        rows = tuple(
            {**row, "view": _VIEW, "instance_key": None}
            for row in center_rows
            if int(row["source_class_index"]) in selected
        )
    else:
        center_by_name = {str(row["name"]): row for row in center_rows}
        rows = tuple(
            row
            for row in _read_three_view_rows(root, center_by_name)
            if int(row["source_class_index"]) in selected
        )
        files.update({f"{split}_three_view": name for split, name in _THREE_VIEW_FILES.items()})
    source_files = {}
    for key, filename in files.items():
        path = _find_metadata_file(root, filename)
        source_files[key] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return MultiVSLSelection(
        rows=rows,
        classes=classes,
        views=_VIEW_MODES[view_mode],
        signer_splits=_validate_signer_isolation(list(rows)),
        source_files=source_files,
    )


def multi_vsl_pose_records(selection: MultiVSLSelection) -> tuple[ManifestRecord, ...]:
    """Map selected rows to the shared manifest contract read by ``ss-extract-pose``.

    Class indices follow the training-count rank (0 = most clips), matching the demo
    class order used by the RGB baseline. Center clips keep the RGB baseline sample ID;
    the three views of one recording share ``instance_id``. Videos are addressed by their
    official filename relative to one flat video directory.
    """

    class_index = {label: rank for rank, label in enumerate(selection.classes)}
    records = []
    for row in selection.rows:
        label = int(row["source_class_index"])
        sample_id = _sample_id(row)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                instance_id=str(row["instance_key"] or sample_id),
                video_id=str(row["name"]),
                signer_id=str(row["signer_id"]),
                gloss_id=str(label),
                gloss_name=_gloss_name(label),
                class_index=class_index[label],
                view=str(row["view"]),
                video_path=str(row["name"]),
                split=str(row["split"]),
            )
        )
    return tuple(sorted(records, key=lambda item: (item.class_index, item.sample_id)))


def build_multi_vsl_pose_dataset(
    *,
    metadata_root: Path,
    video_root: Path,
    output_root: Path,
    class_count: int = 50,
    view_mode: str = "center",
    level: str = "metadata",
    workers: int = 4,
) -> MultiVSLPoseDataset:
    """Validate local videos and write the manifest, labels and split used for pose work."""

    selection = select_multi_vsl(metadata_root, class_count, view_mode=view_mode)
    records = multi_vsl_pose_records(selection)
    validation = validate_manifest(
        records,
        dataset_root=video_root,
        expected=ExpectedConfig(views_per_instance=len(selection.views)),
        expected_views=selection.views,
        level=level,
        workers=workers,
    )
    records = validation.records
    labels = tuple(
        LabelDefinition(class_index=rank, gloss_id=str(label), gloss_name=_gloss_name(label))
        for rank, label in enumerate(selection.classes)
    )
    splits = tuple(_SPLIT_FILES)
    clip_counts = {split: sum(item.split == split for item in records) for split in splits}
    instance_counts = {
        split: len({item.instance_id for item in records if item.split == split})
        for split in splits
    }
    split_definition = SplitDefinition(
        seed=None,
        target_ratios={},
        signer_ids={split: tuple(selection.signer_splits[split]) for split in splits},
        signer_counts={split: len(selection.signer_splits[split]) for split in splits},
        instance_counts=instance_counts,
        clip_counts=clip_counts,
        gloss_counts={
            split: len({item.class_index for item in records if item.split == split})
            for split in splits
        },
        score=None,
        strategy="official",
    )

    output_root = output_root.resolve()
    artifacts = MultiVSLPoseDataset(
        manifest_csv=output_root / "manifest.csv",
        manifest_parquet=output_root / "manifest.parquet",
        labels_path=output_root / "labels.json",
        split_path=output_root / "split.json",
        selection_path=output_root / "selection.json",
        report_path=output_root / "validation_report.json",
        validation=validation,
    )
    write_manifest(records, artifacts.manifest_csv)
    write_manifest(records, artifacts.manifest_parquet)
    write_labels(labels, artifacts.labels_path, dataset="multi_vsl_m_vsl200")
    write_split_definition(split_definition, artifacts.split_path)
    write_validation_report(validation, artifacts.report_path)
    instances: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        instances[record.class_index][str(record.split)].add(record.instance_id)
    _write_json(
        artifacts.selection_path,
        {
            "schema_version": 1,
            "dataset_name": _DATASET_NAME,
            "view_mode": view_mode,
            "views": list(selection.views),
            "class_count": len(selection.classes),
            "selection": _SELECTION_RULE,
            "split_policy": _SPLIT_POLICY,
            "class_name_note": _CLASS_NAME_NOTE,
            "source_files": selection.source_files,
            "clips": clip_counts,
            "instances": instance_counts,
            "signers": selection.signer_splits,
            "classes": [
                {
                    "rank": rank + 1,
                    "class_index": rank,
                    "source_label": label,
                    "gloss_name": _gloss_name(label),
                    "instances": {split: len(instances[rank][split]) for split in splits},
                }
                for rank, label in enumerate(selection.classes)
            ],
        },
    )
    return artifacts


def _read_three_view_rows(
    metadata_root: Path, center_by_name: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand official triplets into one row per view, reusing center labels and IDs."""

    rows: list[dict[str, Any]] = []
    for split, filename in _THREE_VIEW_FILES.items():
        path = _find_metadata_file(metadata_root, filename)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {*_VIEW_MODES["three_view"], "label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise RuntimeError(f"Unexpected Multi-VSL three-view columns in {path}.")
            for triplet in reader:
                center = center_by_name.get(str(triplet["center"]).strip())
                if center is None or center["split"] != split:
                    raise RuntimeError(
                        f"Three-view center clip is not in the official {split} center list: "
                        f"{triplet['center']}"
                    )
                if int(triplet["label"]) != int(center["source_class_index"]):
                    raise RuntimeError(f"Three-view label disagrees for {triplet['center']}.")
                for view in _VIEW_MODES["three_view"]:
                    name = str(triplet[view]).strip()
                    signer = _SIGNER_PATTERN.search(name)
                    if f"_{view}_" not in name or signer is None:
                        raise RuntimeError(f"Unexpected {view} filename in {path}: {name}")
                    if signer.group(1) != center["signer_id"]:
                        raise RuntimeError(f"Three-view signers disagree for {center['name']}.")
                    rows.append(
                        {
                            "name": name,
                            "source_class_index": center["source_class_index"],
                            "video_lb_id": center["video_lb_id"],
                            "signer_id": center["signer_id"],
                            "split": split,
                            "view": view,
                            "instance_key": f"multi-vsl-{center['video_lb_id']}",
                        }
                    )
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("The official Multi-VSL three-view metadata repeats a filename.")
    return rows
