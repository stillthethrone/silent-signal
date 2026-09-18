"""Create a frozen 50-class ASL subset from the ranked top-200 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord
from silent_signal.data.manifest import ManifestError, read_manifest, write_labels, write_manifest
from silent_signal.data.ranked_subset import select_ranked_asl_subset
from silent_signal.pose.cache import sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-select-asl-ranked-subset",
        description="Freeze a smaller ASL subset while preserving official split membership.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=50)
    parser.add_argument("--project-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_manifest = args.manifest.resolve()
        source_selection = args.selection_report.resolve()
        selection_payload = json.loads(source_selection.read_text(encoding="utf-8"))
        if not isinstance(selection_payload, dict):
            raise ManifestError("Source selection report must be a JSON object.")
        result = select_ranked_asl_subset(
            read_manifest(source_manifest),
            selection_payload,
            class_count=args.classes,
        )
        destination = args.output_root.resolve()
        manifests_root = destination / "manifests"
        manifest_csv = destination / "manifest.csv"
        manifest_parquet = destination / "manifest.parquet"
        labels_json = destination / "labels.json"
        selected_words = destination / f"selected_{args.classes}_words.json"

        _write_manifest_atomic(result.records, manifest_csv)
        _write_manifest_atomic(result.records, manifest_parquet)
        _write_manifest_atomic(result.records, manifests_root / "all.csv")
        for split in ("train", "validation", "test"):
            _write_manifest_atomic(
                tuple(record for record in result.records if record.split == split),
                manifests_root / f"{split}.csv",
            )
        _write_labels_atomic(
            result.labels,
            labels_json,
            dataset=f"asl_citizen_asllex_top{args.classes}",
        )

        report: dict[str, Any] = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": "asl_citizen",
            "subset": f"asl_citizen_asllex_top{args.classes}",
            "project_commit": args.project_commit,
            "selection": (
                f"highest-ranked {args.classes} classes from the frozen source report "
                "with clips in every official split"
            ),
            "split_policy": (
                "official ASL Citizen train/validation/test assignments; never re-split"
            ),
            "sources": {
                "manifest": {
                    "path": str(source_manifest),
                    "sha256": sha256_file(source_manifest),
                },
                "selection_report": {
                    "path": str(source_selection),
                    "sha256": sha256_file(source_selection),
                },
            },
            "outputs": {
                "manifest_csv": str(manifest_csv),
                "manifest_parquet": str(manifest_parquet),
                "labels_json": str(labels_json),
                "split_manifests": {
                    split: str(manifests_root / f"{split}.csv")
                    for split in ("train", "validation", "test")
                },
            },
            **result.report(),
        }
        write_json_atomic(selected_words, report)
        print(
            json.dumps(
                {
                    "output_root": str(destination),
                    "manifest": str(manifest_csv),
                    "manifest_sha256": sha256_file(manifest_csv),
                    "classes": len(result.classes),
                    "clips": len(result.records),
                    "split_counts": result.split_counts,
                    "selected_words": str(selected_words),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (json.JSONDecodeError, ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _write_manifest_atomic(records: Sequence[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        write_manifest(records, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_labels_atomic(
    labels: Sequence[LabelDefinition], path: Path, *, dataset: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_labels(labels, temporary, dataset=dataset)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    entrypoint()
