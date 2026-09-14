"""CLI for selecting frequent ASL Citizen classes with ASL-LEX 2.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from silent_signal.contracts import LabelDefinition, ManifestRecord
from silent_signal.data.manifest import ManifestError, read_manifest, write_labels, write_manifest
from silent_signal.data.subsets import read_asl_lex_frequencies, select_asl_citizen_top_classes

_DEFAULT_MANIFEST = Path("data/manifests/asl_citizen.csv")
_DEFAULT_OUTPUT_ROOT = Path("data/subsets")


def build_parser() -> argparse.ArgumentParser:
    """Build the selection CLI parser."""

    parser = argparse.ArgumentParser(
        prog="ss-select-asl-subset",
        description=("Select ASL Citizen classes by ASL-LEX 2.0 subjective sign-frequency rating."),
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--asl-lex-csv",
        type=Path,
        required=True,
        help="Official ASL-LEX 2.0 signdata.csv file.",
    )
    parser.add_argument("--classes", type=int, default=200)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--project-commit",
        help="Optional Git commit recorded in the selection report for provenance.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Select and serialize the requested subset."""

    args = build_parser().parse_args(argv)
    try:
        source_manifest = args.manifest.resolve()
        asl_lex_csv = args.asl_lex_csv.resolve()
        frequencies = read_asl_lex_frequencies(asl_lex_csv)
        result = select_asl_citizen_top_classes(
            read_manifest(source_manifest), frequencies, class_count=args.classes
        )

        subset_name = f"asl_citizen_asllex_top{args.classes}"
        destination = args.output_root.resolve() / subset_name
        manifest_csv = destination / "manifest.csv"
        manifest_parquet = destination / "manifest.parquet"
        labels_json = destination / "labels.json"
        report_json = destination / "selection_report.json"

        _write_manifest_atomic(result.records, manifest_csv)
        _write_manifest_atomic(result.records, manifest_parquet)
        _write_labels_atomic(result.labels, labels_json, dataset=subset_name)
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": "asl_citizen",
            "subset": subset_name,
            "project_commit": args.project_commit,
            "criterion": {
                "source": "ASL-LEX 2.0 signdata.csv",
                "source_url": "https://asl-lex.org/download.html",
                "paper_doi": "10.1093/deafed/enaa038",
                "license": "CC BY-NC 4.0 (database and visualization)",
                "column": "SignFrequency(M)",
                "meaning": "mean subjective frequency in everyday ASL conversation",
                "scale": [1, 7],
                "sort": [
                    "sign_frequency_mean descending",
                    "gloss_name case-insensitive ascending",
                    "source_class_index ascending",
                ],
            },
            "sources": {
                "manifest": {
                    "path": str(source_manifest),
                    "sha256": _sha256_file(source_manifest),
                },
                "asl_lex_csv": {
                    "path": str(asl_lex_csv),
                    "sha256": _sha256_file(asl_lex_csv),
                },
            },
            "outputs": {
                "manifest_csv": str(manifest_csv),
                "manifest_parquet": str(manifest_parquet),
                "labels_json": str(labels_json),
            },
            **result.report(),
        }
        _write_json_atomic(report_json, report)
        _print_json(
            {
                "subset": subset_name,
                "classes": len(result.classes),
                "clips": len(result.records),
                "output_root": str(destination),
                "report": str(report_json),
            }
        )
        return 0
    except (ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    """Console-script entry point."""

    raise SystemExit(main())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest_atomic(records: Sequence[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        write_manifest(records, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_labels_atomic(labels: Sequence[LabelDefinition], path: Path, *, dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_labels(labels, temporary, dataset=dataset)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
