"""Fetch manifest-selected canonical cropped RGB clips from one Kaggle ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from silent_signal.cli.fetch_kaggle_vsl_mediapipe import (
    DEFAULT_DATASET,
    DEFAULT_VERSION,
    _credentials,
    kaggle_archive_resolver,
)
from silent_signal.data.kaggle_vsl_rgb import (
    CANONICAL_RGB_DIRECTORY,
    RGBPair,
    extraction_targets,
    locate_canonical_rgb,
    pair_manifest_rgb,
    rgb_manifest_records,
)
from silent_signal.data.manifest import ManifestError, read_manifest, write_manifest
from silent_signal.data.remote_zip import (
    RemoteArchive,
    RemoteZipError,
    extract_members_coalesced,
    inspect_directory,
)
from silent_signal.pose.cache import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-fetch-kaggle-vsl-rgb",
        description=(
            "Open one Kaggle archive, pair the canonical cropped RGB clips to an exact "
            "pose manifest, and extract only those MP4 members."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--rgb-manifest",
        type=Path,
        help="Optional output manifest with the same sample IDs and local MP4 paths.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent coalesced Range requests (default: 2).",
    )
    parser.add_argument(
        "--max-range-mib",
        type=int,
        default=128,
        help="Maximum compressed ZIP bytes fetched by one Range request (default: 128 MiB).",
    )
    parser.add_argument(
        "--max-gap-mib",
        type=int,
        default=4,
        help="Maximum unselected gap merged into one Range request (default: 4 MiB).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help="Kaggle dataset version, or zero to use the latest public version.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--download-url", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive.")
        if args.max_range_mib < 1:
            raise ValueError("--max-range-mib must be positive.")
        if args.max_gap_mib < 0:
            raise ValueError("--max-gap-mib must be non-negative.")
        manifest_path = args.manifest.resolve()
        records = tuple(read_manifest(manifest_path))
        if any(record.split not in {"train", "validation", "test"} for record in records):
            raise ManifestError(
                "Every RGB manifest row must have a train, validation, or test split."
            )

        auth_headers, auth_mode = _credentials()
        archive = RemoteArchive(
            kaggle_archive_resolver(
                auth_headers,
                dataset=args.dataset,
                version=args.version,
                download_url=args.download_url,
            )
        )
        print(
            f"[kaggle] archive={archive.identity.size / 1024**3:.2f} GiB "
            f"| auth={auth_mode}",
            flush=True,
        )
        directory = inspect_directory(archive)
        archive_members = directory.members
        print(f"[kaggle] central directory: {len(archive_members):,} entries", flush=True)

        canonical = locate_canonical_rgb(archive_members)
        pairs = pair_manifest_rgb(records, canonical)
        targets = extraction_targets(pairs, args.output_root)
        selected_bytes = sum(pair.member.file_size for pair in pairs)
        inventory_sha256 = _inventory_sha256(pairs)
        class_count = len({record.class_index for record in records})
        print(
            f"[selection] {class_count} classes | {len(pairs):,} canonical RGB clips "
            f"| {selected_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        extraction = extract_members_coalesced(
            archive,
            targets,
            directory,
            workers=args.workers,
            reserve_bytes=1024**3,
            max_span_bytes=args.max_range_mib * 1024**2,
            max_gap_bytes=args.max_gap_mib * 1024**2,
        )

        rgb_manifest_path = args.rgb_manifest.resolve() if args.rgb_manifest else None
        if rgb_manifest_path is not None:
            write_manifest(rgb_manifest_records(records, pairs), rgb_manifest_path)
        report = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "requested_version": args.version or "latest",
            "archive_size": archive.identity.size,
            "archive_etag": archive.identity.etag,
            "auth_mode": auth_mode,
            "source_directory": CANONICAL_RGB_DIRECTORY,
            "selection_strategy": "exact_manifest_canonical_pairs",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rgb_manifest": str(rgb_manifest_path) if rgb_manifest_path else None,
            "rgb_manifest_sha256": (
                sha256_file(rgb_manifest_path) if rgb_manifest_path is not None else None
            ),
            "classes": class_count,
            "files": len(pairs),
            "selected_bytes": selected_bytes,
            "inventory_sha256": inventory_sha256,
            "coalescing": {
                "workers": args.workers,
                "max_range_mib": args.max_range_mib,
                "max_gap_mib": args.max_gap_mib,
            },
            "experimental_split_counts": dict(
                sorted(Counter(str(record.split) for record in records).items())
            ),
            "physical_split_counts": dict(
                sorted(Counter(pair.source_split for pair in pairs).items())
            ),
            "extraction": extraction,
            "output_root": str(args.output_root.resolve()),
        }
        report_path = (args.report or args.output_root / "_fetch_report.json").resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ManifestError, OSError, RemoteZipError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _inventory_sha256(pairs: Sequence[RGBPair]) -> str:
    """Fingerprint the selected member identity without hashing downloaded video bytes."""

    digest = hashlib.sha256()
    for pair in pairs:
        fields = (
            pair.sample_id,
            pair.member.member,
            str(pair.member.file_size),
            f"{pair.member.crc32:08x}",
        )
        digest.update("\0".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    entrypoint()
