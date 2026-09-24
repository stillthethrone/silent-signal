"""Write a class-subset manifest that keeps the source manifest's split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.data.class_subset import select_classes
from silent_signal.data.manifest import (
    ManifestError,
    manifest_summary,
    read_manifest,
    write_labels,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-select-classes",
        description="Select whole classes from a split manifest; the split is never changed.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Split manifest CSV/Parquet.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes",
        type=int,
        default=50,
        help="Top classes by training recordings (ignored with --gloss-id).",
    )
    parser.add_argument(
        "--gloss-id",
        action="append",
        default=[],
        help="Keep exactly this gloss ID; repeat to list classes in order.",
    )
    parser.add_argument("--dataset-name", default="vsl400")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = args.manifest.resolve()
        subset = select_classes(
            read_manifest(source), class_count=args.classes, gloss_ids=args.gloss_id
        )
        root = args.output_root.resolve()
        write_manifest(subset.records, root / "manifest.csv")
        write_manifest(subset.records, root / "manifest.parquet")
        write_labels(subset.labels, root / "labels.json", dataset=args.dataset_name)
        videos = sorted(record.video_path for record in subset.records)
        (root / "required_videos.txt").write_text("\n".join(videos) + "\n", encoding="utf-8")
        summary = manifest_summary(subset.records)
        splits = ("train", "validation", "test")
        payload = {
            "schema_version": 1,
            "dataset_name": args.dataset_name,
            "selection": subset.rule,
            "split_policy": "split copied unchanged from the source manifest; never re-split",
            "source_manifest": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "class_count": len(subset.classes),
            "clips": {split: summary["splits"].get(split, 0) for split in splits},
            "instances": {
                split: len({r.instance_id for r in subset.records if r.split == split})
                for split in splits
            },
            "signers": {
                split: sorted({r.signer_id for r in subset.records if r.split == split})
                for split in splits
            },
            "views": summary["views"],
            "classes": list(subset.classes),
        }
        (root / "selection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {key: payload[key] for key in ("class_count", "clips", "instances", "views")},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
