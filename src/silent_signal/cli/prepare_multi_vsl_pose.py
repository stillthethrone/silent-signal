"""Build the Multi-VSL center-view manifest consumed by RTMPose extraction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.contracts import ValidationLevel
from silent_signal.data.manifest import ManifestError, manifest_summary
from silent_signal.data.multi_vsl import build_multi_vsl_pose_dataset, select_multi_vsl
from silent_signal.data.validation import ValidationError


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Directory containing the official *_1_200_center_ord1.csv files.",
    )
    common.add_argument(
        "--classes",
        type=int,
        default=50,
        help="Top classes by official training clip count; 0 keeps every class.",
    )

    parser = argparse.ArgumentParser(
        prog="ss-prepare-multi-vsl-pose",
        description="Select official Multi-VSL center-view clips and write a pose manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-videos",
        parents=[common],
        help="Write the official filenames the selection needs, one per line.",
    )
    list_parser.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser(
        "build",
        parents=[common],
        help="Validate local videos and write manifest, labels, split and reports.",
    )
    build.add_argument("--video-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument(
        "--level",
        choices=[level.value for level in ValidationLevel],
        default=ValidationLevel.METADATA.value,
        help="probe additionally reads every header with ffprobe.",
    )
    build.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-videos":
            selection = select_multi_vsl(args.metadata_root, args.classes)
            names = selection.required_videos
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(names) + "\n", encoding="utf-8")
            _print_json(
                {
                    "classes": len(selection.classes),
                    "videos": len(names),
                    "signers": selection.signer_splits,
                    "output": str(args.output.resolve()),
                }
            )
            return 0

        dataset = build_multi_vsl_pose_dataset(
            metadata_root=args.metadata_root,
            video_root=args.video_root,
            output_root=args.output_root,
            class_count=args.classes,
            level=args.level,
            workers=args.workers,
        )
        report = dataset.validation.to_report()
        _print_json(
            {
                "passed": report["passed"],
                "summary": manifest_summary(dataset.validation.records),
                "issue_counts": report["issue_counts"],
                "manifest": str(dataset.manifest_csv),
                "report": str(dataset.report_path),
            }
        )
        return 1 if dataset.validation.has_errors else 0
    except (FileNotFoundError, ManifestError, RuntimeError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    entrypoint()
