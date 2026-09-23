"""Build a Multi-VSL M-VSL200 center-view baseline manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from silent_signal.data.multi_vsl import prepare_multi_vsl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_multi_vsl(
        metadata_root=args.metadata_root,
        video_root=args.video_root,
        output_root=args.output_root,
        class_count=args.classes,
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "selection": str(result.selection_path),
                "summary": str(result.summary_path),
                "selected_classes": list(result.selected_classes),
                "split_counts": result.split_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
