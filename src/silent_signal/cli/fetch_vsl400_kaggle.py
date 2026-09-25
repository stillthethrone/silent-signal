"""Fetch VSL400 metadata and selected raw videos from the Kaggle archive via HTTP Range."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.data.kaggle_vsl400 import (
    KAGGLE_DATASET,
    KAGGLE_VERSION,
    fetch_metadata,
    fetch_videos,
    kaggle_resolver,
)
from silent_signal.data.remote_zip import RemoteArchive, RemoteZipError, list_members


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--dataset", default=KAGGLE_DATASET)
    common.add_argument("--version", type=int, default=KAGGLE_VERSION)
    common.add_argument("--download-url", help=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        prog="ss-fetch-vsl400-kaggle",
        description=(
            "Read VSL400 from the Kaggle archive without downloading it whole. Credentials "
            "come from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "metadata",
        parents=[common],
        help="Merge the seven parts' view JSONs into <output-root>/<view>.json.",
    )
    videos = subparsers.add_parser(
        "videos",
        parents=[common],
        help="Extract the listed <view>/<id>.mp4 paths into <output-root>.",
    )
    videos.add_argument("--list", type=Path, required=True, help="One video_path per line.")
    videos.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        username, key = _credentials()
        source = {"dataset": args.dataset, "version": args.version}
        archive = RemoteArchive(
            kaggle_resolver(
                username,
                key,
                dataset=args.dataset,
                version=args.version,
                download_url=args.download_url,
            )
        )
        print(f"[kaggle] {args.dataset} v{args.version}: {archive.identity.size / 1024**3:.2f} GiB")
        members = list_members(archive)
        print(f"[kaggle] central directory: {len(members):,} entries", flush=True)
        if args.command == "metadata":
            summary = fetch_metadata(archive, members, args.output_root, source=source)
        else:
            paths = [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines()]
            summary = fetch_videos(
                archive,
                members,
                args.output_root,
                [path for path in paths if path],
                source=source,
                workers=args.workers,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RemoteZipError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("username") and payload.get("key"):
            return str(payload["username"]), str(payload["key"])
    raise ValueError("Set KAGGLE_USERNAME and KAGGLE_KEY, or provide ~/.kaggle/kaggle.json.")


if __name__ == "__main__":
    entrypoint()
