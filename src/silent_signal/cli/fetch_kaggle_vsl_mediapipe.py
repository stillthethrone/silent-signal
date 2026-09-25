"""Fetch a selected canonical MediaPipe subset through one Kaggle ZIP archive."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from silent_signal.data.remote_zip import (
    RemoteArchive,
    RemoteZipError,
    Resolver,
    extract_members,
    list_members,
)

DEFAULT_DATASET = "nguyenanfms/vsl-vietnamese-sign-language-v2"
DEFAULT_VERSION = 0
DEFAULT_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}"
_KEYPOINT = re.compile(
    r"(?:^|/)processed/processed/keypoints_splited/"
    r"(train|test)/([^/]+)/([^/]+\.npy)$"
)


@dataclass(frozen=True, slots=True)
class KeypointMember:
    split: str
    gloss: str
    filename: str
    member: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-fetch-kaggle-vsl-mediapipe",
        description=(
            "Open the Kaggle dataset as one remote ZIP and extract only the selected "
            "canonical MediaPipe keypoints. No per-file Kaggle API listing is used."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes",
        type=int,
        default=70,
        help="Number of top classes, or zero to extract every canonical keypoint class.",
    )
    parser.add_argument("--min-official-train-samples", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
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
        if args.classes < 0:
            raise ValueError("--classes must be zero (all) or positive.")
        if args.min_official_train_samples < 1:
            raise ValueError("--min-official-train-samples must be positive.")
        if args.workers < 1:
            raise ValueError("--workers must be positive.")

        auth_headers, auth_mode = _credentials()
        resolver = kaggle_archive_resolver(
            auth_headers,
            dataset=args.dataset,
            version=args.version,
            download_url=args.download_url,
        )
        archive = RemoteArchive(resolver)
        print(
            f"[kaggle] archive={archive.identity.size / 1024**3:.2f} GiB "
            f"| auth={auth_mode}",
            flush=True,
        )
        members = list_members(archive)
        print(f"[kaggle] central directory: {len(members):,} entries", flush=True)

        canonical = locate_canonical_keypoints(members)
        selected = select_glosses(
            canonical,
            classes=args.classes,
            min_train=args.min_official_train_samples,
        )
        targets = extraction_targets(canonical, selected, args.output_root.resolve())
        counts = {
            gloss: {
                split: sum(item.gloss == gloss and item.split == split for item in canonical)
                for split in ("train", "test")
            }
            for gloss in selected
        }
        print(
            f"[selection] {len(selected)} classes | {len(targets):,} canonical keypoint files",
            flush=True,
        )
        extraction = extract_members(
            archive,
            targets,
            members,
            workers=args.workers,
            reserve_bytes=1024**3,
        )
        report = {
            "schema_version": 1,
            "status": "complete",
            "dataset": args.dataset,
            "requested_version": args.version or "latest",
            "archive_size": archive.identity.size,
            "archive_etag": archive.identity.etag,
            "auth_mode": auth_mode,
            "selection_strategy": (
                "all_canonical_glosses"
                if args.classes == 0
                else "descending_official_train_count_then_gloss"
            ),
            "classes": len(selected),
            "min_official_train_samples": args.min_official_train_samples,
            "selected_glosses": list(selected),
            "counts": counts,
            "files": len(targets),
            "extraction": extraction,
            "output_root": str(args.output_root.resolve()),
        }
        report_path = args.report or args.output_root / "_fetch_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, RemoteZipError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _credentials() -> tuple[dict[str, str], str]:
    username = (os.environ.get("KAGGLE_USERNAME") or "").strip()
    key = (os.environ.get("KAGGLE_KEY") or "").strip()
    if username and key:
        encoded = base64.b64encode(f"{username}:{key}".encode()).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}, "username_key"
    token = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}, "api_token"
    raise ValueError(
        "Set KAGGLE_USERNAME and KAGGLE_KEY, or set KAGGLE_API_TOKEN."
    )


def kaggle_archive_resolver(
    auth_headers: Mapping[str, str],
    *,
    dataset: str = DEFAULT_DATASET,
    version: int = DEFAULT_VERSION,
    download_url: str | None = None,
) -> Resolver:
    """Resolve one authenticated Kaggle archive URL without leaking credentials."""

    base = download_url or DEFAULT_DOWNLOAD_URL.format(dataset=dataset)
    if version < 0:
        raise ValueError("version must be zero (latest) or positive")
    if version:
        separator = "&" if "?" in base else "?"
        url = f"{base}{separator}datasetVersionNumber={version}"
    else:
        url = base
    opener = urllib.request.build_opener(_NoRedirect)

    def resolve() -> tuple[str, dict[str, str]]:
        request = urllib.request.Request(
            url,
            headers={**dict(auth_headers), "Range": "bytes=0-0"},
        )
        try:
            with opener.open(request, timeout=60):
                return url, dict(auth_headers)
        except urllib.error.HTTPError as error:
            location = error.headers.get("Location")
            if error.code in (301, 302, 303, 307, 308) and location:
                return location, {}
            if error.code in (401, 403):
                raise RemoteZipError(
                    "Kaggle rejected the credentials or dataset access."
                ) from error
            if error.code == 429:
                raise RemoteZipError(
                    "Kaggle rate-limited the single archive request; wait a few minutes once."
                ) from error
            raise RemoteZipError(
                f"Kaggle archive request failed with HTTP {error.code}."
            ) from error

    return resolve


def locate_canonical_keypoints(
    members: Sequence[zipfile.ZipInfo],
) -> tuple[KeypointMember, ...]:
    found: list[KeypointMember] = []
    for item in members:
        path = PurePosixPath(item.filename)
        if item.is_dir() or "processed_augmented" in path.parts:
            continue
        match = _KEYPOINT.search(item.filename)
        if not match:
            continue
        found.append(
            KeypointMember(
                split=match[1],
                gloss=unicodedata.normalize("NFC", match[2]),
                filename=match[3],
                member=item.filename,
            )
        )
    if not found:
        raise RemoteZipError(
            "No canonical processed/processed/keypoints_splited files in the archive."
        )
    return tuple(found)


def select_glosses(
    members: Sequence[KeypointMember],
    *,
    classes: int,
    min_train: int,
) -> tuple[str, ...]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    display_by_key: dict[str, str] = {}
    for item in members:
        key = _gloss_key(item.gloss)
        display_by_key.setdefault(key, item.gloss)
        counts[key][item.split] += 1
    if classes == 0:
        return tuple(display_by_key[key] for key in sorted(counts))
    eligible = [
        key
        for key, split_counts in counts.items()
        if split_counts["train"] >= min_train and split_counts["test"] > 0
    ]
    eligible.sort(key=lambda key: (-counts[key]["train"], key))
    selected = tuple(display_by_key[key] for key in eligible[:classes])
    if len(selected) != classes:
        raise RemoteZipError(
            f"Requested {classes} classes but only {len(selected)} satisfy the requirements."
        )
    return selected


def extraction_targets(
    members: Sequence[KeypointMember],
    selected: Sequence[str],
    output_root: Path,
) -> dict[str, Path]:
    selected_keys = {_gloss_key(value) for value in selected}
    targets: dict[str, Path] = {}
    destinations: set[Path] = set()
    for item in members:
        if _gloss_key(item.gloss) not in selected_keys:
            continue
        destination = output_root / item.split / item.gloss / item.filename
        if destination in destinations:
            raise RemoteZipError(f"Duplicate keypoint destination: {destination}")
        destinations.add(destination)
        targets[item.member] = destination
    if not targets:
        raise RemoteZipError("The selected classes contain no keypoint files.")
    return targets


def _gloss_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


if __name__ == "__main__":
    entrypoint()
