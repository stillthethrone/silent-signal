"""Read VSL400 from the Kaggle ``vsl-vietnamese-sign-language-v2`` archive.

The Kaggle dataset re-distributes the seven VSL400 release parts under
``raw/raw/VSL400/Part_{1..7}`` inside one ~75 GB download, next to the uploader's
front-view MediaPipe keypoints (``processed/processed/keypoints_splited``). Nothing here
downloads the whole archive: metadata JSONs are merged from their byte ranges, and only
the videos or keypoint arrays a subset needs are extracted. The Kaggle copy is a
third-party redistribution of a controlled-access dataset; confirm your right to use it
with the VSL400 maintainers.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from silent_signal.contracts import ManifestRecord
from silent_signal.data.keypoint_pack import write_packed_keypoints
from silent_signal.data.remote_zip import (
    RemoteArchive,
    RemoteZipError,
    Resolver,
    extract_members,
)

KAGGLE_DATASET = "nguyenanfms/vsl-vietnamese-sign-language-v2"
KAGGLE_VERSION = 8
KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}"
VIEWS = ("front_view", "left_view", "right_view")
STATE_FILE = ".kaggle_vsl400_source.json"

_JSON = re.compile(r"(?:^|/)VSL400/Part_(\d+)/(?:.*/)?(front_view|left_view|right_view)\.json$")
_VIDEO = re.compile(r"(?:^|/)VSL400/Part_(\d+)/(?:.*/)?(front_view|left_view|right_view)/([^/]+)$")
_KEYPOINTS = re.compile(r"(?:^|/)keypoints_splited/(train|test)/([^/]+)/([^/]+)\.npy$")


@dataclass(frozen=True, slots=True)
class VSL400Layout:
    """Where each release part keeps its metadata and videos inside the archive."""

    metadata: dict[str, list[tuple[int, str]]]
    videos: dict[str, str]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def kaggle_resolver(
    username: str,
    key: str,
    *,
    dataset: str = KAGGLE_DATASET,
    version: int = KAGGLE_VERSION,
    download_url: str | None = None,
) -> Resolver:
    """Resolve the pinned Kaggle download to a range-capable URL.

    Kaggle answers an authenticated request with a redirect to a short-lived signed
    storage URL. The credentials are sent only to Kaggle, never to the redirect target.
    """

    token = base64.b64encode(f"{username}:{key}".encode()).decode("ascii")
    auth = {"Authorization": f"Basic {token}"}
    base = download_url or KAGGLE_DOWNLOAD_URL.format(dataset=dataset)
    url = f"{base}?datasetVersionNumber={version}"
    opener = urllib.request.build_opener(_NoRedirect)

    def resolve() -> tuple[str, dict[str, str]]:
        request = urllib.request.Request(url, headers={**auth, "Range": "bytes=0-0"})
        try:
            with opener.open(request, timeout=60):
                return url, dict(auth)
        except urllib.error.HTTPError as error:
            location = error.headers.get("Location")
            if error.code in (301, 302, 303, 307, 308) and location:
                return location, {}
            if error.code in (401, 403):
                raise RemoteZipError(
                    "Kaggle rejected the credentials; check KAGGLE_USERNAME/KAGGLE_KEY and "
                    "that the account can download this dataset."
                ) from error
            raise RemoteZipError(
                f"Kaggle download request failed with HTTP {error.code}."
            ) from error

    return resolve


def locate_vsl400(members: Sequence[zipfile.ZipInfo]) -> VSL400Layout:
    """Find per-part metadata and videos; videos map to merged ``<view>/<file>`` paths."""

    metadata: dict[str, list[tuple[int, str]]] = defaultdict(list)
    videos: dict[str, str] = {}
    parts: set[int] = set()
    for item in members:
        if item.is_dir():
            continue
        if match := _JSON.search(item.filename):
            parts.add(int(match[1]))
            metadata[match[2]].append((int(match[1]), item.filename))
        elif (match := _VIDEO.search(item.filename)) and match[3].lower().endswith(".mp4"):
            relative = f"{match[2]}/{match[3]}"
            if relative in videos:
                raise RemoteZipError(f"Video {relative} occurs in more than one release part.")
            videos[relative] = item.filename
    if not parts:
        raise RemoteZipError("No VSL400/Part_*/<view>.json metadata found in the archive.")
    for view in VIEWS:
        found = {part for part, _ in metadata[view]}
        if found != parts or len(metadata[view]) != len(parts):
            raise RemoteZipError(
                f"{view}.json is missing or repeated in parts {sorted(parts - found)}."
            )
        metadata[view].sort()
    return VSL400Layout(metadata=dict(metadata), videos=videos)


def fetch_metadata(
    archive: RemoteArchive,
    members: Sequence[zipfile.ZipInfo],
    output_root: Path,
    *,
    source: dict[str, Any],
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Write merged ``<view>.json`` files and empty view folders, like ``merge_splits.py``."""

    output_root = output_root.resolve()
    _pin_source(output_root, archive, source)
    layout = locate_vsl400(members)
    summary: dict[str, Any] = {"parts": len(layout.metadata["front_view"]), "views": {}}
    with tempfile.TemporaryDirectory(dir=output_root) as scratch:
        targets = {
            name: Path(scratch) / f"part_{part}" / f"{view}.json"
            for view, entries in layout.metadata.items()
            for part, name in entries
        }
        extract_members(archive, targets, members, workers=4, reserve_bytes=0, report=report)
        for view in VIEWS:
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for part, name in layout.metadata[view]:
                rows = json.loads(targets[name].read_text(encoding="utf-8-sig"))
                if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                    raise RemoteZipError(f"Part {part} {view}.json is not a list of objects.")
                for row in rows:
                    video_id = str(row.get("video_id", ""))
                    if not video_id or video_id in seen:
                        raise RemoteZipError(
                            f"Missing or duplicate video_id {video_id!r} in {view}."
                        )
                    seen.add(video_id)
                    merged.append(row)
            (output_root / view).mkdir(exist_ok=True)
            (output_root / f"{view}.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
            archived = sum(1 for relative in layout.videos if relative.startswith(f"{view}/"))
            summary["views"][view] = {"metadata_rows": len(merged), "archived_videos": archived}
    report(f"[vsl400] merged metadata: {summary}")
    return summary


def fetch_videos(
    archive: RemoteArchive,
    members: Sequence[zipfile.ZipInfo],
    output_root: Path,
    relative_paths: Sequence[str],
    *,
    source: dict[str, Any],
    workers: int = 8,
    report: Callable[[str], None] = print,
) -> dict[str, int]:
    """Extract ``<view>/<file>.mp4`` paths (manifest ``video_path`` values) into a flat root."""

    output_root = output_root.resolve()
    _pin_source(output_root, archive, source)
    layout = locate_vsl400(members)
    targets: dict[str, Path] = {}
    for relative in relative_paths:
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] not in VIEWS
        ):
            raise RemoteZipError(
                f"Unexpected video path {relative!r}; expected <view>_view/<id>.mp4."
            )
        member = layout.videos.get(relative)
        if member is None:
            raise RemoteZipError(f"Video {relative} is not in the Kaggle archive.")
        targets[member] = output_root / path
    return extract_members(archive, targets, members, workers=workers, report=report)


def _pin_source(output_root: Path, archive: RemoteArchive, source: dict[str, Any]) -> None:
    identity = {**source, "size": archive.identity.size, "etag": archive.identity.etag}
    output_root.mkdir(parents=True, exist_ok=True)
    state = output_root / STATE_FILE
    if state.is_file():
        previous = json.loads(state.read_text(encoding="utf-8"))
        if previous != identity:
            raise RemoteZipError(
                f"{output_root} was filled from a different archive ({previous}); use a new folder."
            )
        return
    state.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")


def locate_mediapipe_keypoints(
    members: Sequence[zipfile.ZipInfo],
) -> dict[str, list[tuple[str, str, str]]]:
    """Map video IDs to canonical keypoint members as ``(gloss folder, member, split)``.

    The uploader's augmented copies (``processed_augmented``) are ignored: they are
    imputed and perturbed, so their missing-joint masks cannot be recovered.
    """

    index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for item in members:
        if item.is_dir() or "processed_augmented" in PurePosixPath(item.filename).parts:
            continue
        if match := _KEYPOINTS.search(item.filename):
            index[match[3]].append((match[2], item.filename, match[1]))
    if not index:
        raise RemoteZipError("No keypoints_splited/<split>/<gloss>/<id>.npy files in the archive.")
    return dict(index)


def fetch_keypoints(
    archive: RemoteArchive,
    members: Sequence[zipfile.ZipInfo],
    records: Sequence[ManifestRecord],
    output_path: Path,
    *,
    source: dict[str, Any],
    workers: int = 8,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Pack the MediaPipe keypoints of ``records`` (one view) into ``output_path``.

    A record matches a keypoint file with the same video ID whose gloss folder equals the
    record's gloss after Unicode/case normalization. Records without a usable match are
    reported, not guessed.
    """

    index = locate_mediapipe_keypoints(members)
    matched: dict[str, str] = {}
    status: dict[str, list[str]] = defaultdict(list)
    for record in records:
        candidates = [
            member
            for gloss, member, _split in index.get(record.video_id, [])
            if _gloss_key(gloss) == _gloss_key(record.gloss_name)
        ]
        if len(candidates) == 1:
            matched[record.sample_id] = candidates[0]
        elif candidates:
            status["ambiguous"].append(record.sample_id)
        elif record.video_id in index:
            status["gloss_mismatch"].append(record.sample_id)
        else:
            status["missing"].append(record.sample_id)

    sequences: dict[str, NDArray[np.float32]] = {}
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as scratch:
        targets = {
            member: Path(scratch) / f"{number}.npy"
            for number, member in enumerate(matched.values())
        }
        extract_members(archive, targets, members, workers=workers, reserve_bytes=0, report=report)
        for sample_id, member in matched.items():
            array = np.load(targets[member], allow_pickle=False)
            if (
                array.ndim != 3
                or array.shape[1:] != (76, 3)
                or len(array) == 0
                or not np.isfinite(array).all()
            ):
                status["invalid"].append(sample_id)
                continue
            sequences[sample_id] = array.astype(np.float32)
    counts = {"requested": len(records), "packed": len(sequences)} | {
        key: len(value) for key, value in sorted(status.items())
    }
    write_packed_keypoints(
        output_path,
        sequences,
        {
            "schema_version": 1,
            "format": "vsl_mediapipe_holistic_76",
            "source": {**source, "size": archive.identity.size, "etag": archive.identity.etag},
            "members": {sample_id: matched[sample_id] for sample_id in sorted(sequences)},
            "counts": counts,
            "unpacked": {key: sorted(value) for key, value in sorted(status.items())},
        },
    )
    report(f"[keypoints] {counts}")
    return counts


def _gloss_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())
