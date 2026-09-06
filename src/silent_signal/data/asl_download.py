"""Small, dependency-free download and extraction helpers for ASL Citizen.

No network requests run on import. The notebook explicitly enables each operation.
The official archive URL is published on the Microsoft Research project page.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ASL_CITIZEN_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
_CHUNK_SIZE = 8 * 1024**2
_SPACE_MARGIN = 2 * 1024**3
_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class ArchiveError(RuntimeError):
    """An archive cannot be downloaded or extracted without losing integrity."""


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    """Layout and storage requirements read from the ZIP central directory."""

    prefix: str
    file_count: int
    uncompressed_bytes: int


def download_archive(
    destination: Path,
    *,
    url: str = ASL_CITIZEN_URL,
    reserve_bytes: int = _SPACE_MARGIN,
    report: Callable[[str], None] = print,
) -> Path:
    """Download with validated HTTP Range resume into a separate ``.part`` file.

    The remote size and ETag/Last-Modified are saved alongside the archive. A
    changed remote object, unknown partial file or unsupported resume stops the
    operation instead of silently combining files or overwriting an existing ZIP.
    ZIP CRCs are checked later as entries are extracted; size alone is not a hash.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    metadata_path = destination.with_name(destination.name + ".download.json")
    request = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        size_header = response.headers.get("Content-Length")
        if not size_header or not size_header.isdigit() or int(size_header) <= 0:
            raise ArchiveError("Server did not provide a positive archive Content-Length.")
        remote: dict[str, Any] = {
            "url": url,
            "size": int(size_header),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != remote:
            raise ArchiveError("Remote archive changed. Choose a new archive destination.")
    elif destination.exists() or partial.exists():
        raise ArchiveError("Existing archive has no download metadata. Choose a new destination.")
    else:
        metadata_path.write_text(json.dumps(remote, indent=2) + "\n", encoding="utf-8")

    expected_size = int(remote["size"])
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != expected_size:
            raise ArchiveError("Existing archive size is incorrect; it will not be overwritten.")
        report(f"Archive already downloaded: {destination}")
        return destination
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        raise ArchiveError("Partial archive is larger than the remote file.")
    _require_space(destination.parent, expected_size - offset, reserve_bytes)
    report(
        f"Archive {expected_size / 1024**3:.2f} GiB; "
        f"remaining {(expected_size - offset) / 1024**3:.2f} GiB"
    )
    if offset < expected_size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            validator = remote["etag"] or remote["last_modified"]
            if not validator or str(validator).startswith("W/"):
                raise ArchiveError("Server has no reliable validator for resuming this file.")
            headers.update({"Range": f"bytes={offset}-", "If-Range": str(validator)})
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status == 206:
                match = _RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (
                    offset,
                    expected_size - 1,
                    expected_size,
                ):
                    raise ArchiveError("Server returned an unexpected Content-Range.")
            elif response.status != 200 or offset:
                raise ArchiveError(
                    "Server did not honor resume. Partial file was kept; use a new destination."
                )
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ArchiveError("Unexpected HTTP content encoding for ZIP download.")
            next_report = offset + 1024**3
            with partial.open("ab" if offset else "wb") as handle:
                while chunk := response.read(_CHUNK_SIZE):
                    if offset + len(chunk) > expected_size:
                        raise ArchiveError("Server sent more data than its advertised size.")
                    handle.write(chunk)
                    offset += len(chunk)
                    if offset >= next_report:
                        report(
                            f"Downloaded {offset / 1024**3:.2f}/{expected_size / 1024**3:.2f} GiB"
                        )
                        next_report += 1024**3
        if offset != expected_size:
            raise ArchiveError("Download interrupted. Rerun to resume the saved .part file.")
    partial.rename(destination)
    report(f"Download complete: {destination}")
    return destination


def inspect_archive(archive_path: Path) -> ArchiveInspection:
    """Find the unique dataset layout, reject unsafe members and measure disk needs."""

    with zipfile.ZipFile(archive_path) as archive:
        prefix, members = _dataset_members(archive)
        return ArchiveInspection(
            prefix=prefix,
            file_count=sum(not item.is_dir() for item, _ in members),
            uncompressed_bytes=sum(item.file_size for item, _ in members),
        )


def extract_archive(
    archive_path: Path,
    dataset_root: Path,
    *,
    reserve_bytes: int = _SPACE_MARGIN,
    report: Callable[[str], None] = print,
) -> Path:
    """Extract under the requested root, checking ZIP CRCs and never replacing files.

    A repeated call resumes: existing files must match their archive CRC and size.
    Each new file is made visible only after its entire ZIP entry has been checked.
    An interruption leaves already completed files available for the next run.
    """

    dataset_root = Path(dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _, members = _dataset_members(archive)
        pending: list[tuple[zipfile.ZipInfo, Path]] = []
        for item, relative in members:
            target = dataset_root / relative
            _check_target(dataset_root, target)
            if item.is_dir():
                if target.exists() and not target.is_dir():
                    raise ArchiveError(f"An existing file blocks a directory: {target}")
                continue
            if target.exists():
                if not target.is_file() or not _matches_entry(target, item):
                    raise ArchiveError(f"Existing file differs from archive: {target}")
            else:
                pending.append((item, target))
        required = sum(item.file_size for item, _ in pending)
        _require_space(dataset_root, required, reserve_bytes)
        report(f"Extract {len(pending):,} remaining files ({required / 1024**3:.2f} GiB)")
        for index, (item, target) in enumerate(pending, start=1):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with (
                    archive.open(item) as source,
                    tempfile.NamedTemporaryFile(
                        prefix=".silent-signal-",
                        suffix=".extracting",
                        dir=target.parent,
                        delete=False,
                    ) as output,
                ):
                    temporary = Path(output.name)
                    shutil.copyfileobj(source, output, length=_CHUNK_SIZE)
                if temporary.stat().st_size != item.file_size:
                    raise ArchiveError(
                        f"Extracted file size differs from ZIP entry: {item.filename}"
                    )
                if target.exists():
                    raise ArchiveError(f"A file appeared during extraction: {target}")
                temporary.rename(target)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            if index % 1000 == 0 or index == len(pending):
                report(f"Extracted {index:,}/{len(pending):,}")
    report(f"Dataset ready for metadata validation: {dataset_root}")
    return dataset_root


def _dataset_members(archive: zipfile.ZipFile) -> tuple[str, list[tuple[zipfile.ZipInfo, Path]]]:
    entries = archive.infolist()
    names: set[str] = set()
    for item in entries:
        name = item.filename
        path = PurePosixPath(name)
        file_type = stat.S_IFMT(item.external_attr >> 16)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or "\x00" in name
            or file_type not in (0, stat.S_IFREG, stat.S_IFDIR)
            or item.flag_bits & 0x1
        ):
            raise ArchiveError(f"Unsafe or unsupported ZIP member: {name!r}")
        normalized = path.as_posix().rstrip("/").casefold()
        if normalized in names:
            raise ArchiveError(f"Duplicate ZIP member: {name!r}")
        names.add(normalized)
    file_names = {item.filename for item in entries if not item.is_dir()}
    prefixes = {
        name.removesuffix("splits/train.csv")
        for name in file_names
        if name == "splits/train.csv" or name.endswith("/splits/train.csv")
    }
    candidates = [
        prefix
        for prefix in prefixes
        if all(prefix + f"splits/{split}.csv" in file_names for split in ("train", "val", "test"))
        and any(name.startswith(prefix + "videos/") for name in file_names)
    ]
    if len(candidates) != 1:
        raise ArchiveError(
            "Expected one ASL Citizen root with videos/ and splits/train.csv, val.csv, test.csv."
        )
    prefix = candidates[0]
    members = [
        (item, Path(*PurePosixPath(item.filename[len(prefix) :]).parts))
        for item in entries
        if item.filename.startswith(prefix) and item.filename != prefix
    ]
    return prefix, members


def _check_target(root: Path, target: Path) -> None:
    if not target.resolve().is_relative_to(root):
        raise ArchiveError(f"Extraction path escapes dataset root: {target}")
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ArchiveError(f"Extraction path contains a symlink: {candidate}")


def _matches_entry(path: Path, item: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != item.file_size:
        return False
    checksum = 0
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == item.CRC


def _require_space(path: Path, required: int, reserve: int) -> None:
    if reserve < 0:
        raise ValueError("reserve_bytes must not be negative")
    free = shutil.disk_usage(path).free
    if free < required + reserve:
        raise ArchiveError(
            f"Insufficient disk space at {path}: {free / 1024**3:.2f} GiB free, "
            f"{(required + reserve) / 1024**3:.2f} GiB required including reserve."
        )
