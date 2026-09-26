from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from silent_signal.data.remote_zip import (
    RemoteIdentity,
    RemoteZipDirectory,
    RemoteZipError,
    extract_members_coalesced,
)

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")


class _MemoryArchive:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.identity = RemoteIdentity(size=len(blob), etag='"memory"')
        self.calls: list[tuple[int, int]] = []

    def fetch(self, start: int, end: int) -> bytes:
        self.calls.append((start, end))
        return self.blob[start : end + 1]


def _directory(blob: bytes) -> RemoteZipDirectory:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return RemoteZipDirectory(
            members=tuple(archive.infolist()),
            start_dir=archive.start_dir,
        )


def _zip_blob(*, stored: bool = False) -> bytes:
    buffer = io.BytesIO()
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("selected/a.mp4", b"a" * 400)
        archive.writestr("unused/gap.bin", b"gap" * 500)
        archive.writestr("selected/b.mp4", b"b" * 500)
        archive.writestr("selected/c.mp4", b"c" * 600)
    return buffer.getvalue()


def test_coalesced_extraction_uses_exact_start_dir_and_resumes_crc(tmp_path: Path) -> None:
    blob = _zip_blob()
    directory = _directory(blob)
    archive = _MemoryArchive(blob)
    targets = {
        "selected/a.mp4": tmp_path / "a.mp4",
        "selected/b.mp4": tmp_path / "b.mp4",
        "selected/c.mp4": tmp_path / "c.mp4",
    }

    first = extract_members_coalesced(
        archive,
        targets,
        directory,
        workers=1,
        reserve_bytes=0,
        max_span_bytes=len(blob),
        max_gap_bytes=len(blob),
        report=lambda _: None,
    )

    first_selected = min(
        item.header_offset
        for item in directory.members
        if item.filename in targets
    )
    assert archive.calls == [(first_selected, directory.start_dir - 1)]
    assert first["extracted"] == 3
    assert first["range_requests"] == 1
    assert first["overfetch_bytes"] > 0
    assert targets["selected/a.mp4"].read_bytes() == b"a" * 400
    assert targets["selected/b.mp4"].read_bytes() == b"b" * 500
    assert targets["selected/c.mp4"].read_bytes() == b"c" * 600

    archive.calls.clear()
    resumed = extract_members_coalesced(
        archive,
        targets,
        directory,
        workers=1,
        reserve_bytes=0,
        report=lambda _: None,
    )

    assert archive.calls == []
    assert resumed == {
        "kept": 3,
        "extracted": 0,
        "range_requests": 0,
        "range_bytes": 0,
        "selected_record_bytes": 0,
        "overfetch_bytes": 0,
    }

    targets["selected/c.mp4"].write_bytes(b"x" * 600)
    archive.calls.clear()
    repaired = extract_members_coalesced(
        archive,
        targets,
        directory,
        workers=1,
        reserve_bytes=0,
        report=lambda _: None,
    )

    last = next(item for item in directory.members if item.filename == "selected/c.mp4")
    assert archive.calls == [(last.header_offset, directory.start_dir - 1)]
    assert repaired["kept"] == 2
    assert repaired["extracted"] == 1
    assert targets["selected/c.mp4"].read_bytes() == b"c" * 600


def test_coalescing_respects_unselected_gap_limit(tmp_path: Path) -> None:
    blob = _zip_blob()
    directory = _directory(blob)
    archive = _MemoryArchive(blob)
    targets = {
        "selected/a.mp4": tmp_path / "a.mp4",
        "selected/b.mp4": tmp_path / "b.mp4",
        "selected/c.mp4": tmp_path / "c.mp4",
    }

    result = extract_members_coalesced(
        archive,
        targets,
        directory,
        workers=1,
        reserve_bytes=0,
        max_span_bytes=len(blob),
        max_gap_bytes=0,
        report=lambda _: None,
    )

    assert result["range_requests"] == 2
    assert len(archive.calls) == 2
    assert result["overfetch_bytes"] == 0


def test_coalescing_respects_maximum_merged_span(tmp_path: Path) -> None:
    blob = _zip_blob()
    directory = _directory(blob)
    archive = _MemoryArchive(blob)
    second = next(item for item in directory.members if item.filename == "selected/b.mp4")
    targets = {
        "selected/b.mp4": tmp_path / "b.mp4",
        "selected/c.mp4": tmp_path / "c.mp4",
    }
    combined_span = directory.start_dir - second.header_offset

    result = extract_members_coalesced(
        archive,
        targets,
        directory,
        workers=1,
        reserve_bytes=0,
        max_span_bytes=combined_span - 1,
        max_gap_bytes=len(blob),
        report=lambda _: None,
    )

    assert result["range_requests"] == 2
    assert len(archive.calls) == 2


def test_corrupt_span_never_publishes_target(tmp_path: Path) -> None:
    blob = _zip_blob(stored=True)
    directory = _directory(blob)
    item = next(item for item in directory.members if item.filename == "selected/a.mp4")
    fields = _LOCAL_HEADER.unpack_from(blob, item.header_offset)
    data_start = item.header_offset + _LOCAL_HEADER.size + fields[9] + fields[10]
    corrupted = bytearray(blob)
    corrupted[data_start] ^= 0xFF
    archive = _MemoryArchive(bytes(corrupted))
    target = tmp_path / "a.mp4"

    with pytest.raises(RemoteZipError, match="CRC mismatch"):
        extract_members_coalesced(
            archive,
            {item.filename: target},
            directory,
            workers=1,
            reserve_bytes=0,
            report=lambda _: None,
        )

    assert not target.exists()
