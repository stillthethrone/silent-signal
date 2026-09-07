from __future__ import annotations

import io
import json
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from silent_signal.data.asl_download import (
    ArchiveError,
    download_archive,
    extract_archive,
    extract_remote_archive,
    inspect_archive,
)


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers


def _mock_download(
    monkeypatch: pytest.MonkeyPatch, responses: list[Response]
) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    def request(item: urllib.request.Request, **kwargs: Any) -> Response:
        requests.append(item)
        return responses.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", request)
    return requests


def _head() -> Response:
    return Response(b"", 200, {"Content-Length": "6", "ETag": '"release-v1"'})


def test_truncated_download_resumes_from_saved_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset.zip"
    requests = _mock_download(
        monkeypatch,
        [
            _head(),
            Response(b"abc", 200, {}),
            _head(),
            Response(b"def", 206, {"Content-Range": "bytes 3-5/6"}),
            _head(),
        ],
    )
    with pytest.raises(ArchiveError, match="interrupted"):
        download_archive(destination, reserve_bytes=0)
    assert not destination.exists()
    assert (tmp_path / "dataset.zip.part").read_bytes() == b"abc"

    download_archive(destination, reserve_bytes=0)
    assert destination.read_bytes() == b"abcdef"
    assert requests[3].get_header("Range") == "bytes=3-"
    assert requests[3].get_header("If-range") == '"release-v1"'
    assert not (tmp_path / "dataset.zip.part").exists()
    download_archive(destination, reserve_bytes=0)
    assert len(requests) == 5


@pytest.mark.parametrize(
    "response",
    [
        Response(b"abcdef", 200, {}),
        Response(b"def", 206, {"Content-Range": "bytes 2-5/6"}),
    ],
)
def test_resume_rejects_wrong_range_without_clobbering_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: Response
) -> None:
    destination = tmp_path / "dataset.zip"
    _mock_download(monkeypatch, [_head(), Response(b"abc", 200, {}), _head(), response])
    with pytest.raises(ArchiveError, match="interrupted"):
        download_archive(destination, reserve_bytes=0)
    with pytest.raises(ArchiveError, match=r"resume|Content-Range"):
        download_archive(destination, reserve_bytes=0)
    assert (tmp_path / "dataset.zip.part").read_bytes() == b"abc"


def test_changed_remote_and_unknown_file_are_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"user-content")
    _mock_download(monkeypatch, [_head(), _head()])
    with pytest.raises(ArchiveError, match="no download metadata"):
        download_archive(destination, reserve_bytes=0)
    destination.with_name("dataset.zip.download.json").write_text(
        json.dumps({"url": "https://example.com/old.zip"}), encoding="utf-8"
    )
    with pytest.raises(ArchiveError, match="changed"):
        download_archive(destination, reserve_bytes=0)
    assert destination.read_bytes() == b"user-content"


def _archive(path: Path, prefix: str = "ASL_Citizen/") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for split in ("train", "val", "test"):
            archive.writestr(f"{prefix}splits/{split}.csv", "Participant ID,Video file,Gloss\n")
        archive.writestr(f"{prefix}videos/001.mp4", b"synthetic-video")
    return path


def _mock_remote_archive(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    *,
    etag: str = '"release-v1"',
) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    def request(item: urllib.request.Request, **kwargs: Any) -> Response:
        requests.append(item)
        if item.get_method() == "HEAD":
            return Response(
                b"",
                200,
                {"Content-Length": str(len(payload)), "ETag": etag},
            )
        range_header = item.get_header("Range")
        assert range_header is not None
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        start, end = int(start_text), int(end_text)
        data = payload[start : end + 1]
        return Response(
            data,
            206,
            {
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "Content-Length": str(len(data)),
            },
        )

    monkeypatch.setattr(urllib.request, "urlopen", request)
    return requests


def test_remote_extract_uses_ranges_and_resumes_completed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path / "dataset.zip")
    requests = _mock_remote_archive(monkeypatch, archive.read_bytes())
    destination = tmp_path / "data"

    extract_remote_archive(
        destination,
        url="https://example.com/dataset.zip",
        reserve_bytes=0,
        range_chunk_size=64,
    )
    target = destination / "videos/001.mp4"
    original = target.stat().st_mtime_ns
    assert target.read_bytes() == b"synthetic-video"
    assert (destination / ".silent-signal-remote-archive.json").is_file()
    assert all(
        request.get_header("Range") for request in requests if request.get_method() != "HEAD"
    )
    assert all(
        request.get_header("If-range") == '"release-v1"'
        for request in requests
        if request.get_method() != "HEAD"
    )

    extract_remote_archive(
        destination,
        url="https://example.com/dataset.zip",
        reserve_bytes=0,
        range_chunk_size=64,
    )
    assert target.stat().st_mtime_ns == original


def test_remote_extract_rejects_changed_release_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path / "dataset.zip")
    payload = archive.read_bytes()
    destination = tmp_path / "data"
    _mock_remote_archive(monkeypatch, payload)
    extract_remote_archive(
        destination,
        url="https://example.com/dataset.zip",
        reserve_bytes=0,
        range_chunk_size=64,
    )

    _mock_remote_archive(monkeypatch, payload, etag='"release-v2"')
    with pytest.raises(ArchiveError, match="identity changed"):
        extract_remote_archive(
            destination,
            url="https://example.com/dataset.zip",
            reserve_bytes=0,
            range_chunk_size=64,
        )


@pytest.mark.parametrize("prefix", ["", "ASL_Citizen/", "release/ASL_Citizen/"])
def test_extract_detects_root_and_resumes_existing_files(tmp_path: Path, prefix: str) -> None:
    archive = _archive(tmp_path / "dataset.zip", prefix)
    inspection = inspect_archive(archive)
    assert inspection.prefix == prefix
    assert inspection.file_count == 4
    destination = tmp_path / "data"
    extract_archive(archive, destination, reserve_bytes=0)
    original = (destination / "videos/001.mp4").stat().st_mtime_ns
    extract_archive(archive, destination, reserve_bytes=0)
    assert (destination / "videos/001.mp4").stat().st_mtime_ns == original
    assert (destination / "splits/val.csv").is_file()


@pytest.mark.parametrize(
    "bad_name", ["../outside", "/absolute", "ASL_Citizen/../outside", "C:/file"]
)
def test_zip_traversal_is_rejected_before_extraction(tmp_path: Path, bad_name: str) -> None:
    archive = _archive(tmp_path / "dataset.zip")
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(bad_name, b"not-a-video")
    with pytest.raises(ArchiveError, match="Unsafe"):
        extract_archive(archive, tmp_path / "data", reserve_bytes=0)
    assert not (tmp_path / "data/videos").exists()


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "dataset.zip")
    with zipfile.ZipFile(archive, "a") as output:
        link = zipfile.ZipInfo("ASL_Citizen/videos/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, "../../../outside")
    with pytest.raises(ArchiveError, match="Unsafe"):
        inspect_archive(archive)


def test_existing_different_file_is_preserved(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "dataset.zip")
    destination = tmp_path / "data"
    target = destination / "videos/001.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different-video")
    with pytest.raises(ArchiveError, match="differs"):
        extract_archive(archive, destination, reserve_bytes=0)
    assert target.read_bytes() == b"different-video"


def test_extraction_checks_uncompressed_disk_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import namedtuple

    archive = _archive(tmp_path / "dataset.zip")
    disk = namedtuple("Disk", "total used free")
    monkeypatch.setattr(
        "silent_signal.data.asl_download.shutil.disk_usage", lambda path: disk(10, 10, 0)
    )
    with pytest.raises(ArchiveError, match="Insufficient disk"):
        extract_archive(archive, tmp_path / "data", reserve_bytes=0)
    assert not (tmp_path / "data/videos").exists()
