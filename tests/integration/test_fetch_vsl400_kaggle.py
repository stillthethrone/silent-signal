from __future__ import annotations

import base64
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from silent_signal.cli import fetch_vsl400_kaggle, prepare, select_classes
from silent_signal.data.manifest import read_manifest
from silent_signal.data.remote_zip import RemoteArchive, RemoteZipError

_USER, _KEY = "tester", "secret-key"
_VIEWS = ("front_view", "left_view", "right_view")


def _rows() -> list[dict[str, object]]:
    rows = []
    video_id = 0
    for signer in range(1, 9):
        for repetition in range(2):
            for gloss in ("Anh", "Cảm ơn"):
                rows.append(
                    {
                        "video_id": f"{video_id:06d}",
                        "signer_id": f"{signer:03d}",
                        "fps": 25.0,
                        "resolution": 1080,
                        "gloss": gloss,
                        "num_frames": 25 + repetition,
                        "length_seconds": 1.0 + repetition / 25,
                    }
                )
                video_id += 1
    return rows


def _archive_bytes() -> tuple[bytes, dict[str, bytes]]:
    rows = _rows()
    videos: dict[str, bytes] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("processed/processed/frame_splited/test/Anh/000000.mp4", b"crop")
        for part, chunk in ((1, rows[:16]), (2, rows[16:])):
            root = f"raw/raw/VSL400/Part_{part}/split_{part}"
            for view in _VIEWS:
                archive.writestr(f"{root}/{view}.json", json.dumps(chunk, ensure_ascii=False))
                for row in chunk:
                    payload = f"{view}-{row['video_id']}".encode() * 50
                    method = (
                        zipfile.ZIP_DEFLATED
                        if int(str(row["video_id"])) % 2
                        else zipfile.ZIP_STORED
                    )
                    archive.writestr(
                        f"{root}/{view}/{row['video_id']}.mp4", payload, compress_type=method
                    )
                    videos[f"{view}/{row['video_id']}.mp4"] = payload
    return buffer.getvalue(), videos


class _KaggleStub(BaseHTTPRequestHandler):
    blob = b""
    signature = 0
    blob_requests = 0
    expire_after = 10**9
    leaked_auth = False

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        cls = type(self)
        if self.path.startswith("/download"):
            expected = "Basic " + base64.b64encode(f"{_USER}:{_KEY}".encode()).decode()
            if self.headers.get("Authorization") != expected:
                self.send_error(401)
                return
            cls.signature += 1
            self.send_response(302)
            self.send_header("Location", f"http://{self.headers['Host']}/blob?sig={cls.signature}")
            self.end_headers()
            return
        if self.headers.get("Authorization"):
            cls.leaked_auth = True
        cls.blob_requests += 1
        if f"sig={cls.signature}" not in self.path or cls.blob_requests > cls.expire_after:
            cls.expire_after = 10**9
            self.send_error(403)
            return
        start, end = (
            int(value) for value in self.headers["Range"].removeprefix("bytes=").split("-")
        )
        end = min(end, len(cls.blob) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(cls.blob)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", '"v8"')
        self.end_headers()
        self.wfile.write(cls.blob[start : end + 1])


@pytest.fixture
def kaggle_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, dict[str, bytes]]]:
    blob, videos = _archive_bytes()
    _KaggleStub.blob, _KaggleStub.signature, _KaggleStub.blob_requests = blob, 0, 0
    _KaggleStub.expire_after, _KaggleStub.leaked_auth = 10**9, False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KaggleStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("KAGGLE_USERNAME", _USER)
    monkeypatch.setenv("KAGGLE_KEY", _KEY)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/download", videos
    finally:
        server.shutdown()
    assert not _KaggleStub.leaked_auth, "Kaggle credentials must not reach the storage URL."


def _run(url: str, *args: str) -> int:
    return fetch_vsl400_kaggle.main([*args, "--download-url", url])


def test_metadata_merges_parts_and_videos_extract_only_requested(
    kaggle_stub: tuple[str, dict[str, bytes]], tmp_path: Path
) -> None:
    url, videos = kaggle_stub
    meta_root = tmp_path / "meta"
    assert _run(url, "metadata", "--output-root", str(meta_root)) == 0
    for view in _VIEWS:
        assert len(json.loads((meta_root / f"{view}.json").read_text(encoding="utf-8"))) == 32
        assert (meta_root / view).is_dir()

    wanted = ["front_view/000003.mp4", "left_view/000020.mp4", "right_view/000031.mp4"]
    listing = tmp_path / "wanted.txt"
    listing.write_text("\n".join(wanted) + "\n", encoding="utf-8")
    local = tmp_path / "local"
    assert _run(url, "videos", "--output-root", str(local), "--list", str(listing)) == 0
    for relative in wanted:
        assert (local / relative).read_bytes() == videos[relative]
    assert sorted(p.relative_to(local).as_posix() for p in local.rglob("*.mp4")) == sorted(wanted)

    requests_before = _KaggleStub.blob_requests
    assert _run(url, "videos", "--output-root", str(local), "--list", str(listing)) == 0
    # A rerun only re-reads the central directory; kept files are not downloaded again.
    assert _KaggleStub.blob_requests - requests_before <= 4


def test_expired_signed_url_is_renewed(kaggle_stub: tuple[str, dict[str, bytes]]) -> None:
    url, _videos = kaggle_stub
    from silent_signal.data.kaggle_vsl400 import kaggle_resolver

    archive = RemoteArchive(kaggle_resolver(_USER, _KEY, download_url=url))
    _KaggleStub.expire_after = _KaggleStub.blob_requests
    assert archive.fetch(0, 3) == _KaggleStub.blob[:4]
    assert _KaggleStub.signature == 2


def test_rejects_bad_credentials_and_unknown_videos(
    kaggle_stub: tuple[str, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    url, _videos = kaggle_stub
    listing = tmp_path / "wanted.txt"
    listing.write_text("front_view/999999.mp4\n", encoding="utf-8")
    assert _run(url, "videos", "--output-root", str(tmp_path / "v"), "--list", str(listing)) == 2
    assert "not in the Kaggle archive" in capsys.readouterr().err

    monkeypatch.setenv("KAGGLE_KEY", "wrong")
    assert _run(url, "metadata", "--output-root", str(tmp_path / "m")) == 2
    assert "rejected the credentials" in capsys.readouterr().err


def test_kaggle_metadata_feeds_manifest_split_and_subset(
    kaggle_stub: tuple[str, dict[str, bytes]], config_file: Path, tmp_path: Path
) -> None:
    url, videos = kaggle_stub
    meta_root = tmp_path / "meta"
    assert _run(url, "metadata", "--output-root", str(meta_root)) == 0

    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["dataset"]["root"] = str(meta_root)
    full_config = tmp_path / "full.yaml"
    full_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert prepare.main(["build-manifest", "--config", str(full_config)]) == 0
    assert prepare.main(["create-splits", "--config", str(full_config)]) == 0
    full_manifest = Path(config["outputs"]["manifest_parquet"])
    records = read_manifest(full_manifest)
    assert len(records) == 96 and all(record.split for record in records)
    assert {record.metadata_duration_seconds for record in records} == {1.0, 1.04}

    prepared = tmp_path / "prepared"
    args = ["--manifest", str(full_manifest), "--output-root", str(prepared), "--classes", "2"]
    assert select_classes.main(args) == 0
    local = tmp_path / "local"
    listing = prepared / "required_videos.txt"
    assert _run(url, "videos", "--output-root", str(local), "--list", str(listing)) == 0
    for relative in listing.read_text(encoding="utf-8").split():
        assert (local / relative).read_bytes() == videos[relative]


def test_remote_archive_requires_range_support(tmp_path: Path) -> None:
    def resolve() -> tuple[str, dict[str, str]]:
        return (tmp_path / "missing").as_uri(), {}

    with pytest.raises((RemoteZipError, OSError)):
        RemoteArchive(resolve, retries=0)
