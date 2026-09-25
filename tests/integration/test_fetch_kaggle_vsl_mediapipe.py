from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from silent_signal.cli.fetch_kaggle_vsl_mediapipe import main

_TOKEN = "KGAT_test-token"


def _archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for gloss, train_count in (("A", 3), ("B", 2), ("C", 1)):
            for split, count in (("train", train_count), ("test", 1)):
                for index in range(count):
                    values = np.full((4, 76, 3), index + 1, dtype=np.float32)
                    payload = io.BytesIO()
                    np.save(payload, values, allow_pickle=False)
                    archive.writestr(
                        f"processed/processed/keypoints_splited/{split}/{gloss}/{index}.npy",
                        payload.getvalue(),
                    )
        archive.writestr(
            "processed_augmented/processed_augmented/keypoints_splited/train/A/extra.npy",
            b"not-canonical",
        )
    return buffer.getvalue()


class _ArchiveStub(BaseHTTPRequestHandler):
    blob = b""
    download_requests = 0
    blob_requests = 0
    leaked_auth = False

    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        cls = type(self)
        if self.path.startswith("/download"):
            cls.download_requests += 1
            if self.headers.get("Authorization") != f"Bearer {_TOKEN}":
                self.send_error(401)
                return
            self.send_response(302)
            self.send_header("Location", f"http://{self.headers['Host']}/blob?signature=1")
            self.end_headers()
            return
        if self.headers.get("Authorization"):
            cls.leaked_auth = True
        cls.blob_requests += 1
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
def archive_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    _ArchiveStub.blob = _archive_bytes()
    _ArchiveStub.download_requests = 0
    _ArchiveStub.blob_requests = 0
    _ArchiveStub.leaked_auth = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ArchiveStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", _TOKEN)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/download"
    finally:
        server.shutdown()
    assert not _ArchiveStub.leaked_auth


def test_fetches_selected_keypoints_from_one_remote_archive(
    archive_server: str, tmp_path: Path
) -> None:
    output = tmp_path / "keypoints"
    arguments = [
        "--output-root",
        str(output),
        "--classes",
        "2",
        "--min-official-train-samples",
        "2",
        "--workers",
        "2",
        "--download-url",
        archive_server,
    ]
    assert main(arguments) == 0
    assert _ArchiveStub.download_requests == 1
    assert len(list(output.rglob("*.npy"))) == 7
    assert not (output / "train" / "C").exists()
    assert not (output / "processed_augmented").exists()
    report = json.loads((output / "_fetch_report.json").read_text(encoding="utf-8"))
    assert report["selected_glosses"] == ["A", "B"]
    assert report["auth_mode"] == "api_token"

    assert main(arguments) == 0
    assert _ArchiveStub.download_requests == 2
    resumed = json.loads((output / "_fetch_report.json").read_text(encoding="utf-8"))
    assert resumed["extraction"] == {"kept": 7, "extracted": 0}


def test_fetches_all_canonical_keypoints_when_classes_is_zero(
    archive_server: str, tmp_path: Path
) -> None:
    output = tmp_path / "all-keypoints"
    arguments = [
        "--output-root",
        str(output),
        "--classes",
        "0",
        "--workers",
        "2",
        "--download-url",
        archive_server,
    ]
    assert main(arguments) == 0
    assert _ArchiveStub.download_requests == 1
    assert len(list(output.rglob("*.npy"))) == 9
    assert (output / "train" / "C" / "0.npy").is_file()
    assert not (output / "processed_augmented").exists()
    report = json.loads((output / "_fetch_report.json").read_text(encoding="utf-8"))
    assert report["selection_strategy"] == "all_canonical_glosses"
    assert report["selected_glosses"] == ["A", "B", "C"]
    assert report["classes"] == 3
    assert report["files"] == 9
