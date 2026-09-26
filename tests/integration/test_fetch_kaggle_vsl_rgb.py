from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from silent_signal.cli.fetch_kaggle_vsl_rgb import main
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import read_manifest, write_manifest

_TOKEN = "KGAT_rgb-test-token"


def _record(
    sample_id: str,
    physical_split: str,
    experimental_split: str,
    gloss: str,
    stem: str,
    class_index: int,
) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=stem,
        signer_id="unknown",
        gloss_id=f"kvsl:{gloss}",
        gloss_name=gloss,
        class_index=class_index,
        view="front",
        video_path=f"{physical_split}/{gloss}/{stem}.npy",
        split=experimental_split,
    )


def _archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split, gloss, stem, payload in (
            ("train", "A", "000001", b"train-a"),
            ("train", "A", "000002", b"validation-a"),
            ("test", "A", "000003", b"test-a"),
            ("train", "B", "000010", b"unused-b"),
        ):
            archive.writestr(
                f"processed/processed/frame_splited/{split}/{gloss}/{stem}.mp4",
                payload,
            )
        archive.writestr(
            "processed_augmented/processed_augmented/frame_splited/train/A/extra.mp4",
            b"augmented",
        )
    return buffer.getvalue()


class _ArchiveStub(BaseHTTPRequestHandler):
    blob = b""
    download_requests = 0
    range_requests = 0
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
        cls.range_requests += 1
        start, end = (
            int(value) for value in self.headers["Range"].removeprefix("bytes=").split("-")
        )
        end = min(end, len(cls.blob) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(cls.blob)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("ETag", '"rgb-v1"')
        self.end_headers()
        self.wfile.write(cls.blob[start : end + 1])


@pytest.fixture
def archive_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    _ArchiveStub.blob = _archive_bytes()
    _ArchiveStub.download_requests = 0
    _ArchiveStub.range_requests = 0
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


def test_fetches_exact_manifest_rgb_subset_and_resumes(
    archive_server: str, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        (
            _record("train", "train", "train", "A", "000001", 0),
            _record("validation", "train", "validation", "A", "000002", 0),
            _record("test", "test", "test", "A", "000003", 0),
        ),
        manifest,
    )
    output = tmp_path / "rgb"
    rgb_manifest = tmp_path / "rgb_manifest.csv"
    arguments = [
        "--manifest",
        str(manifest),
        "--output-root",
        str(output),
        "--rgb-manifest",
        str(rgb_manifest),
        "--workers",
        "2",
        "--download-url",
        archive_server,
    ]

    assert main(arguments) == 0
    assert len(list(output.rglob("*.mp4"))) == 3
    assert (output / "train" / "A" / "000002.mp4").read_bytes() == b"validation-a"
    assert not (output / "train" / "B").exists()
    assert not (output / "processed_augmented").exists()
    paired = {record.sample_id: record for record in read_manifest(rgb_manifest)}
    assert paired["validation"].split == "validation"
    assert paired["validation"].video_path == "train/A/000002.mp4"

    report = json.loads((output / "_fetch_report.json").read_text(encoding="utf-8"))
    assert report["selection_strategy"] == "exact_manifest_canonical_pairs"
    assert report["classes"] == 1
    assert report["files"] == 3
    assert report["experimental_split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert report["physical_split_counts"] == {"test": 1, "train": 2}
    assert report["extraction"]["kept"] == 0
    assert report["extraction"]["extracted"] == 3
    assert report["extraction"]["range_requests"] == 1
    assert report["extraction"]["overfetch_bytes"] == 0

    ranges_after_first_run = _ArchiveStub.range_requests
    assert main(arguments) == 0
    resumed = json.loads((output / "_fetch_report.json").read_text(encoding="utf-8"))
    assert resumed["inventory_sha256"] == report["inventory_sha256"]
    assert resumed["extraction"]["kept"] == 3
    assert resumed["extraction"]["extracted"] == 0
    assert resumed["extraction"]["range_requests"] == 0
    # The second run still probes and reads the central directory, but performs no
    # member-extraction Range request after the CRC resume scan.
    assert _ArchiveStub.range_requests > ranges_after_first_run
    assert _ArchiveStub.download_requests == 2


def test_fetch_fails_instead_of_silently_dropping_missing_rgb(
    archive_server: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        (_record("missing", "test", "test", "A", "999999", 0),),
        manifest,
    )

    assert main(
        [
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "rgb"),
            "--download-url",
            archive_server,
        ]
    ) == 2
    assert "Missing canonical RGB counterpart" in capsys.readouterr().err
