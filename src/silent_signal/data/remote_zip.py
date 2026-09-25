"""Extract selected members of a remote ZIP through HTTP Range requests.

Only the central directory and the byte ranges of the requested members are downloaded,
so a few gigabytes can be taken from a much larger archive without storing it. Each
member is decompressed, CRC-checked and published atomically; a rerun skips files whose
size and CRC already match. No network request runs on import.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import stat
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

Resolver = Callable[[], tuple[str, dict[str, str]]]

_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIGNATURE = 0x04034B50
_LOCAL_EXTRA_MARGIN = 1024
_EXPIRED_STATUSES = frozenset({400, 401, 403, 410})
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_COPY_CHUNK = 8 * 1024**2


class RemoteZipError(RuntimeError):
    """A remote archive cannot be read or extracted without losing integrity."""


@dataclass(frozen=True, slots=True)
class RemoteIdentity:
    """Size and validator that pin one immutable remote archive."""

    size: int
    etag: str | None


class RemoteArchive:
    """Byte-range access to one remote file whose signed URL may expire and be renewed."""

    def __init__(self, resolve: Resolver, *, retries: int = 5, timeout: float = 120.0) -> None:
        self._resolve = resolve
        self._retries = retries
        self._timeout = timeout
        self._url, self._headers = resolve()
        self.identity = self._probe()

    def fetch(self, start: int, end: int) -> bytes:
        """Return bytes ``start..end`` inclusive, renewing the URL once if it expired."""

        if start < 0 or end < start or end >= self.identity.size:
            raise RemoteZipError(f"Invalid range {start}-{end} for {self.identity.size} bytes.")
        renewed = False
        for attempt in range(self._retries + 1):
            try:
                data, total, etag = self._get(start, end)
            except urllib.error.HTTPError as error:
                if error.code in _EXPIRED_STATUSES and not renewed:
                    self._url, self._headers = self._resolve()
                    renewed = True
                    continue
                if error.code not in _RETRY_STATUSES or attempt == self._retries:
                    raise RemoteZipError(f"HTTP {error.code} while reading the archive.") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == self._retries:
                    raise RemoteZipError(
                        f"Network error while reading the archive: {error}"
                    ) from error
            else:
                if total != self.identity.size or (
                    etag and self.identity.etag and etag != self.identity.etag
                ):
                    raise RemoteZipError("The remote archive changed while it was being read.")
                return data
            time.sleep(min(60.0, 2.0 * 2**attempt))
        raise RemoteZipError("Remote range request failed after retries.")

    def _probe(self) -> RemoteIdentity:
        _data, total, etag = self._get(0, 0)
        return RemoteIdentity(size=total, etag=etag)

    def _get(self, start: int, end: int) -> tuple[bytes, int, str | None]:
        request = urllib.request.Request(
            self._url,
            headers={
                **self._headers,
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            if response.status != 206:
                raise RemoteZipError("The server does not support HTTP Range requests.")
            match = _RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[2])) != (start, end):
                raise RemoteZipError("The server returned an unexpected Content-Range.")
            data = response.read(end - start + 2)
            etag = response.headers.get("ETag")
        if len(data) != end - start + 1:
            raise RemoteZipError("The server returned an incomplete range.")
        return data, int(match[3]), etag


class _SequentialReader(io.RawIOBase):
    """Cached, seekable view used by ``zipfile`` to read the central directory."""

    def __init__(self, archive: RemoteArchive, *, chunk_size: int = 8 * 1024**2) -> None:
        super().__init__()
        self._archive = archive
        self._chunk = chunk_size
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self._position,
            io.SEEK_END: self._archive.identity.size,
        }
        self._position = base[whence] + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        total = self._archive.identity.size
        if size < 0 or self._position + size > total:
            size = max(0, total - self._position)
        output = bytearray()
        while size:
            index = self._position // self._chunk
            block = self._block(index)
            offset = self._position - index * self._chunk
            piece = block[offset : offset + size]
            output += piece
            self._position += len(piece)
            size -= len(piece)
        return bytes(output)

    def _block(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * self._chunk
            end = min(self._archive.identity.size, start + self._chunk) - 1
            self._cache[index] = self._archive.fetch(start, end)
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]


def list_members(archive: RemoteArchive) -> list[zipfile.ZipInfo]:
    """Read the central directory and reject unsafe or unsupported entries."""

    with zipfile.ZipFile(_SequentialReader(archive)) as handle:
        members = handle.infolist()
    for item in members:
        path = PurePosixPath(item.filename)
        file_type = stat.S_IFMT(item.external_attr >> 16)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or "\x00" in item.filename
            or item.flag_bits & 0x1
            or file_type not in (0, stat.S_IFREG, stat.S_IFDIR)
        ):
            raise RemoteZipError(f"Unsafe or encrypted ZIP member: {item.filename!r}")
    return members


def extract_members(
    archive: RemoteArchive,
    targets: Mapping[str, Path],
    members: Sequence[zipfile.ZipInfo],
    *,
    workers: int = 8,
    reserve_bytes: int = 2 * 1024**3,
    report: Callable[[str], None] = print,
) -> dict[str, int]:
    """Write each requested member to its target path; returns counts per outcome."""

    by_name = {item.filename: item for item in members}
    missing = sorted(set(targets) - by_name.keys())
    if missing:
        raise RemoteZipError(f"{len(missing)} requested members are absent; first: {missing[0]}")
    pending: list[tuple[zipfile.ZipInfo, Path]] = []
    kept = 0
    for name, target in targets.items():
        item = by_name[name]
        if item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RemoteZipError(f"Unsupported compression for {name}: {item.compress_type}")
        if target.is_file() and _matches(target, item):
            kept += 1
        else:
            pending.append((item, target))
    pending.sort(key=lambda pair: pair[0].header_offset)
    needed = sum(item.file_size for item, _ in pending)
    if pending:
        root = Path(os.path.commonpath([str(target.parent) for _, target in pending]))
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < needed + reserve_bytes:
            required = (needed + reserve_bytes) / 1024**3
            raise RemoteZipError(f"Need {required:.1f} GiB free, have {free / 1024**3:.1f} GiB.")
    report(f"[zip] kept {kept:,}; extracting {len(pending):,} files ({needed / 1024**3:.2f} GiB)")
    done = 0
    step = max(1, len(pending) // 20)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_extract_one, archive, item, target) for item, target in pending]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % step == 0 or done == len(pending):
                report(f"[zip] extracted {done:,}/{len(pending):,}")
    return {"kept": kept, "extracted": done}


def _extract_one(archive: RemoteArchive, item: zipfile.ZipInfo, target: Path) -> None:
    # The local extra field may differ from the central one; fetch a small margin and
    # re-read the exact span only when the local header turns out to be longer.
    last = archive.identity.size - 1
    fixed = _LOCAL_HEADER.size + len(item.filename.encode("utf-8"))
    end = min(last, item.header_offset + fixed + _LOCAL_EXTRA_MARGIN + item.compress_size - 1)
    blob = archive.fetch(item.header_offset, end)
    fields = _LOCAL_HEADER.unpack_from(blob)
    if fields[0] != _LOCAL_SIGNATURE:
        raise RemoteZipError(f"Bad local header for {item.filename}.")
    start = _LOCAL_HEADER.size + fields[9] + fields[10]
    if start + item.compress_size > len(blob):
        exact_end = min(last, item.header_offset + start + item.compress_size - 1)
        blob = archive.fetch(item.header_offset, exact_end)
    compressed = blob[start : start + item.compress_size]
    if len(compressed) != item.compress_size:
        raise RemoteZipError(f"Truncated data for {item.filename}.")
    if item.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        data = compressed
    if len(data) != item.file_size or zlib.crc32(data) & 0xFFFFFFFF != item.CRC:
        raise RemoteZipError(f"Size or CRC mismatch for {item.filename}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(target)


def _matches(path: Path, item: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != item.file_size:
        return False
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == item.CRC
