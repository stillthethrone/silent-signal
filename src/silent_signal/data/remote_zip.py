"""Extract selected members of a remote ZIP through HTTP Range requests.

Only the central directory and byte ranges covering requested members are downloaded,
so a few gigabytes can be taken from a much larger archive without storing it. Callers
may fetch one range per member or coalesce adjacent records into bounded spans. Every
member is decompressed, CRC-checked and published atomically; reruns skip files whose
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


@dataclass(frozen=True, slots=True)
class RemoteZipDirectory:
    """Validated central-directory metadata for one remote ZIP archive."""

    members: tuple[zipfile.ZipInfo, ...]
    start_dir: int


@dataclass(frozen=True, slots=True)
class _SelectedInterval:
    item: zipfile.ZipInfo
    target: Path
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class _RangeSpan:
    start: int
    end: int
    intervals: tuple[_SelectedInterval, ...]

    @property
    def size(self) -> int:
        return self.end - self.start + 1


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


def inspect_directory(archive: RemoteArchive) -> RemoteZipDirectory:
    """Read and validate members plus the exact central-directory start offset."""

    with zipfile.ZipFile(_SequentialReader(archive)) as handle:
        members = tuple(handle.infolist())
        start_dir = handle.start_dir
    if not 0 <= start_dir <= archive.identity.size:
        raise RemoteZipError(f"Invalid ZIP central-directory offset: {start_dir}.")
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
        if not 0 <= item.header_offset < start_dir:
            raise RemoteZipError(
                f"ZIP member has an invalid local-header offset: {item.filename!r}"
            )
    return RemoteZipDirectory(members=members, start_dir=start_dir)


def list_members(archive: RemoteArchive) -> list[zipfile.ZipInfo]:
    """Read the central directory and reject unsafe or unsupported entries."""

    return list(inspect_directory(archive).members)


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


def extract_members_coalesced(
    archive: RemoteArchive,
    targets: Mapping[str, Path],
    directory: RemoteZipDirectory,
    *,
    workers: int = 2,
    reserve_bytes: int = 2 * 1024**3,
    max_span_bytes: int = 128 * 1024**2,
    max_gap_bytes: int = 4 * 1024**2,
    report: Callable[[str], None] = print,
) -> dict[str, int]:
    """Extract selected members using a bounded number of coalesced Range GETs.

    Local-record boundaries come from the next member's ``header_offset`` and the
    exact central-directory start returned by :func:`inspect_directory`.  This
    includes data descriptors without guessing their length.  Nearby pending
    records are fetched in one bounded byte span, then independently decompressed,
    CRC-checked and atomically published.
    """

    if workers < 1:
        raise ValueError("workers must be positive.")
    if max_span_bytes < 1:
        raise ValueError("max_span_bytes must be positive.")
    if max_gap_bytes < 0:
        raise ValueError("max_gap_bytes must be non-negative.")

    members = directory.members
    if not 0 <= directory.start_dir <= archive.identity.size:
        raise RemoteZipError(
            f"Invalid ZIP central-directory offset: {directory.start_dir}."
        )
    by_name = {item.filename: item for item in members}
    missing = sorted(set(targets) - by_name.keys())
    if missing:
        raise RemoteZipError(f"{len(missing)} requested members are absent; first: {missing[0]}")

    ordered = sorted(members, key=lambda item: item.header_offset)
    offsets = [item.header_offset for item in ordered]
    if len(offsets) != len(set(offsets)):
        raise RemoteZipError("ZIP members have duplicate local-header offsets.")
    end_by_offset = {
        item.header_offset: (
            ordered[position + 1].header_offset - 1
            if position + 1 < len(ordered)
            else directory.start_dir - 1
        )
        for position, item in enumerate(ordered)
    }

    intervals: list[_SelectedInterval] = []
    kept = 0
    needed = 0
    for name, target in targets.items():
        item = by_name[name]
        if item.is_dir():
            raise RemoteZipError(f"Cannot extract a directory as a file: {name}")
        if item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RemoteZipError(f"Unsupported compression for {name}: {item.compress_type}")
        if target.is_file() and _matches(target, item):
            kept += 1
            continue
        end = end_by_offset[item.header_offset]
        if end < item.header_offset:
            raise RemoteZipError(f"Invalid local-record boundary for {name}.")
        intervals.append(
            _SelectedInterval(
                item=item,
                target=target,
                start=item.header_offset,
                end=end,
            )
        )
        needed += item.file_size

    intervals.sort(key=lambda interval: interval.start)
    spans = _coalesced_spans(
        intervals,
        max_span_bytes=max_span_bytes,
        max_gap_bytes=max_gap_bytes,
    )
    if intervals:
        root = Path(os.path.commonpath([str(interval.target.parent) for interval in intervals]))
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < needed + reserve_bytes:
            required = (needed + reserve_bytes) / 1024**3
            raise RemoteZipError(f"Need {required:.1f} GiB free, have {free / 1024**3:.1f} GiB.")

    selected_record_bytes = sum(interval.size for interval in intervals)
    range_bytes = sum(span.size for span in spans)
    overfetch_bytes = range_bytes - selected_record_bytes
    report(
        f"[zip] kept {kept:,}; extracting {len(intervals):,} files in "
        f"{len(spans):,} ranges ({range_bytes / 1024**3:.2f} GiB; "
        f"overfetch {overfetch_bytes / 1024**2:.1f} MiB)"
    )

    done = 0
    step = max(1, len(intervals) // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_span, archive, span) for span in spans]
        for future in as_completed(futures):
            done += future.result()
            if done % step == 0 or done == len(intervals):
                report(f"[zip] extracted {done:,}/{len(intervals):,}")
    return {
        "kept": kept,
        "extracted": done,
        "range_requests": len(spans),
        "range_bytes": range_bytes,
        "selected_record_bytes": selected_record_bytes,
        "overfetch_bytes": overfetch_bytes,
    }


def _coalesced_spans(
    intervals: Sequence[_SelectedInterval],
    *,
    max_span_bytes: int,
    max_gap_bytes: int,
) -> tuple[_RangeSpan, ...]:
    spans: list[_RangeSpan] = []
    current: list[_SelectedInterval] = []
    start = end = 0
    for interval in intervals:
        if not current:
            current = [interval]
            start, end = interval.start, interval.end
            continue
        gap = interval.start - end - 1
        merged_size = interval.end - start + 1
        if gap <= max_gap_bytes and merged_size <= max_span_bytes:
            current.append(interval)
            end = interval.end
            continue
        spans.append(_RangeSpan(start=start, end=end, intervals=tuple(current)))
        current = [interval]
        start, end = interval.start, interval.end
    if current:
        spans.append(_RangeSpan(start=start, end=end, intervals=tuple(current)))
    return tuple(spans)


def _extract_span(archive: RemoteArchive, span: _RangeSpan) -> int:
    blob = archive.fetch(span.start, span.end)
    for interval in span.intervals:
        item = interval.item
        offset = item.header_offset - span.start
        if offset < 0 or offset + _LOCAL_HEADER.size > len(blob):
            raise RemoteZipError(f"Missing local header for {item.filename} in fetched span.")
        fields = _LOCAL_HEADER.unpack_from(blob, offset)
        if fields[0] != _LOCAL_SIGNATURE:
            raise RemoteZipError(f"Bad local header for {item.filename}.")
        if fields[2] & 0x1 or fields[3] != item.compress_type:
            raise RemoteZipError(f"Local header disagrees with metadata for {item.filename}.")
        data_start = offset + _LOCAL_HEADER.size + fields[9] + fields[10]
        data_end = data_start + item.compress_size
        if span.start + data_end > interval.end + 1:
            raise RemoteZipError(
                f"Compressed payload crosses the local-record boundary for {item.filename}."
            )
        compressed = blob[data_start:data_end]
        if len(compressed) != item.compress_size:
            raise RemoteZipError(f"Truncated data for {item.filename}.")
        try:
            if item.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(compressed, -zlib.MAX_WBITS)
            else:
                data = compressed
        except zlib.error as exc:
            raise RemoteZipError(f"Cannot decompress {item.filename}.") from exc
        if len(data) != item.file_size or zlib.crc32(data) & 0xFFFFFFFF != item.CRC:
            raise RemoteZipError(f"Size or CRC mismatch for {item.filename}.")
        _write_atomic(interval.target, data)
    return len(span.intervals)


def _write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        temporary.replace(target)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


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
