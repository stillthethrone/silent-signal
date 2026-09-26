"""Pair the canonical cropped Kaggle VSL videos with a pose manifest."""

from __future__ import annotations

import re
import unicodedata
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import ManifestError
from silent_signal.data.remote_zip import RemoteZipError

CANONICAL_RGB_DIRECTORY = "processed/processed/frame_splited"
_RGB_MEMBER = re.compile(
    r"(?:^|/)processed/processed/frame_splited/"
    r"(train|test)/([^/]+)/([^/]+\.mp4)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RGBMember:
    """One canonical cropped RGB clip inside the Kaggle ZIP."""

    source_split: str
    gloss: str
    filename: str
    member: str
    file_size: int
    crc32: int


@dataclass(frozen=True, slots=True)
class RGBPair:
    """A strict one-to-one mapping between a manifest sample and one RGB member."""

    sample_id: str
    source_split: str
    gloss: str
    video_id: str
    member: RGBMember
    relative_path: str


def locate_canonical_rgb(members: Sequence[zipfile.ZipInfo]) -> tuple[RGBMember, ...]:
    """Return only canonical ``frame_splited`` MP4 entries, never augmented videos."""

    found: list[RGBMember] = []
    for item in members:
        path = PurePosixPath(item.filename)
        if item.is_dir() or "processed_augmented" in path.parts:
            continue
        match = _RGB_MEMBER.search(item.filename)
        if not match:
            continue
        found.append(
            RGBMember(
                source_split=match[1].lower(),
                gloss=unicodedata.normalize("NFC", match[2]),
                filename=match[3],
                member=item.filename,
                file_size=item.file_size,
                crc32=item.CRC,
            )
        )
    if not found:
        raise RemoteZipError(
            "No canonical processed/processed/frame_splited MP4 files in the archive."
        )
    return tuple(found)


def pair_manifest_rgb(
    records: Sequence[ManifestRecord],
    members: Sequence[RGBMember],
) -> tuple[RGBPair, ...]:
    """Join pose rows to RGB clips by their physical source split, gloss and stem.

    Validation rows are sampled from the uploader's official ``train`` directory.  Their
    experimental split is therefore ``validation`` while their physical source split remains
    ``train``.  The latter is intentionally derived from ``video_path`` rather than
    ``record.split``.
    """

    if not records:
        raise ManifestError("The manifest contains no records to pair with RGB videos.")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError("The manifest contains duplicate sample_id values.")

    by_key: dict[tuple[str, str, str], RGBMember] = {}
    for member in members:
        key = _member_key(member)
        previous = by_key.get(key)
        if previous is not None:
            raise RemoteZipError(
                "Canonical RGB members collide after Unicode/case normalization: "
                f"{previous.member!r}, {member.member!r}"
            )
        by_key[key] = member

    pairs: list[RGBPair] = []
    used_members: set[str] = set()
    missing: list[str] = []
    duplicate_sources: list[str] = []
    for record in sorted(records, key=lambda item: item.sample_id):
        source_split, gloss, video_id = _manifest_source(record)
        member = by_key.get((source_split, _gloss_key(gloss), _stem_key(video_id)))
        if member is None:
            missing.append(
                f"{record.sample_id} ({source_split}/{gloss}/{video_id}.mp4)"
            )
            continue
        if member.member in used_members:
            duplicate_sources.append(member.member)
            continue
        used_members.add(member.member)
        # Normalize the local suffix so the pose-manifest ``.npy -> .mp4`` derivation
        # is portable on Colab's case-sensitive filesystem even if an archive uses
        # an upper-case ``.MP4`` member name.
        local_filename = f"{PurePosixPath(member.filename).stem}.mp4"
        relative = PurePosixPath(
            member.source_split, member.gloss, local_filename
        ).as_posix()
        pairs.append(
            RGBPair(
                sample_id=record.sample_id,
                source_split=source_split,
                gloss=member.gloss,
                video_id=video_id,
                member=member,
                relative_path=relative,
            )
        )
    if missing:
        suffix = "" if len(missing) == 1 else f" (and {len(missing) - 1} more)"
        raise RemoteZipError(
            f"Missing canonical RGB counterpart for {len(missing)} manifest samples; "
            f"first: {missing[0]}{suffix}"
        )
    if duplicate_sources:
        raise ManifestError(
            "Multiple manifest rows resolve to one canonical RGB member; first: "
            f"{duplicate_sources[0]}"
        )
    if len(pairs) != len(records):
        raise RemoteZipError(f"Paired {len(pairs)} RGB clips for {len(records)} manifest rows.")
    return tuple(pairs)


def rgb_manifest_records(
    records: Sequence[ManifestRecord], pairs: Sequence[RGBPair]
) -> tuple[ManifestRecord, ...]:
    """Return a video manifest that preserves labels/splits and replaces only source metadata."""

    pair_by_sample = {pair.sample_id: pair for pair in pairs}
    if len(pair_by_sample) != len(pairs):
        raise ManifestError("RGB pairs contain duplicate sample_id values.")
    missing = [record.sample_id for record in records if record.sample_id not in pair_by_sample]
    if missing:
        raise ManifestError(
            f"RGB pairs are missing {len(missing)} manifest samples; first: {missing[0]}"
        )
    return tuple(
        replace(
            record,
            video_path=pair_by_sample[record.sample_id].relative_path,
            file_size_bytes=pair_by_sample[record.sample_id].member.file_size,
        )
        for record in records
    )


def extraction_targets(pairs: Sequence[RGBPair], output_root: Path) -> dict[str, Path]:
    """Map remote ZIP member names to safe local paths below ``output_root``."""

    root = output_root.resolve()
    targets: dict[str, Path] = {}
    destinations: set[Path] = set()
    for pair in pairs:
        destination = (root / Path(*PurePosixPath(pair.relative_path).parts)).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise RemoteZipError(f"RGB destination escapes output root: {destination}") from exc
        if destination in destinations:
            raise RemoteZipError(f"Duplicate RGB destination: {destination}")
        destinations.add(destination)
        targets[pair.member.member] = destination
    if not targets:
        raise RemoteZipError("No RGB members were selected for extraction.")
    return targets


def _manifest_source(record: ManifestRecord) -> tuple[str, str, str]:
    if record.view != "front":
        raise ManifestError(
            f"Kaggle cropped RGB pairing supports only the front view, found {record.view!r}."
        )
    path = PurePosixPath(record.video_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ManifestError(
            "Expected a canonical keypoint path split/gloss/file.npy, found "
            f"{record.video_path!r}."
        )
    source_split, gloss, filename = path.parts
    source_split = source_split.lower()
    if source_split not in {"train", "test"} or path.suffix.casefold() != ".npy":
        raise ManifestError(
            "Expected a canonical train/test keypoint .npy path, found "
            f"{record.video_path!r}."
        )
    return source_split, unicodedata.normalize("NFC", gloss), path.stem


def _member_key(member: RGBMember) -> tuple[str, str, str]:
    return member.source_split, _gloss_key(member.gloss), _stem_key(Path(member.filename).stem)


def _gloss_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _stem_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()
