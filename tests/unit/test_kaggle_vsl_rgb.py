from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path

import pytest

from silent_signal.contracts import ManifestRecord
from silent_signal.data.kaggle_vsl_rgb import (
    RGBMember,
    extraction_targets,
    locate_canonical_rgb,
    pair_manifest_rgb,
    rgb_manifest_records,
)
from silent_signal.data.manifest import ManifestError
from silent_signal.data.remote_zip import RemoteZipError


def _record(
    sample_id: str,
    video_path: str,
    *,
    split: str,
    class_index: int = 0,
    gloss: str = "Cảm ơn",
) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=Path(video_path).stem,
        signer_id="unknown",
        gloss_id=f"kvsl:{gloss}",
        gloss_name=gloss,
        class_index=class_index,
        view="front",
        video_path=video_path,
        split=split,
    )


def _zip_info(name: str, *, size: int = 7, crc32: int = 11) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.file_size = size
    info.CRC = crc32
    return info


def test_locates_only_canonical_frame_splited_mp4_members() -> None:
    members = [
        _zip_info("processed/processed/frame_splited/train/Cảm ơn/000001.mp4"),
        _zip_info("prefix/processed/processed/frame_splited/test/Cảm ơn/000002.MP4"),
        _zip_info(
            "processed_augmented/processed_augmented/frame_splited/train/"
            "Cảm ơn/extra.mp4"
        ),
        _zip_info("processed/processed/keypoints_splited/train/Cảm ơn/000001.npy"),
    ]

    found = locate_canonical_rgb(members)

    assert [(item.source_split, item.gloss, item.filename) for item in found] == [
        ("train", "Cảm ơn", "000001.mp4"),
        ("test", "Cảm ơn", "000002.MP4"),
    ]


def test_pairs_validation_with_physical_train_video_and_preserves_contract(
    tmp_path: Path,
) -> None:
    records = (
        _record("sample-train", "train/Cảm ơn/000001.npy", split="train"),
        _record("sample-validation", "train/Cảm ơn/000002.npy", split="validation"),
        _record("sample-test", "test/Cảm ơn/000003.npy", split="test"),
    )
    members = tuple(
        RGBMember(
            source_split=source_split,
            gloss="Cảm ơn",
            filename=f"{index:06d}.mp4",
            member=(
                "processed/processed/frame_splited/"
                f"{source_split}/Cảm ơn/{index:06d}.mp4"
            ),
            file_size=index,
            crc32=index + 100,
        )
        for source_split, index in (("train", 1), ("train", 2), ("test", 3))
    )

    pairs = pair_manifest_rgb(records, members)
    paired = {pair.sample_id: pair for pair in pairs}

    assert paired["sample-validation"].source_split == "train"
    assert paired["sample-validation"].relative_path == "train/Cảm ơn/000002.mp4"
    rgb_records = rgb_manifest_records(records, pairs)
    assert [(item.sample_id, item.split) for item in rgb_records] == [
        ("sample-train", "train"),
        ("sample-validation", "validation"),
        ("sample-test", "test"),
    ]
    assert rgb_records[1].video_path == "train/Cảm ơn/000002.mp4"
    assert rgb_records[1].file_size_bytes == 2
    targets = extraction_targets(pairs, tmp_path / "rgb")
    assert targets[members[1].member] == (
        tmp_path / "rgb" / "train" / "Cảm ơn" / "000002.mp4"
    ).resolve()


def test_pairing_rejects_missing_and_unicode_equivalent_duplicate_members() -> None:
    record = _record("sample", "train/Cảm ơn/000001.npy", split="train")
    with pytest.raises(RemoteZipError, match="Missing canonical RGB counterpart"):
        pair_manifest_rgb((record,), ())

    decomposed = unicodedata.normalize("NFD", "Cảm ơn")
    members = (
        RGBMember("train", "Cảm ơn", "000001.mp4", "first", 1, 1),
        RGBMember("train", decomposed, "000001.mp4", "second", 1, 1),
    )
    with pytest.raises(RemoteZipError, match="collide"):
        pair_manifest_rgb((record,), members)


def test_pairing_rejects_noncanonical_pose_paths_and_views() -> None:
    member = RGBMember("train", "Cảm ơn", "000001.mp4", "member", 1, 1)
    invalid_path = _record("bad-path", "validation/Cảm ơn/000001.npy", split="validation")
    with pytest.raises(ManifestError, match="canonical train/test"):
        pair_manifest_rgb((invalid_path,), (member,))

    side = _record("side", "train/Cảm ơn/000001.npy", split="train")
    side = ManifestRecord.from_dict({**side.to_dict(), "view": "left"})
    with pytest.raises(ManifestError, match="front view"):
        pair_manifest_rgb((side,), (member,))


def test_pairing_normalizes_uppercase_video_suffix_for_local_colab_path() -> None:
    record = _record("sample", "test/Cảm ơn/000001.npy", split="test")
    member = RGBMember(
        "test",
        "Cảm ơn",
        "000001.MP4",
        "processed/processed/frame_splited/test/Cảm ơn/000001.MP4",
        10,
        20,
    )

    pair = pair_manifest_rgb((record,), (member,))[0]

    assert pair.relative_path == "test/Cảm ơn/000001.mp4"
