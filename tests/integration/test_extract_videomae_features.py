from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from silent_signal.cli.extract_videomae_features import extract_videomae_features
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import write_manifest
from silent_signal.data.rgb_feature_pack import read_rgb_feature_pack


def _record(sample_id: str, class_index: int, split: str) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        instance_id=sample_id,
        video_id=sample_id,
        signer_id="unknown",
        gloss_id=f"gloss-{class_index}",
        gloss_name=f"Gloss {class_index}",
        class_index=class_index,
        view="front",
        video_path=f"{split}/Gloss {class_index}/{sample_id}.npy",
        split=split,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[ManifestRecord]]:
    records = [
        _record("sample-c", 1, "test"),
        _record("sample-a", 0, "train"),
        _record("sample-b", 1, "validation"),
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    videos = tmp_path / "videos"
    for record in records:
        path = videos / Path(record.video_path).with_suffix(".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    return manifest, videos, records


def _features(paths: list[Path] | tuple[Path, ...]) -> np.ndarray:
    output = np.empty((len(paths), 8, 768), dtype=np.float16)
    for index, path in enumerate(paths):
        output[index].fill(sum(path.stem.encode("utf-8")) % 97)
    return output


def test_extractor_writes_resumable_shards_and_reuses_complete_pack_without_videos(
    tmp_path: Path,
) -> None:
    manifest, videos, _ = _fixture(tmp_path)
    output = tmp_path / "rgb_features.npz"
    report = tmp_path / "rgb_features.report.json"
    shards = tmp_path / "shards"

    first = extract_videomae_features(
        manifest_path=manifest,
        dataset_root=videos,
        output_path=output,
        report_path=report,
        shard_root=shards,
        batch_size=2,
        shard_size=2,
        extract_batch=_features,
        log=lambda _: None,
    )
    packed = read_rgb_feature_pack(output)

    assert first["samples"] == 3
    assert first["extracted_shards"] == 2
    assert len(list(shards.glob("shard_*.npz"))) == 2
    assert packed.sample_ids == ("sample-a", "sample-b", "sample-c")
    assert packed.features.shape == (3, 8, 768)

    shutil.rmtree(videos)

    def forbidden(_: list[Path] | tuple[Path, ...]) -> np.ndarray:
        raise AssertionError("A complete pack must be reused before videos or model are needed.")

    reused = extract_videomae_features(
        manifest_path=manifest,
        dataset_root=videos,
        output_path=output,
        report_path=report,
        shard_root=shards,
        extract_batch=forbidden,
        log=lambda _: None,
    )

    assert reused == first


def test_extractor_resumes_completed_shards_after_interruption(tmp_path: Path) -> None:
    manifest, videos, _ = _fixture(tmp_path)
    output = tmp_path / "rgb_features.npz"
    report = tmp_path / "rgb_features.report.json"
    shards = tmp_path / "shards"
    calls = 0

    def interrupted(paths: list[Path] | tuple[Path, ...]) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return _features(paths)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        extract_videomae_features(
            manifest_path=manifest,
            dataset_root=videos,
            output_path=output,
            report_path=report,
            shard_root=shards,
            batch_size=1,
            shard_size=1,
            extract_batch=interrupted,
            log=lambda _: None,
        )

    assert (shards / "shard_00000.npz").is_file()
    resumed_paths: list[str] = []

    def resumed(paths: list[Path] | tuple[Path, ...]) -> np.ndarray:
        resumed_paths.extend(path.stem for path in paths)
        return _features(paths)

    completed = extract_videomae_features(
        manifest_path=manifest,
        dataset_root=videos,
        output_path=output,
        report_path=report,
        shard_root=shards,
        batch_size=1,
        shard_size=1,
        extract_batch=resumed,
        log=lambda _: None,
    )

    assert completed["resumed_shards"] == 1
    assert completed["extracted_shards"] == 2
    assert resumed_paths == ["sample-b", "sample-c"]
