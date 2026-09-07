from __future__ import annotations

from dataclasses import replace

from silent_signal.configuration import DatasetConfig
from silent_signal.contracts import ValidationLevel
from silent_signal.data.manifest import build_manifest
from silent_signal.data.validation import (
    MediaInfo,
    media_info_from_ffprobe,
    validate_manifest,
)


def test_metadata_validation_passes_complete_fixture(dataset_config: DatasetConfig) -> None:
    records = build_manifest(dataset_config).records

    result = validate_manifest(
        records,
        dataset_root=dataset_config.root,
        expected=dataset_config.expected,
        expected_views=tuple(dataset_config.views),
    )

    assert not result.has_errors
    assert all(record.is_valid for record in result.records)
    assert all(record.file_size_bytes == len(b"synthetic-video") for record in result.records)


def test_detects_missing_video_and_incomplete_instance(dataset_config: DatasetConfig) -> None:
    records = build_manifest(dataset_config).records
    missing_path = dataset_config.root / records[0].video_path
    missing_path.unlink()
    incomplete_records = tuple(
        record
        for record in records
        if not (record.instance_id == records[-1].instance_id and record.view == "right")
    )

    result = validate_manifest(
        incomplete_records,
        dataset_root=dataset_config.root,
        expected=dataset_config.expected,
        expected_views=tuple(dataset_config.views),
    )
    codes = {issue.code for issue in result.issues}

    assert "video_missing" in codes
    assert "incomplete_multiview_instance" in codes
    assert "unexpected_clips_count" in codes


def test_probe_mode_accepts_injected_media_probe(dataset_config: DatasetConfig) -> None:
    records = build_manifest(dataset_config).records

    result = validate_manifest(
        records,
        dataset_root=dataset_config.root,
        expected=dataset_config.expected,
        expected_views=tuple(dataset_config.views),
        level=ValidationLevel.PROBE,
        workers=2,
        probe_function=lambda _: MediaInfo(25, 1.0, 25.0, 1080, 1080, "h264"),
    )

    assert not result.has_errors
    assert {record.codec for record in result.records} == {"h264"}


def test_revalidation_recomputes_stale_record_errors(dataset_config: DatasetConfig) -> None:
    records = list(build_manifest(dataset_config).records)
    records[0] = replace(
        records[0],
        is_valid=False,
        validation_errors=("video_missing",),
    )

    result = validate_manifest(
        records,
        dataset_root=dataset_config.root,
        expected=dataset_config.expected,
        expected_views=tuple(dataset_config.views),
    )

    assert not result.has_errors
    assert result.records[0].is_valid
    assert result.records[0].validation_errors == ()


def test_parses_ffprobe_fraction_and_format_duration() -> None:
    media = media_info_from_ffprobe(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1080,
                    "height": 1080,
                    "avg_frame_rate": "25/1",
                    "nb_frames": "50",
                }
            ],
            "format": {"duration": "2.0"},
        }
    )

    assert media == MediaInfo(50, 2.0, 25.0, 1080, 1080, "h264")
