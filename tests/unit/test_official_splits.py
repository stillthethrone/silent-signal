from __future__ import annotations

from dataclasses import replace

import pytest

from silent_signal.contracts import ManifestRecord
from silent_signal.data.splits import SplitError, verify_official_split


@pytest.fixture
def official_records() -> tuple[ManifestRecord, ...]:
    # Distinct raw signer strings must never be coalesced by numeric normalization.
    return tuple(
        ManifestRecord(
            sample_id=f"asl_citizen:{split}.mp4",
            instance_id=f"asl_citizen:{split}.mp4",
            video_id=f"{split}.mp4",
            signer_id=signer,
            gloss_id="HELLO",
            gloss_name="HELLO",
            class_index=0,
            view="single",
            video_path=f"videos/{split}.mp4",
            split=split,
            asl_lex_code="hello_1",
        )
        for split, signer in (("train", "7"), ("validation", "007"), ("test", "test-user"))
    )


def test_official_split_retains_membership_and_validation_results(
    official_records: tuple[ManifestRecord, ...],
) -> None:
    measured = tuple(replace(row, fps=29.97, width=640, height=480) for row in official_records)
    sources = {"train": {"path": "splits/train.csv", "sha256": "source-digest"}}
    actual, definition = verify_official_split(
        measured, source_records=official_records, source_files=sources
    )
    assert actual == measured
    assert definition.signer_ids["train"] == ("7",)
    assert definition.signer_ids["validation"] == ("007",)
    assert definition.seed is None
    assert definition.score is None
    assert definition.target_ratios == {}
    payload = definition.to_dict()
    assert payload["strategy"] == "official"
    assert payload["source_files"] == sources
    assert len(payload["assignments_sha256"]) == 64
    _, reordered = verify_official_split(tuple(reversed(measured)), source_records=official_records)
    assert reordered.assignments_sha256 == definition.assignments_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "test"),
        ("signer_id", "other"),
        ("class_index", 1),
        ("video_path", "videos/other.mp4"),
        ("gloss_name", "other"),
        ("asl_lex_code", "other"),
    ],
)
def test_official_split_rejects_tampered_source_fields(
    official_records: tuple[ManifestRecord, ...], field: str, value: object
) -> None:
    modified = (replace(official_records[0], **{field: value}), *official_records[1:])
    with pytest.raises(SplitError, match="differs from its official CSV"):
        verify_official_split(modified, source_records=official_records)


def test_official_split_rejects_missing_rows(official_records: tuple[ManifestRecord, ...]) -> None:
    with pytest.raises(SplitError, match="missing=1"):
        verify_official_split(official_records[:-1], source_records=official_records)


def test_official_split_rejects_duplicate_rows(
    official_records: tuple[ManifestRecord, ...],
) -> None:
    with pytest.raises(SplitError, match="Duplicate sample"):
        verify_official_split(
            (*official_records, official_records[0]), source_records=official_records
        )


def test_official_split_rejects_source_signer_leakage(
    official_records: tuple[ManifestRecord, ...],
) -> None:
    leaked = (replace(official_records[0], signer_id="test-user"), *official_records[1:])
    with pytest.raises(SplitError, match="Signer overlap"):
        verify_official_split(leaked, source_records=leaked)


def test_official_split_rejects_unseen_eval_class(
    official_records: tuple[ManifestRecord, ...],
) -> None:
    unseen = (*official_records[:-1], replace(official_records[-1], class_index=1))
    with pytest.raises(SplitError, match="absent from training"):
        verify_official_split(unseen, source_records=unseen)
