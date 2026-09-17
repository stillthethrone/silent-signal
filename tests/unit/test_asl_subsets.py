from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import ManifestError
from silent_signal.data.subsets import (
    ASLLexFrequency,
    read_asl_lex_frequencies,
    select_asl_citizen_top_classes,
)


def _record(
    gloss: str,
    class_index: int,
    code: str | None,
    split: str,
    suffix: str,
) -> ManifestRecord:
    return ManifestRecord(
        sample_id=f"asl_citizen:{suffix}",
        instance_id=f"asl_citizen:{suffix}",
        video_id=f"{suffix}.mp4",
        signer_id=f"signer-{split}",
        gloss_id=gloss,
        gloss_name=gloss,
        class_index=class_index,
        view="single",
        video_path=f"videos/{suffix}.mp4",
        split=split,
        asl_lex_code=code,
    )


def _frequency(code: str, value: float) -> ASLLexFrequency:
    return ASLLexFrequency(
        code=code,
        frequency=value,
        entry_id=f"entry-{code}",
        lemma_id=f"lemma-{code}",
    )


def test_selects_by_frequency_remaps_classes_and_preserves_splits() -> None:
    records = (
        _record("MEDIUM", 10, "code-m", "train", "m-train"),
        _record("HIGH", 22, "shared", "train", "h-train"),
        _record("HIGH", 22, "shared", "validation", "h-val"),
        _record("ALSO_HIGH", 31, "shared", "train", "a-train"),
        _record("LOW", 45, "code-l", "train", "l-train"),
        _record("NO_CODE", 90, None, "train", "n-train"),
    )
    frequencies = {
        "code-m": _frequency("code-m", 4.5),
        "shared": _frequency("shared", 6.75),
        "code-l": _frequency("code-l", 2.0),
    }

    result = select_asl_citizen_top_classes(records, frequencies, class_count=3)

    assert [item.gloss_name for item in result.classes] == ["ALSO_HIGH", "HIGH", "MEDIUM"]
    assert [item.source_class_index for item in result.classes] == [31, 22, 10]
    assert [item.subset_class_index for item in result.classes] == [0, 1, 2]
    assert [label.class_index for label in result.labels] == [0, 1, 2]
    assert {item.gloss_name: item.class_index for item in result.records} == {
        "MEDIUM": 2,
        "HIGH": 1,
        "ALSO_HIGH": 0,
    }
    assert {item.split for item in result.records if item.gloss_name == "HIGH"} == {
        "train",
        "validation",
    }
    assert result.classes[1].split_clip_counts == {"train": 1, "validation": 1}
    assert result.excluded_class_counts == {"missing_asl_lex_code": 1}
    report = result.report()
    assert report["unique_asl_lex_codes_selected"] == 2
    assert report["shared_asl_lex_codes"] == {"shared": ["ALSO_HIGH", "HIGH"]}


def test_tie_break_does_not_use_dataset_clip_count() -> None:
    records = (
        _record("ZEBRA", 0, "z", "train", "z-1"),
        _record("ZEBRA", 0, "z", "test", "z-2"),
        _record("APPLE", 1, "a", "train", "a-1"),
    )
    frequencies = {"z": _frequency("z", 5.0), "a": _frequency("a", 5.0)}

    result = select_asl_citizen_top_classes(records, frequencies, class_count=2)

    assert [item.gloss_name for item in result.classes] == ["APPLE", "ZEBRA"]


def test_excludes_unmapped_and_trainless_classes() -> None:
    records = (
        _record("GOOD", 0, "good", "train", "good"),
        _record("UNKNOWN", 1, "unknown", "train", "unknown"),
        _record("EVAL_ONLY", 2, "eval", "test", "eval"),
    )
    frequencies = {"good": _frequency("good", 6.0), "eval": _frequency("eval", 7.0)}

    result = select_asl_citizen_top_classes(records, frequencies, class_count=1)

    assert [item.gloss_name for item in result.classes] == ["GOOD"]
    assert result.excluded_class_counts == {"code_not_in_asl_lex": 1, "no_train_clips": 1}


def test_rejects_insufficient_eligible_classes() -> None:
    records = (_record("ONLY", 0, "only", "train", "only"),)

    with pytest.raises(ManifestError, match="Requested 2 classes, but only 1"):
        select_asl_citizen_top_classes(records, {"only": _frequency("only", 6.0)}, class_count=2)


def test_rejects_conflicting_class_metadata() -> None:
    first = _record("WORD", 0, "one", "train", "one")
    records = (first, replace(first, sample_id="two", video_id="two.mp4", asl_lex_code="two"))

    with pytest.raises(ManifestError, match="conflicting ASL-LEX codes"):
        select_asl_citizen_top_classes(
            records, {"one": _frequency("one", 5.0), "two": _frequency("two", 4.0)}
        )


def _write_asl_lex(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", encoding="latin-1", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EntryID", "LemmaID", "Code", "SignFrequency(M)"])
        writer.writerows(rows)


def test_reads_published_latin1_columns(tmp_path: Path) -> None:
    path = tmp_path / "signdata.csv"
    _write_asl_lex(path, [("café", "lemma", "A_01_001", "5.143")])

    result = read_asl_lex_frequencies(path)

    assert result["A_01_001"] == ASLLexFrequency(
        code="A_01_001", frequency=5.143, entry_id="café", lemma_id="lemma"
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [("one", "one", "duplicate", "5"), ("two", "two", "duplicate", "6")],
            "Duplicate ASL-LEX Code",
        ),
        ([("one", "one", "bad", "8")], "outside the 1-7 scale"),
        ([("one", "one", "bad", "not-a-number")], "invalid frequency"),
        ([("one", "one", "bad", "")], "has no SignFrequency"),
    ],
)
def test_rejects_invalid_frequency_data(
    tmp_path: Path, rows: list[tuple[str, str, str, str]], message: str
) -> None:
    path = tmp_path / "signdata.csv"
    _write_asl_lex(path, rows)

    with pytest.raises(ManifestError, match=message):
        read_asl_lex_frequencies(path)


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "signdata.csv"
    path.write_text("Code\nA_01_001\n", encoding="latin-1")

    with pytest.raises(ManifestError, match="missing required columns"):
        read_asl_lex_frequencies(path)
