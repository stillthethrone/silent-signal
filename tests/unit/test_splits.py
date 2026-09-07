from __future__ import annotations

from silent_signal.configuration import DatasetConfig
from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import build_manifest
from silent_signal.data.splits import create_signer_disjoint_split


def test_split_is_deterministic_and_signer_disjoint(dataset_config: DatasetConfig) -> None:
    records = build_manifest(dataset_config).records
    arguments = {
        "ratios": dataset_config.split.ratios,
        "seed": dataset_config.split.seed,
        "search_trials": dataset_config.split.search_trials,
    }

    first_records, first = create_signer_disjoint_split(records, **arguments)
    second_records, second = create_signer_disjoint_split(records, **arguments)

    assert first == second
    assert first_records == second_records
    assert first.signer_counts == {"train": 6, "validation": 1, "test": 1}
    signer_sets = [set(first.signer_ids[name]) for name in ("train", "validation", "test")]
    assert not signer_sets[0] & signer_sets[1]
    assert not signer_sets[0] & signer_sets[2]
    assert not signer_sets[1] & signer_sets[2]

    splits_by_instance: dict[str, set[str | None]] = {}
    for record in first_records:
        splits_by_instance.setdefault(record.instance_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in splits_by_instance.values())


def test_explicit_official_allocation_takes_precedence(dataset_config: DatasetConfig) -> None:
    records = build_manifest(dataset_config).records
    allocation = {
        "train": ("001", "002", "003", "004", "005", "006"),
        "validation": ("007",),
        "test": ("008",),
    }

    _, definition = create_signer_disjoint_split(
        records,
        ratios=dataset_config.split.ratios,
        signer_ids=allocation,
    )

    assert definition.signer_ids == allocation


def test_twenty_eight_signers_allocate_as_twenty_two_three_three() -> None:
    records = tuple(
        ManifestRecord(
            sample_id=f"{signer:03d}_front",
            instance_id=f"{signer:03d}",
            video_id=f"{signer:06d}",
            signer_id=f"{signer:03d}",
            gloss_id="0",
            gloss_name="test",
            class_index=0,
            view="front",
            video_path=f"front_view/{signer:06d}.mp4",
        )
        for signer in range(1, 29)
    )

    _, definition = create_signer_disjoint_split(
        records,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        search_trials=1,
    )

    assert definition.signer_counts == {"train": 22, "validation": 3, "test": 3}
