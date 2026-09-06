"""Deterministic signer-disjoint splits for VSL400."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from silent_signal.contracts import ManifestRecord, SplitDefinition, SplitName

_SPLITS = tuple(item.value for item in SplitName)


class SplitError(ValueError):
    """Raised when a valid signer-disjoint split cannot be created."""


@dataclass(slots=True)
class _SignerStats:
    instance_ids: set[str]
    clip_count: int
    gloss_counts: Counter[int]


def create_signer_disjoint_split(
    records: Sequence[ManifestRecord],
    *,
    ratios: Mapping[str, float],
    seed: int = 42,
    search_trials: int = 5000,
    signer_ids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[ManifestRecord, ...], SplitDefinition]:
    """Assign whole signers to splits while balancing size and gloss distribution."""

    normalized_ratios = _validate_ratios(ratios)
    if not records:
        raise SplitError("Cannot split an empty manifest.")
    stats = _aggregate_signers(records)
    all_signers = tuple(sorted(stats))
    if len(all_signers) < len(_SPLITS):
        raise SplitError("At least three signers are required for train/validation/test.")
    _validate_instance_ownership(records)

    if signer_ids is None:
        if search_trials < 1:
            raise SplitError("search_trials must be at least 1.")
        allocation = _search_allocation(
            stats,
            normalized_ratios,
            seed=seed,
            search_trials=search_trials,
        )
    else:
        allocation = _validate_explicit_allocation(signer_ids, all_signers)

    split_by_signer = {signer: split for split, signers in allocation.items() for signer in signers}
    assigned = tuple(replace(record, split=split_by_signer[record.signer_id]) for record in records)
    score = _score_allocation(allocation, stats, normalized_ratios)
    definition = _make_definition(
        assigned,
        allocation,
        normalized_ratios,
        seed=seed,
        score=score,
    )
    return assigned, definition


def load_signer_allocation(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load an official or previously generated signer split JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SplitError(f"Split file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(f"Invalid split JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SplitError("Split JSON root must be an object.")
    candidate = payload.get("signer_ids", payload)
    if not isinstance(candidate, dict):
        raise SplitError("Split JSON must contain a signer_ids object.")
    result: dict[str, tuple[str, ...]] = {}
    for split in _SPLITS:
        values = candidate.get(split)
        if not isinstance(values, list) or not all(
            isinstance(value, (str, int)) for value in values
        ):
            raise SplitError(f"signer_ids.{split} must be a list of signer ids.")
        result[split] = tuple(sorted(_normalize_signer(value) for value in values))
    return result


def write_split_definition(definition: SplitDefinition, path: str | Path) -> None:
    """Write a reproducible split artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(definition.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _aggregate_signers(records: Sequence[ManifestRecord]) -> dict[str, _SignerStats]:
    stats: dict[str, _SignerStats] = {}
    for record in records:
        signer = stats.setdefault(
            record.signer_id,
            _SignerStats(instance_ids=set(), clip_count=0, gloss_counts=Counter()),
        )
        signer.instance_ids.add(record.instance_id)
        signer.clip_count += 1
        signer.gloss_counts[record.class_index] += 1
    return stats


def _validate_instance_ownership(records: Sequence[ManifestRecord]) -> None:
    signers_by_instance: dict[str, set[str]] = defaultdict(set)
    for record in records:
        signers_by_instance[record.instance_id].add(record.signer_id)
    invalid = {
        instance: signers for instance, signers in signers_by_instance.items() if len(signers) != 1
    }
    if invalid:
        example, signers = next(iter(invalid.items()))
        raise SplitError(f"Instance {example!r} belongs to multiple signers: {sorted(signers)}.")


def _search_allocation(
    stats: Mapping[str, _SignerStats],
    ratios: dict[str, float],
    *,
    seed: int,
    search_trials: int,
) -> dict[str, tuple[str, ...]]:
    signers = sorted(stats)
    counts = _allocate_counts(len(signers), ratios)
    rng = random.Random(seed)
    best: dict[str, tuple[str, ...]] | None = None
    best_key: tuple[float, tuple[tuple[str, ...], ...]] | None = None

    for _ in range(search_trials):
        shuffled = signers.copy()
        rng.shuffle(shuffled)
        cursor = 0
        candidate: dict[str, tuple[str, ...]] = {}
        for split in _SPLITS:
            next_cursor = cursor + counts[split]
            candidate[split] = tuple(sorted(shuffled[cursor:next_cursor]))
            cursor = next_cursor
        score = _score_allocation(candidate, stats, ratios)
        lexical = tuple(candidate[split] for split in _SPLITS)
        key = (score, lexical)
        if best_key is None or key < best_key:
            best = candidate
            best_key = key

    if best is None:
        raise SplitError("Split search did not produce a candidate.")
    return best


def _allocate_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {split: total * ratios[split] for split in _SPLITS}
    counts = {split: int(exact[split]) for split in _SPLITS}
    remaining = total - sum(counts.values())
    priority = sorted(
        _SPLITS,
        key=lambda split: (exact[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for split in priority[:remaining]:
        counts[split] += 1

    zero_splits = [split for split in _SPLITS if counts[split] == 0]
    for split in zero_splits:
        donor = max(_SPLITS, key=lambda name: (counts[name], ratios[name]))
        if counts[donor] <= 1:
            raise SplitError("Not enough signers to give every split at least one.")
        counts[donor] -= 1
        counts[split] += 1
    return counts


def _score_allocation(
    allocation: Mapping[str, Sequence[str]],
    stats: Mapping[str, _SignerStats],
    ratios: Mapping[str, float],
) -> float:
    total_instances = sum(len(item.instance_ids) for item in stats.values())
    global_gloss = Counter[int]()
    for item in stats.values():
        global_gloss.update(item.gloss_counts)
    glosses = tuple(global_gloss)
    global_total = sum(global_gloss.values())

    ratio_error = 0.0
    coverage_error = 0.0
    distribution_error = 0.0
    for split in _SPLITS:
        split_instances = sum(len(stats[signer].instance_ids) for signer in allocation[split])
        actual_ratio = split_instances / total_instances
        ratio_error += abs(actual_ratio - ratios[split])

        split_gloss = Counter[int]()
        for signer in allocation[split]:
            split_gloss.update(stats[signer].gloss_counts)
        split_total = sum(split_gloss.values())
        missing = sum(split_gloss[gloss] == 0 for gloss in glosses)
        coverage_error += missing / max(len(glosses), 1)
        if split_total:
            distribution_error += sum(
                abs(split_gloss[gloss] / split_total - global_gloss[gloss] / global_total)
                for gloss in glosses
            )
    return round(10.0 * ratio_error + 5.0 * coverage_error + distribution_error, 12)


def _make_definition(
    records: Sequence[ManifestRecord],
    allocation: Mapping[str, Sequence[str]],
    ratios: dict[str, float],
    *,
    seed: int,
    score: float,
) -> SplitDefinition:
    return SplitDefinition(
        seed=seed,
        target_ratios=dict(ratios),
        signer_ids={split: tuple(sorted(allocation[split])) for split in _SPLITS},
        signer_counts={split: len(allocation[split]) for split in _SPLITS},
        instance_counts={
            split: len({item.instance_id for item in records if item.split == split})
            for split in _SPLITS
        },
        clip_counts={split: sum(item.split == split for item in records) for split in _SPLITS},
        gloss_counts={
            split: len({item.class_index for item in records if item.split == split})
            for split in _SPLITS
        },
        score=score,
    )


def _validate_explicit_allocation(
    candidate: Mapping[str, Sequence[str]],
    expected_signers: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    allocation = {
        split: tuple(sorted(_normalize_signer(value) for value in candidate.get(split, ())))
        for split in _SPLITS
    }
    flattened = [signer for split in _SPLITS for signer in allocation[split]]
    duplicates = sorted(signer for signer, count in Counter(flattened).items() if count > 1)
    if duplicates:
        raise SplitError(f"Signer overlap between splits: {duplicates}.")
    missing = sorted(set(expected_signers) - set(flattened))
    unknown = sorted(set(flattened) - set(expected_signers))
    if missing or unknown:
        raise SplitError(
            f"Explicit split does not match manifest signers; missing={missing}, unknown={unknown}."
        )
    if any(not allocation[split] for split in _SPLITS):
        raise SplitError("Every split must contain at least one signer.")
    return allocation


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    missing = [split for split in _SPLITS if split not in ratios]
    if missing:
        raise SplitError(f"Missing split ratios: {missing}.")
    normalized = {split: float(ratios[split]) for split in _SPLITS}
    if any(value <= 0 for value in normalized.values()):
        raise SplitError("Split ratios must be positive.")
    if abs(sum(normalized.values()) - 1.0) > 1e-9:
        raise SplitError("Split ratios must sum to 1.0.")
    return normalized


def _normalize_signer(value: str | int) -> str:
    text = str(value).strip()
    return text.zfill(3) if text.isdecimal() else text
