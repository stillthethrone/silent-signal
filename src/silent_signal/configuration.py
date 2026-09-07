"""Typed configuration loading for the Silent Signal pipeline."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from silent_signal.contracts import SplitName, View

_ENV_PATTERN = re.compile(r"^\$\{(?:oc\.env:)?([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigurationError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ViewConfig:
    directory: str
    metadata: str | None = None
    directory_aliases: tuple[str, ...] = ()
    metadata_aliases: tuple[str, ...] = ()

    def resolve_directory(self, root: Path) -> Path:
        return _first_existing(root, (self.directory, *self.directory_aliases), "directory")

    def resolve_metadata(self, root: Path) -> Path:
        if self.metadata is None:
            raise ConfigurationError("This view has no per-view metadata file.")
        return _first_existing(root, (self.metadata, *self.metadata_aliases), "metadata file")


@dataclass(frozen=True, slots=True)
class ExpectedConfig:
    clips: int | None = None
    glosses: int | None = None
    signers: int | None = None
    views_per_instance: int = 3
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    enforce_counts: bool = True
    fps_tolerance: float = 0.05
    split_clips: dict[str, int] = field(default_factory=dict)
    split_signers: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    ratios: dict[str, float]
    seed: int = 42
    search_trials: int = 5000
    official_file: Path | None = None
    strategy: str = "signer_disjoint"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    manifest_csv: Path
    manifest_parquet: Path
    labels: Path
    split: Path
    report: Path
    invalid_records: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    root: Path
    video_extension: str
    gloss_file: str | None
    views: dict[str, ViewConfig]
    expected: ExpectedConfig
    split: SplitConfig
    outputs: OutputConfig
    adapter: str = "vsl400"
    metadata_splits: dict[str, str] = field(default_factory=dict)


def load_dataset_config(
    path: str | Path,
    *,
    root_override: str | Path | None = None,
) -> DatasetConfig:
    """Load and validate a dataset YAML file."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping.")
    if raw.get("schema_version") != 1:
        raise ConfigurationError("Only dataset configuration schema_version 1 is supported.")
    dataset = _mapping(raw, "dataset")
    expected = _optional_mapping(raw, "expected")
    split = _mapping(raw, "split")
    outputs = _mapping(raw, "outputs")

    root = Path(root_override).expanduser() if root_override else _configured_root(dataset)
    root = root.resolve()

    adapter = str(dataset.get("adapter", "vsl400"))
    if adapter not in {"vsl400", "asl_citizen"}:
        raise ConfigurationError(f"Unsupported dataset adapter: {adapter!r}.")
    views_raw = _mapping(dataset, "views")
    canonical_views = (
        (View.FRONT.value, View.LEFT.value, View.RIGHT.value)
        if adapter == "vsl400"
        else (View.SINGLE.value,)
    )
    if set(views_raw) != set(canonical_views):
        raise ConfigurationError(
            f"Adapter {adapter!r} requires exactly these dataset views: {canonical_views}."
        )
    views: dict[str, ViewConfig] = {}
    for view in canonical_views:
        entry = _mapping(views_raw, view)
        views[view] = ViewConfig(
            directory=_required_str(entry, "directory"),
            metadata=(
                _required_str(entry, "metadata")
                if adapter == "vsl400"
                else _optional_string(entry.get("metadata"))
            ),
            directory_aliases=_string_tuple(entry.get("directory_aliases", ())),
            metadata_aliases=_string_tuple(entry.get("metadata_aliases", ())),
        )

    strategy = str(split.get("strategy", "signer_disjoint"))
    required_strategy = "official" if adapter == "asl_citizen" else "signer_disjoint"
    if strategy != required_strategy:
        raise ConfigurationError(
            f"Adapter {adapter!r} requires split.strategy={required_strategy}."
        )
    metadata_splits: dict[str, str] = {}
    ratios: dict[str, float] = {}
    if strategy == "official":
        if "ratios" in split or split.get("official_file"):
            raise ConfigurationError(
                "Official CSV splits cannot be replaced by ratios or a signer allocation JSON."
            )
        sources = _mapping(dataset, "metadata_splits")
        if set(sources) != {name.value for name in SplitName}:
            raise ConfigurationError("metadata_splits must contain train, validation and test.")
        metadata_splits = {name.value: _required_str(sources, name.value) for name in SplitName}
        if len(set(metadata_splits.values())) != len(SplitName):
            raise ConfigurationError("Each official split must use a different metadata file.")
    else:
        ratios_raw = _mapping(split, "ratios")
        missing_ratios = [name.value for name in SplitName if name.value not in ratios_raw]
        if missing_ratios:
            raise ConfigurationError(f"Missing split ratios: {missing_ratios}.")
        ratios = {name.value: float(ratios_raw[name.value]) for name in SplitName}
        if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
            raise ConfigurationError("Split ratios must be positive and sum to 1.0.")

    extension = str(dataset.get("video_extension", ".mp4"))
    if not extension.startswith("."):
        extension = f".{extension}"

    views_per_instance = int(expected.get("views_per_instance", len(views)))
    if views_per_instance != len(views):
        raise ConfigurationError(
            "expected.views_per_instance must equal the number of configured views."
        )
    search_trials = int(split.get("search_trials", 5000))
    if search_trials < 1:
        raise ConfigurationError("split.search_trials must be at least 1.")
    fps_tolerance = float(expected.get("fps_tolerance", 0.05))
    if fps_tolerance < 0:
        raise ConfigurationError("expected.fps_tolerance must not be negative.")

    official_value = split.get("official_file")
    return DatasetConfig(
        name=str(dataset.get("name", "vsl400")),
        adapter=adapter,
        metadata_splits=metadata_splits,
        root=root,
        video_extension=extension.lower(),
        gloss_file=_optional_string(dataset.get("gloss_file")),
        views=views,
        expected=ExpectedConfig(
            clips=_optional_int(expected.get("clips")),
            glosses=_optional_int(expected.get("glosses")),
            signers=_optional_int(expected.get("signers")),
            views_per_instance=views_per_instance,
            fps=_optional_float(expected.get("fps")),
            width=_optional_int(expected.get("width")),
            height=_optional_int(expected.get("height")),
            enforce_counts=bool(expected.get("enforce_counts", True)),
            fps_tolerance=fps_tolerance,
            split_clips=_split_counts(expected, "split_clips"),
            split_signers=_split_counts(expected, "split_signers"),
        ),
        split=SplitConfig(
            ratios=ratios,
            seed=int(split.get("seed", 42)),
            search_trials=search_trials,
            official_file=Path(str(official_value)) if official_value else None,
            strategy=strategy,
        ),
        outputs=OutputConfig(
            manifest_csv=Path(_required_str(outputs, "manifest_csv")),
            manifest_parquet=Path(_required_str(outputs, "manifest_parquet")),
            labels=Path(_required_str(outputs, "labels")),
            split=Path(_required_str(outputs, "split")),
            report=Path(_required_str(outputs, "report")),
            invalid_records=Path(_required_str(outputs, "invalid_records")),
        ),
    )


def _configured_root(dataset: Mapping[str, Any]) -> Path:
    value = dataset.get("root")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("dataset.root must be set or supplied with --root.")
    match = _ENV_PATTERN.fullmatch(value.strip())
    if not match:
        return Path(value)
    env_name = match.group(1)
    resolved = os.environ.get(env_name)
    if not resolved:
        raise ConfigurationError(
            f"Environment variable {env_name} is not set. Set it or pass --root."
        )
    return Path(resolved)


def _first_existing(root: Path, candidates: tuple[str, ...], kind: str) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    formatted = ", ".join(str(root / candidate) for candidate in candidates)
    raise ConfigurationError(f"Dataset {kind} not found; checked: {formatted}")


def _split_counts(parent: Mapping[str, Any], key: str) -> dict[str, int]:
    values = _optional_mapping(parent, key)
    if not values:
        return {}
    if set(values) != {name.value for name in SplitName}:
        raise ConfigurationError(f"expected.{key} must contain train, validation and test.")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values.values()
    ):
        raise ConfigurationError(f"expected.{key} values must be positive integers.")
    return {str(name): int(value) for name, value in values.items()}


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping.")
    return value


def _optional_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping.")
    return value


def _required_str(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string.")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError("Alias values must be strings or lists.")
    return tuple(str(item) for item in value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)
