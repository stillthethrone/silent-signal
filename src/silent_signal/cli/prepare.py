"""Command line workflow for ASL Citizen and VSL400 data preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.configuration import ConfigurationError, DatasetConfig, load_dataset_config
from silent_signal.contracts import ManifestRecord, SplitDefinition, ValidationLevel
from silent_signal.data.manifest import (
    ManifestBuildResult,
    ManifestError,
    build_manifest,
    manifest_summary,
    read_manifest,
    write_labels,
    write_manifest,
)
from silent_signal.data.splits import (
    SplitError,
    create_signer_disjoint_split,
    load_signer_allocation,
    verify_official_split,
    write_split_definition,
)
from silent_signal.data.validation import (
    ValidationError,
    ValidationResult,
    validate_manifest,
    write_validation_report,
)

_DEFAULT_CONFIG = Path("configs/dataset/vsl400.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without importing heavyweight ML dependencies."""

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Dataset YAML configuration.",
    )
    common.add_argument(
        "--root",
        type=Path,
        help="Dataset root. Overrides dataset.root and its environment variable.",
    )

    parser = argparse.ArgumentParser(
        prog="ss-prepare",
        description="Prepare and validate ASL Citizen or VSL400 without extracting pose.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "build-manifest",
        parents=[common],
        help="Parse dataset metadata and write canonical manifests.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common],
        help="Validate an existing manifest and its video files.",
    )
    _add_validation_arguments(validate_parser)
    validate_parser.add_argument("--manifest", type=Path, help="Input manifest CSV/Parquet.")

    split_parser = subparsers.add_parser(
        "create-splits",
        parents=[common],
        help="Create or verify a signer-disjoint train/validation/test split.",
    )
    split_parser.add_argument("--manifest", type=Path, help="Input manifest CSV/Parquet.")
    split_parser.add_argument(
        "--official-split",
        type=Path,
        help="VSL400 signer allocation JSON; not applicable to ASL Citizen's official CSVs.",
    )
    split_parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Allow split generation when the manifest contains invalid clips.",
    )

    summary_parser = subparsers.add_parser(
        "summarize",
        parents=[common],
        help="Print compact statistics for an existing manifest.",
    )
    summary_parser.add_argument("--manifest", type=Path, help="Input manifest CSV/Parquet.")

    all_parser = subparsers.add_parser(
        "all",
        parents=[common],
        help="Build, validate, split, and write every data-preparation artifact.",
    )
    _add_validation_arguments(all_parser)
    all_parser.add_argument(
        "--official-split",
        type=Path,
        help="VSL400 signer allocation JSON; not applicable to ASL Citizen's official CSVs.",
    )
    all_parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Create splits despite validation errors (intended only for diagnostics).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_dataset_config(args.config, root_override=args.root)
        if config.split.strategy == "official" and getattr(args, "official_split", None):
            raise SplitError("ASL Citizen's official CSV splits cannot be overridden by JSON.")
        if args.command == "build-manifest":
            build = build_manifest(config)
            _write_build_artifacts(build, config)
            _print_json(manifest_summary(build.records))
            return 0
        if args.command == "validate":
            records = read_manifest(_manifest_path(args.manifest, config))
            result = _validate(records, config, args)
            _write_validation_artifacts(result, config)
            _print_json(result.to_report()["summary"])
            return 1 if result.has_errors else 0
        if args.command == "create-splits":
            records = read_manifest(_manifest_path(args.manifest, config))
            if any(not item.is_valid for item in records) and not args.allow_invalid:
                raise SplitError(
                    "Manifest contains invalid clips. Fix validation findings or pass "
                    "--allow-invalid for diagnostic use."
                )
            assigned, definition = _split(
                records,
                config,
                official_path=args.official_split,
            )
            _write_split_artifacts(assigned, definition, config)
            _print_json(definition.to_dict())
            return 0
        if args.command == "summarize":
            records = read_manifest(_manifest_path(args.manifest, config))
            _print_json(manifest_summary(records))
            return 0
        if args.command == "all":
            return _run_all(config, args)
    except (ConfigurationError, ManifestError, SplitError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


def entrypoint() -> None:
    """Console-script adapter."""

    raise SystemExit(main())


def _run_all(config: DatasetConfig, args: argparse.Namespace) -> int:
    build = build_manifest(config)
    _write_build_artifacts(build, config)
    result = _validate(build.records, config, args)
    _write_validation_artifacts(result, config)
    if result.has_errors and not args.allow_invalid:
        print(
            "Validation failed; split generation was skipped. "
            f"See {_output_path(config.outputs.report)}.",
            file=sys.stderr,
        )
        _print_json(result.to_report()["summary"])
        return 1
    assigned, definition = _split(
        result.records,
        config,
        official_path=args.official_split,
    )
    _write_split_artifacts(assigned, definition, config)
    payload = {
        "validation": result.to_report()["summary"],
        "split": definition.to_dict(),
    }
    _print_json(payload)
    return 0


def _validate(
    records: Sequence[ManifestRecord],
    config: DatasetConfig,
    args: argparse.Namespace,
) -> ValidationResult:
    return validate_manifest(
        records,
        dataset_root=config.root,
        expected=config.expected,
        expected_views=tuple(config.views),
        level=ValidationLevel(args.level),
        workers=args.workers,
    )


def _split(
    records: Sequence[ManifestRecord],
    config: DatasetConfig,
    *,
    official_path: Path | None,
) -> tuple[tuple[ManifestRecord, ...], SplitDefinition]:
    if config.split.strategy == "official":
        if official_path is not None:
            raise SplitError("Official CSV splits cannot be overridden by a signer allocation.")
        source = build_manifest(config)
        source_files = {
            split: {
                "path": path,
                "sha256": hashlib.sha256((config.root / path).read_bytes()).hexdigest(),
            }
            for split, path in source.source_metadata.items()
        }
        return verify_official_split(
            records, source_records=source.records, source_files=source_files
        )
    allocation: dict[str, tuple[str, ...]] | None = None
    selected_official = official_path
    if selected_official is None and config.split.official_file is not None:
        configured = _output_path(config.split.official_file)
        if configured.is_file():
            selected_official = configured
    if selected_official is not None:
        allocation = load_signer_allocation(_output_path(selected_official))
    return create_signer_disjoint_split(
        records,
        ratios=config.split.ratios,
        seed=config.split.seed,
        search_trials=config.split.search_trials,
        signer_ids=allocation,
    )


def _write_build_artifacts(build: ManifestBuildResult, config: DatasetConfig) -> None:
    write_manifest(build.records, _output_path(config.outputs.manifest_csv))
    write_manifest(build.records, _output_path(config.outputs.manifest_parquet))
    write_labels(build.labels, _output_path(config.outputs.labels), dataset=config.name)


def _write_validation_artifacts(
    result: ValidationResult,
    config: DatasetConfig,
) -> None:
    write_manifest(result.records, _output_path(config.outputs.manifest_csv))
    write_manifest(result.records, _output_path(config.outputs.manifest_parquet))
    write_validation_report(result, _output_path(config.outputs.report))
    invalid = tuple(item for item in result.records if not item.is_valid)
    write_manifest(invalid, _output_path(config.outputs.invalid_records))


def _write_split_artifacts(
    records: Sequence[ManifestRecord],
    definition: SplitDefinition,
    config: DatasetConfig,
) -> None:
    write_manifest(records, _output_path(config.outputs.manifest_csv))
    write_manifest(records, _output_path(config.outputs.manifest_parquet))
    write_split_definition(definition, _output_path(config.outputs.split))


def _manifest_path(value: Path | None, config: DatasetConfig) -> Path:
    return _output_path(value or config.outputs.manifest_parquet)


def _output_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--level",
        choices=[item.value for item in ValidationLevel],
        default=ValidationLevel.METADATA.value,
        help="metadata: structural/files; probe: ffprobe; decode: full ffmpeg decode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent ffprobe/ffmpeg processes.",
    )


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    entrypoint()
