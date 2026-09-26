"""Plot train/validation curves of a pose-transformer run and print a diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from silent_signal.evaluation.training_curves import (
    load_run,
    plot_training_curves,
    summarize_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ss-plot-training",
        description="Draw train/validation curves from <run-root>/history.json (needs matplotlib).",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path; defaults to <run-root>/figures/training_curves.png.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        history, report = load_run(args.run_root)
        summary = summarize_history(history, report=report)
        output = args.output or args.run_root / "figures" / "training_curves.png"
        plot_training_curves(history, output, summary=summary, title=args.run_root.name)
        (args.run_root / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"figure": str(output), **summary}, indent=2))
        return 0
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
