# Contributing

Use Python 3.12 and install development dependencies with:

    uv sync --extra dev

Before submitting a change:

    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

Do not commit datasets, participant information, model artifacts, secrets, or
notebook outputs. Any data-contract change must update docs/data_contract.md,
its tests, and the schema version when compatibility is broken.
