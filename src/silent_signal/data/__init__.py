"""Dataset ingestion, validation, and splitting utilities."""

from silent_signal.data.manifest import build_manifest, read_manifest, write_manifest
from silent_signal.data.splits import create_signer_disjoint_split
from silent_signal.data.validation import validate_manifest

__all__ = [
    "build_manifest",
    "create_signer_disjoint_split",
    "read_manifest",
    "validate_manifest",
    "write_manifest",
]
