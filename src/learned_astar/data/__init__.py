"""Exact distance labels and datasets split by whole map layout."""

from .dataset import DatasetConfig, generate_dataset, load_record, read_manifest, verify_dataset
from .labels import BLOCKED, UNREACHABLE, reverse_bfs, validate_labels

__all__ = [
    "DatasetConfig",
    "generate_dataset",
    "load_record",
    "read_manifest",
    "verify_dataset",
    "BLOCKED",
    "UNREACHABLE",
    "reverse_bfs",
    "validate_labels",
]
