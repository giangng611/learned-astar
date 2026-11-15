from dataclasses import replace

import numpy as np
import pytest

from learned_astar.data import (
    BLOCKED,
    UNREACHABLE,
    DatasetConfig,
    generate_dataset,
    load_record,
    read_manifest,
    reverse_bfs,
    validate_labels,
    verify_dataset,
)
from learned_astar.data.dataset import layout_id
from learned_astar.environments import Grid, random_grid
from learned_astar.reproducibility import make_rng
from learned_astar.search import bfs


def test_labels_match_independent_bfs_for_every_free_cell():
    for seed in range(8):
        grid = random_grid(7, 9, 0.3, make_rng(seed))
        goal = next(grid.free_cells())
        labels = reverse_bfs(grid, goal)
        validate_labels(grid, goal, labels)
        for cell in grid.free_cells():
            cost = bfs(grid, cell, goal).cost
            assert labels[cell] == (UNREACHABLE if cost is None else cost)
        for cell in grid.blocked:
            assert labels[cell] == BLOCKED


def test_label_validator_rejects_wrong_distance_and_connectivity():
    grid = Grid.from_ascii(["..#.", "..#."])
    labels = reverse_bfs(grid, (0, 0))
    assert labels[0, 3] == UNREACHABLE
    for cell, value in [((0, 1), 5), ((0, 1), -1), ((0, 2), -1), ((1, 1), 0)]:
        broken = labels.copy()
        broken[cell] = value
        with pytest.raises(ValueError):
            validate_labels(grid, (0, 0), broken)


def test_dataset_is_reproducible_and_splits_do_not_overlap(tmp_path):
    cfg = DatasetConfig(train_maps=4, validation_maps=2, test_maps=2, extra_maps=0, size=8)
    first, second = tmp_path / "first", tmp_path / "second"
    generate_dataset(first, cfg)
    generate_dataset(second, cfg)
    assert verify_dataset(first) == {"maps": 8, "verified_goal_maps": 14, "unique_layout_groups": 8}
    a, b = read_manifest(first), read_manifest(second)
    assert [r["group_id"] for r in a] == [r["group_id"] for r in b]
    for left, right in zip(a, b, strict=True):
        ld, rd = load_record(first, left), load_record(second, right)
        for key in ld:
            np.testing.assert_array_equal(ld[key], rd[key])
    with pytest.raises(FileExistsError):
        generate_dataset(first, cfg)


def test_rotated_layouts_share_a_group():
    grid = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
    assert layout_id(grid, rotations=True) == layout_id(np.rot90(grid), rotations=True)


def test_corrupt_manifest_and_map_are_rejected(tmp_path):
    root = tmp_path / "data"
    generate_dataset(
        root, DatasetConfig(train_maps=2, validation_maps=0, test_maps=0, extra_maps=0, size=8)
    )
    record = read_manifest(root)[0]
    (root / record["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_record(root, record)
    (root / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(root)


def test_invalid_dataset_configuration():
    with pytest.raises(ValueError):
        replace(DatasetConfig(), train_maps=3)
    with pytest.raises(ValueError):
        replace(DatasetConfig(), starts_per_map=0)


def test_maps_with_insufficient_free_cells_are_recorded_without_resampling(tmp_path):
    config = DatasetConfig(
        train_maps=2, validation_maps=0, test_maps=0, extra_maps=0, size=2, goals_per_map=5
    )
    metadata = generate_dataset(tmp_path / "small", config)
    records = read_manifest(tmp_path / "small")
    assert metadata["status_counts"] == {"insufficient_free_cells": 2}
    assert len(records) == 2
    assert all(record["file"] is None for record in records)
    assert verify_dataset(tmp_path / "small")["verified_goal_maps"] == 0
