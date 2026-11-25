import importlib.util
from pathlib import Path

import pytest

from learned_astar.environments import Grid, validate_path
from learned_astar.search import bfs
from learned_astar.search.prepared import PreparedAStar, PreparedGrid

SPEC = importlib.util.spec_from_file_location(
    "landmark_probe", Path(__file__).parents[1] / "scripts/landmark_probe.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_landmark_bounds_and_optimal_paths_on_every_three_by_three_grid():
    cells = tuple((r, c) for r in range(3) for c in range(3))
    for mask in range(512):
        grid = Grid(3, 3, frozenset(c for i, c in enumerate(cells) if mask & (1 << i)))
        prepared = PreparedGrid.build(grid)
        index = module.LandmarkIndex.build(prepared)
        for goal in grid.free_cells():
            scores = index.primary(goal)
            planner = PreparedAStar(prepared, goal, primary=scores)
            assert scores[prepared.index(goal)] == 0
            for start in grid.free_cells():
                i = prepared.index(start)
                manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                assert scores[i] >= manhattan
                for neighbor in prepared.neighbors[i]:
                    assert scores[i] <= 1 + scores[neighbor]
                expected = bfs(grid, start, goal)
                result = planner.search(start)
                assert (result.status, result.cost) == (expected.status, expected.cost)
                assert result.stats.reexpanded == 0
                if result.cost is not None:
                    assert scores[i] <= result.cost
                    validate_path(grid, result.path, start, goal, result.cost)


def test_empty_or_disconnected_landmarks_fall_back_to_manhattan():
    grid = Grid.from_ascii(["...", "###", "..."])
    prepared = PreparedGrid.build(grid)
    for landmarks in ((), ((0, 0),)):
        index = module.LandmarkIndex.build(prepared, landmarks)
        goal = (2, 2)
        assert index.primary(goal) == PreparedAStar(prepared, goal).primary
    assert module.corner_landmarks(Grid(1, 1, frozenset({(0, 0)}))) == ()
    assert module.corner_landmarks(Grid(1, 1)) == ((0, 0),)


def test_landmark_validation_and_reuse_for_new_goals():
    grid = Grid.from_ascii(["#..", "...", "..#"])
    prepared = PreparedGrid.build(grid)
    index = module.LandmarkIndex.build(prepared)
    assert index.landmarks == ((0, 1), (0, 2), (2, 0), (1, 2))
    assert index.preprocess_expanded == 4 * 7
    fields = index.distances
    first = index.primary((1, 1))
    index.primary((2, 1))
    assert index.primary((1, 1)) == first
    assert index.distances is fields
    for bad in (((0, 0),), ((9, 9),), ((1, 1), (1, 1))):
        with pytest.raises(ValueError):
            module.LandmarkIndex.build(prepared, bad)
    with pytest.raises(ValueError):
        index.primary((0, 0))


def test_wall_example_improves_primary_bound_without_changing_optimal_cost():
    examples = module.illustrations()["examples"]
    wall = next(row for row in examples if row["name"] == "wall_detour")
    base, new = (wall["methods"][key] for key in ("manhattan", "landmark"))
    assert base["cost"] == new["cost"] == 14
    assert new["primary_at_start"] > base["primary_at_start"]
    assert new["expanded"] < base["expanded"]
    assert wall["landmark_bfs_expansions"] > 0
