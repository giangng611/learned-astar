import pytest

from learned_astar.environments import Grid, random_grid, sample_query, validate_path
from learned_astar.reproducibility import make_rng


def test_neighbor_order_obstacles_and_boundaries():
    grid = Grid(3, 3, frozenset({(1, 2)}))
    assert list(grid.neighbors((1, 1))) == [(0, 1), (2, 1), (1, 0)]
    assert list(grid.neighbors((0, 0))) == [(0, 1), (1, 0)]
    assert list(Grid(1, 1).neighbors((0, 0))) == []


@pytest.mark.parametrize("fraction", [0, 0.1, 0.25, 0.4, 1])
def test_exact_obstacle_count_and_repeatable_generation(fraction):
    first = random_grid(7, 11, fraction, make_rng(4550))
    second = random_grid(7, 11, fraction, make_rng(4550))
    assert first == second
    assert len(first.blocked) == round(fraction * 77)
    assert first.obstacle_fraction == len(first.blocked) / 77
    assert len(list(first.free_cells())) + len(first.blocked) == 77


def test_query_sampling_keeps_disconnected_cases():
    grid = Grid.from_ascii([".#."])
    start, goal = sample_query(grid, make_rng(1))
    assert {start, goal} == {(0, 0), (0, 2)}
    assert (start, goal) == sample_query(grid, make_rng(1))


@pytest.mark.parametrize("rows", [["#"], ["."], ["##", "#."]])
def test_query_needs_two_free_cells(rows):
    with pytest.raises(ValueError, match="at least two"):
        sample_query(Grid.from_ascii(rows), make_rng(1))


@pytest.mark.parametrize("rows", [[], [""], ["..", "."], [".x"]])
def test_invalid_ascii_maps(rows):
    with pytest.raises(ValueError):
        Grid.from_ascii(rows)


@pytest.mark.parametrize("cell", [(-1, 0), (2, 0), (0, 2), (False, 0), (0.0, 0), (0,)])
def test_invalid_blocked_cells(cell):
    with pytest.raises(ValueError):
        Grid(2, 2, frozenset({cell}))


@pytest.mark.parametrize(
    "height,width,fraction",
    [
        (0, 2, 0.25),
        (2, -1, 0.25),
        (2, 2, -0.1),
        (2, 2, 1.1),
        (2, 2, float("nan")),
    ],
)
def test_invalid_generation_settings(height, width, fraction):
    with pytest.raises(ValueError):
        random_grid(height, width, fraction, make_rng(1))


def test_valid_paths_and_zero_length_path():
    grid = Grid.from_ascii(["..", "#."])
    validate_path(grid, [(0, 0), (0, 1), (1, 1)], (0, 0), (1, 1), 2)
    validate_path(grid, [(0, 0)], (0, 0), (0, 0), 0)


@pytest.mark.parametrize(
    "path,cost",
    [
        ([], 0),
        ([(0, 1), (1, 1)], 1),
        ([(0, 0), (1, 1)], 1),
        ([(0, 0), (1, 0), (1, 1)], 2),
        ([(0, 0), (0, 1), (1, 1)], 3),
        ([(0, 0), (0, -1), (1, 1)], 2),
    ],
)
def test_path_validator_rejects_bad_routes(path, cost):
    with pytest.raises(ValueError):
        validate_path(Grid.from_ascii(["..", "#."]), path, (0, 0), (1, 1), cost)
