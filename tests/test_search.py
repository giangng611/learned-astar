from itertools import permutations

import pytest

from learned_astar.environments import Grid, random_grid, sample_query, validate_path
from learned_astar.reproducibility import make_rng
from learned_astar.search import (
    SearchLimits,
    SearchStatus,
    astar,
    best_first_search,
    bfs,
    dijkstra,
    weighted_astar,
)

BASELINES = [bfs, dijkstra, astar, weighted_astar]


def assert_valid_result(grid, start, goal, result):
    assert result.elapsed_seconds >= 0
    assert result.stats.expanded >= result.stats.unique_expanded
    assert result.stats.reexpanded == result.stats.expanded - result.stats.unique_expanded
    if result.status == SearchStatus.FOUND:
        validate_path(grid, result.path, start, goal, result.cost)
    else:
        assert result.path == ()
        assert result.cost is None


def check_against_bfs(grid, start, goal):
    reference = bfs(grid, start, goal)
    assert_valid_result(grid, start, goal, reference)
    for search in (dijkstra, astar):
        result = search(grid, start, goal)
        assert_valid_result(grid, start, goal, result)
        assert result.status == reference.status
        assert result.cost == reference.cost
    for weight in (1.0, 1.25, 1.5, 2.0):
        result = weighted_astar(grid, start, goal, weight)
        assert_valid_result(grid, start, goal, result)
        assert result.status == reference.status
        if reference.status == SearchStatus.FOUND:
            assert reference.cost <= result.cost <= weight * reference.cost


def test_all_3_by_3_obstacle_layouts_and_free_endpoint_pairs():
    # Exhaustive small maps catch problems that a few lucky random maps can miss.
    cells = [(r, c) for r in range(3) for c in range(3)]
    for mask in range(2 ** len(cells)):
        blocked = frozenset(cell for i, cell in enumerate(cells) if mask & (1 << i))
        grid = Grid(3, 3, blocked)
        for start, goal in permutations(grid.free_cells(), 2):
            check_against_bfs(grid, start, goal)


@pytest.mark.parametrize("seed", range(20))
def test_larger_random_maps_match_bfs_and_weight_bounds(seed):
    rng = make_rng(seed)
    height, width = [(6, 9), (16, 16), (32, 32), (64, 64)][seed % 4]
    grid = random_grid(height, width, [0.1, 0.25, 0.4][seed % 3], rng)
    start, goal = sample_query(grid, rng)
    check_against_bfs(grid, start, goal)


@pytest.mark.parametrize("search", BASELINES)
def test_equal_start_and_goal_needs_no_expansion(search):
    result = search(Grid(1, 1), (0, 0), (0, 0), limits=SearchLimits(0, 0))
    assert result.status == SearchStatus.FOUND
    assert result.path == ((0, 0),)
    assert result.cost == 0
    assert result.stats.expanded == result.stats.generated == 0
    assert result.stats.queue_insertions == 1


@pytest.mark.parametrize("search", BASELINES)
def test_unreachable_goal_is_reported_without_a_partial_path(search):
    grid = Grid.from_ascii(["..#..", "..#.."])
    result = search(grid, (0, 0), (0, 4))
    assert result.status == SearchStatus.NO_PATH
    assert result.stats.expanded == result.stats.unique_expanded == 4
    assert result.stats.generated == 8
    assert result.stats.queue_insertions == 4
    assert_valid_result(grid, (0, 0), (0, 4), result)


@pytest.mark.parametrize("search", BASELINES)
def test_counts_exclude_goal_and_include_repeated_legal_neighbors(search):
    grid = Grid.from_ascii(["...."])
    result = search(grid, (0, 0), (0, 3))
    assert result.cost == 3
    assert result.stats.expanded == result.stats.unique_expanded == 3
    assert result.stats.generated == 5
    assert result.stats.queue_insertions == 4
    assert result.stats.stale_pops == result.stats.reexpanded == 0
    assert result.stats.max_frontier == 1


@pytest.mark.parametrize("search", BASELINES)
def test_corridor_requires_a_detour(search):
    grid = Grid.from_ascii([".....", "####.", "....."])
    result = search(grid, (0, 0), (2, 0))
    assert result.cost == 10
    assert_valid_result(grid, (0, 0), (2, 0), result)


def test_astar_prefers_larger_g_then_insertion_order():
    result = astar(Grid(3, 3), (0, 0), (2, 2))
    assert result.path == ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2))
    assert result.stats.expanded == 4


@pytest.mark.parametrize("search", BASELINES)
def test_runs_repeat_the_path_and_counts(search):
    grid = random_grid(16, 16, 0.25, make_rng(5))
    start, goal = sample_query(grid, make_rng(6))
    first, second = search(grid, start, goal), search(grid, start, goal)
    assert (first.status, first.path, first.cost, first.stats) == (
        second.status,
        second.path,
        second.cost,
        second.stats,
    )


@pytest.mark.parametrize("search", BASELINES)
@pytest.mark.parametrize(
    "start,goal",
    [
        ((0, 1), (1, 1)),
        ((0, 0), (0, 1)),
        ((-1, 0), (1, 1)),
        ((0, 0), (2, 0)),
        ((0, 1), (0, 1)),
        ((True, 0), (1, 1)),
    ],
)
def test_invalid_endpoints_are_errors(search, start, goal):
    with pytest.raises(ValueError):
        search(Grid.from_ascii([".#", ".."]), start, goal)


@pytest.mark.parametrize("search", BASELINES)
def test_expansion_limit_is_not_reported_as_no_path(search):
    grid = Grid(1, 5)
    result = search(grid, (0, 0), (0, 4), limits=SearchLimits(max_expansions=2))
    assert result.status == SearchStatus.EXPANSION_LIMIT
    assert result.stats.expanded == 2
    assert_valid_result(grid, (0, 0), (0, 4), result)
    assert search(grid, (0, 0), (0, 4), limits=SearchLimits(0)).stats.expanded == 0
    # Removing the goal does not expand it, so an exact budget still succeeds.
    assert search(grid, (0, 0), (0, 4), limits=SearchLimits(4)).cost == 4


@pytest.mark.parametrize("search", BASELINES)
def test_time_limit_uses_a_controlled_clock(search, monkeypatch):
    from learned_astar.search import core

    ticks = iter([0.0, 0.25, 1.0, 1.25])
    monkeypatch.setattr(core, "perf_counter", lambda: next(ticks))
    result = search(Grid(1, 5), (0, 0), (0, 4), limits=SearchLimits(time_limit_seconds=0.5))
    assert result.status == SearchStatus.TIME_LIMIT
    assert result.stats.expanded == 1
    assert result.elapsed_seconds == 1.25
    assert result.cost is None


@pytest.mark.parametrize(
    "settings",
    [
        {"max_expansions": -1},
        {"max_expansions": 1.5},
        {"max_expansions": True},
        {"time_limit_seconds": -1},
        {"time_limit_seconds": float("inf")},
        {"time_limit_seconds": float("nan")},
    ],
)
def test_invalid_limits(settings):
    with pytest.raises(ValueError):
        SearchLimits(**settings)


@pytest.mark.parametrize("weight", [0.5, -1, float("nan"), float("inf"), True])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        weighted_astar(Grid(1, 2), (0, 0), (0, 1), weight)


@pytest.mark.parametrize("estimate", [-1, float("nan"), float("inf")])
def test_invalid_heuristic_outputs(estimate):
    with pytest.raises(ValueError, match="heuristic"):
        best_first_search(Grid(1, 2), (0, 0), (0, 1), lambda cell, goal: estimate)


def test_overflowing_priority_is_an_error():
    with pytest.raises(ValueError, match="priority"):
        best_first_search(Grid(1, 2), (0, 0), (0, 1), lambda cell, goal: 1e308, weight=2)


def test_inconsistent_admissible_heuristic_reopens_nodes_and_stays_optimal():
    grid = Grid(4, 4)
    estimates = [
        [6, 3, 0, 0],
        [2, 1, 0, 1],
        [4, 1, 2, 0],
        [0, 2, 0, 0],
    ]
    for cell in grid.free_cells():
        assert estimates[cell[0]][cell[1]] <= bfs(grid, cell, (3, 3)).cost
    result = best_first_search(grid, (0, 0), (3, 3), lambda cell, goal: estimates[cell[0]][cell[1]])
    assert result.stats.reexpanded > 0
    assert result.stats.stale_pops > 0
    assert result.cost == bfs(grid, (0, 0), (3, 3)).cost == 6
    assert_valid_result(grid, (0, 0), (3, 3), result)


def test_stale_entries_do_not_count_as_expansions():
    grid = Grid.from_ascii(["....", "....", "####", "...."])
    estimates = [[14, 14, 6, 11], [1, 10, 4, 1], [0, 0, 0, 0], [13, 1, 6, 9]]
    result = best_first_search(grid, (0, 0), (3, 3), lambda cell, goal: estimates[cell[0]][cell[1]])
    assert result.status == SearchStatus.NO_PATH
    assert result.stats.unique_expanded == 8
    assert result.stats.reexpanded == 1
    assert result.stats.expanded == 9
    assert result.stats.stale_pops == 1
    assert result.stats.queue_insertions == result.stats.expanded + result.stats.stale_pops


def test_search_waits_for_goal_removal_instead_of_first_discovery():
    # Delaying the goal lets the first, longer route improve before it is removed.
    grid = Grid(2, 5)
    estimates = {(0, 1): 10, (0, 4): 20}
    result = best_first_search(grid, (0, 0), (0, 4), lambda cell, goal: estimates.get(cell, 0))
    assert result.cost == 4
    assert_valid_result(grid, (0, 0), (0, 4), result)
