from itertools import permutations

import numpy as np
import pytest

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid
from learned_astar.search import SearchLimits, SearchStatus, astar, best_first_search
from learned_astar.search.core import manhattan
from learned_astar.search.prepared import PreparedAStar, PreparedGrid, ReversePolicy


def test_prepared_search_matches_reference_paths_and_every_counter_exhaustively():
    cells = tuple((r, c) for r in range(3) for c in range(3))
    for mask in range(512):
        grid = Grid(3, 3, frozenset(c for i, c in enumerate(cells) if mask & (1 << i)))
        prepared = PreparedGrid.build(grid)
        scores = tuple(float((i * 7 + mask) % 13) for i in range(9))
        for goal in grid.free_cells():
            ordinary = PreparedAStar(prepared, goal)
            learned = PreparedAStar(prepared, goal, ties=scores)
            policy = ReversePolicy(prepared, goal)
            for start in grid.free_cells():
                reference = astar(grid, start, goal)
                for planner, expected in (
                    (ordinary, reference),
                    (
                        learned,
                        best_first_search(
                            grid,
                            start,
                            goal,
                            manhattan,
                            tie_breaker=lambda c, _, values=scores: values[c[0] * 3 + c[1]],
                        ),
                    ),
                ):
                    result = planner.search(start)
                    assert (result.status, result.path, result.stats) == (
                        expected.status,
                        expected.path,
                        expected.stats,
                    )
                path = policy.path(start)
                assert (len(path) - 1 if path else None) == reference.cost
                assert policy.distances[prepared.index(start)] == (
                    reference.cost if reference.cost is not None else -1
                )


def test_prepared_exact_and_inconsistent_fields_reopen_like_reference():
    grid = Grid.from_ascii([".....", ".###.", "...#.", ".#...", "....."])
    prepared = PreparedGrid.build(grid)
    for start, goal in permutations(tuple(grid.free_cells()), 2):
        exact = np.maximum(reverse_bfs(grid, goal), 0).ravel().tolist()
        field = [float((i * 37 + start[0]) % 41) for i in range(25)]
        field[prepared.index(goal)] = 0
        for scores in (exact, field):
            result = PreparedAStar(prepared, goal, primary=scores).search(start)
            expected = best_first_search(
                grid, start, goal, lambda c, _, values=scores: values[c[0] * 5 + c[1]]
            )
            assert (result.path, result.stats) == (expected.path, expected.stats)


def test_prepared_limits_and_score_validation():
    prepared = PreparedGrid.build(Grid(4, 4))
    planner = PreparedAStar(prepared, (3, 3))
    assert (
        planner.search((0, 0), limits=SearchLimits(max_expansions=2)).status
        == SearchStatus.EXPANSION_LIMIT
    )
    assert (
        planner.search((0, 0), limits=SearchLimits(time_limit_seconds=0)).status
        == SearchStatus.TIME_LIMIT
    )
    assert planner.search((3, 3), limits=SearchLimits(max_expansions=0)).cost == 0
    for bad in ([0], [float("nan")] * 16, [-1] * 16, [float("inf")] * 16):
        with pytest.raises(ValueError, match="finite nonnegative"):
            PreparedAStar(prepared, (3, 3), ties=bad)
    with pytest.raises(ValueError, match="outside"):
        planner.search((4, 0))


def test_planner_copies_scores_and_does_not_reuse_state_across_queries_or_goals():
    prepared = PreparedGrid.build(Grid(3, 3))
    scores = [float(i) for i in range(9)]
    planner = PreparedAStar(prepared, (2, 2), ties=scores)
    first = planner.search((0, 0))
    scores[:] = [100.0] * 9
    planner.search((1, 2))
    again = planner.search((0, 0))
    assert (first.path, first.stats) == (again.path, again.stats)
    assert PreparedAStar(prepared, (0, 0)).search((2, 2)).path[-1] == (0, 0)
