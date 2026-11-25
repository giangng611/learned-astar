from pathlib import Path

import pytest

from learned_astar.environments import Grid
from learned_astar.search import SearchLimits, bfs
from learned_astar.search.prepared import PreparedAStar, PreparedGrid


def test_demand_matches_eager_paths_and_counters_on_all_small_layouts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from demand_heuristic import DemandAStar
    from landmark_probe import LandmarkIndex

    cells = tuple((r, c) for r in range(3) for c in range(3))
    for mask in range(512):
        grid = Grid(3, 3, frozenset(c for i, c in enumerate(cells) if mask & (1 << i)))
        prepared = PreparedGrid.build(grid)
        index = LandmarkIndex.build(prepared)
        for goal in grid.free_cells():
            for landmark in (None, index):
                dense = PreparedAStar(
                    prepared, goal, primary=landmark.primary(goal) if landmark else None
                )
                sparse = DemandAStar(prepared, goal, index=landmark)
                assert not sparse.primary
                for start in grid.free_cells():
                    a, b = dense.search(start), sparse.search(start)
                    truth = bfs(grid, start, goal)
                    assert (a.status, a.cost) == (truth.status, truth.cost)
                    assert (a.status, a.path, a.cost, a.stats) == (
                        b.status,
                        b.path,
                        b.cost,
                        b.stats,
                    )
                assert all(dense.primary[i] == v for i, v in sparse.primary.items())


def test_cache_only_fills_requested_cells_and_does_not_cross_goals(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from demand_heuristic import DemandAStar
    from landmark_probe import LandmarkIndex

    grid = Grid(32, 32)
    prepared = PreparedGrid.build(grid)
    index = LandmarkIndex.build(prepared)
    first = DemandAStar(prepared, (0, 2), index=index)
    other = DemandAStar(prepared, (31, 31), index=index)
    result = first.search((0, 0))
    assert 0 < len(first.primary) < len(prepared.cells)
    before = dict(first.primary)
    assert first.search((0, 0)).path == result.path
    assert first.primary == before
    assert not other.primary
    assert other.search((0, 0)).cost == 62
    assert first.primary == before
    with pytest.raises(ValueError, match="different prepared map"):
        DemandAStar(PreparedGrid.build(grid), (0, 0), index=index)
    with pytest.raises(ValueError):
        DemandAStar(prepared, (32, 0))


def test_demand_preserves_expansion_limit_and_disconnected_fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from demand_heuristic import DemandAStar
    from landmark_probe import LandmarkIndex

    grid = Grid.from_ascii([".....", "#####", "....."])
    prepared = PreparedGrid.build(grid)
    index = LandmarkIndex.build(prepared, ((0, 0),))
    for goal in ((2, 4), (0, 4)):
        dense = PreparedAStar(prepared, goal, primary=index.primary(goal))
        sparse = DemandAStar(prepared, goal, index=index)
        for limit in (SearchLimits(1), SearchLimits()):
            a, b = dense.search((0, 0), limits=limit), sparse.search((0, 0), limits=limit)
            assert (a.status, a.path, a.stats) == (b.status, b.path, b.stats)


def test_complete_session_includes_cache_costs_and_matches_eager(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_demand_heuristic import METHODS, measure, signature

    grid = Grid.from_ascii([".....", "####.", ".....", "#####", "....."])
    pairs = [((0, 0), (2, 0)), ((2, 0), (0, 4)), ((4, 0), (0, 4)), ((2, 0), (2, 0))]
    signatures = {}
    entries = {}
    for method in METHODS:
        timing, results = measure(method, grid, pairs)
        assert timing["total_seconds"] > 0
        assert timing["other_seconds"] >= 0
        assert timing["setup_seconds"] + timing["search_seconds"] <= timing["total_seconds"]
        assert timing["goals_prepared"] == 2
        entries[method] = timing["score_entries"]
        signatures[method] = signature(results)
        for (start, goal), result in zip(pairs, results, strict=True):
            expected = bfs(grid, start, goal)
            assert (result["status"], result["cost"]) == (expected.status, expected.cost)
    for base in ("manhattan", "landmark"):
        assert signatures[base] == signatures[base + "_demand"]
        assert 0 < entries[base + "_demand"] < entries[base] == 50


def test_test_expansion_gate_requires_both_wall_groups_and_correct_paths(monkeypatch):
    import copy

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from report_demand_heuristic import gate_passed

    summary = {
        "config": {
            "gate": {
                "goals": 128,
                "families": ["walls_32", "walls_64"],
                "maximum_mean_ratio_to_eager": 0.95,
            }
        },
        "measurement": {"all_correct": True},
        "verification": {"passed": True},
        "families": {
            f: {
                "128": {
                    "methods": {
                        "landmark_demand": {"ratio_to_landmark": {"mean": 0.9, "high": 0.99}}
                    }
                }
            }
            for f in ("walls_32", "walls_64")
        },
    }
    assert gate_passed(summary)
    for metric, value in (("mean", 0.96), ("high", 1.0)):
        bad = copy.deepcopy(summary)
        bad["families"]["walls_64"]["128"]["methods"]["landmark_demand"]["ratio_to_landmark"][
            metric
        ] = value
        assert not gate_passed(bad)
    summary["measurement"]["all_correct"] = False
    assert not gate_passed(summary)
