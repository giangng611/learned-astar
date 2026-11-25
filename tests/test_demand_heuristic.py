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


def test_portable_archive_check_rejects_changed_means_and_raw_evidence(monkeypatch, tmp_path):
    import gzip
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from check_demand_archive import check, digest

    (tmp_path / "maps").mkdir()
    (tmp_path / "maps/example.npz").write_bytes(b"hash-only fixture")
    record = {
        "layout_id": "example",
        "kind": "random",
        "width": 32,
        "file": "maps/example.npz",
        "sha256": digest(tmp_path / "maps/example.npz"),
    }
    (tmp_path / "policy_manifest.json").write_text(json.dumps([record]))
    methods = {"manhattan": 2.0, "landmark_demand": 1.0}
    with gzip.open(tmp_path / "sessions.jsonl.gz", "wt") as stream:
        for method, median in methods.items():
            for rep, factor in enumerate((0.5, 1, 1.5)):
                stream.write(
                    json.dumps(
                        {
                            "layout_id": "example",
                            "family": "random_32",
                            "goal_count": 1,
                            "method": method,
                            "repetition": rep,
                            "total_seconds": median * factor,
                            "result_sha256": "same-results",
                        }
                    )
                    + "\n"
                )
    with gzip.open(tmp_path / "queries.jsonl.gz", "wt") as stream:
        stream.write("{}\n")
    measurement = {
        "sessions": 6,
        **{
            f"{name}_sha256": digest(tmp_path / f"{name}.jsonl.gz")
            for name in ("sessions", "queries")
        },
    }
    (tmp_path / "measurement.json").write_text(json.dumps(measurement))
    summary = {
        "measurement": measurement,
        "config": {
            "repetitions": 3,
            "queries_per_session": 2,
            "goal_counts": [1],
            "methods": list(methods),
        },
        "families": {
            "random_32": {
                "1": {
                    "methods": {
                        name: {
                            "total_ms_per_query": {"mean": value * 500},
                            "ratio_to_manhattan": {"mean": value / 2},
                        }
                        for name, value in methods.items()
                    }
                }
            }
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert check(tmp_path) == {"maps": 1, "median_sessions": 2, "checked_means": 4}
    summary["families"]["random_32"]["1"]["methods"]["landmark_demand"]["ratio_to_manhattan"][
        "mean"
    ] = 0.4
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Incorrect summary mean"):
        check(tmp_path)
    with (tmp_path / "queries.jsonl.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Raw evidence checksum"):
        check(tmp_path)
