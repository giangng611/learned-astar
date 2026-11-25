from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from learned_astar.environments import Grid, validate_path
from learned_astar.search import bfs


def test_goal_caches_reuse_map_state_without_reusing_wrong_goal_predictions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import run_changing_goals as runner

    grid = Grid.from_ascii([".....", "####.", ".....", "#####", "....."])
    pairs = [
        ((0, 0), (2, 0)),
        ((2, 0), (0, 4)),
        ((0, 1), (2, 0)),
        ((4, 0), (0, 4)),
        ((2, 0), (2, 0)),
    ]
    goals = {goal for _, goal in pairs}
    calls = {"graph": [], "landmark": [], "policy": [], "prediction": []}
    build_graph = runner.PreparedGrid.build
    build_index = runner.LandmarkIndex.build
    build_policy = runner.ReversePolicy
    monkeypatch.setattr(
        runner,
        "PreparedGrid",
        SimpleNamespace(build=lambda grid: calls["graph"].append(grid) or build_graph(grid)),
    )
    monkeypatch.setattr(
        runner,
        "LandmarkIndex",
        SimpleNamespace(
            build=lambda prepared: calls["landmark"].append(prepared) or build_index(prepared)
        ),
    )
    monkeypatch.setattr(
        runner,
        "ReversePolicy",
        lambda prepared, goal: calls["policy"].append(goal) or build_policy(prepared, goal),
    )
    monkeypatch.setattr(
        runner,
        "predict_distances",
        lambda model, grid, goal: (
            calls["prediction"].append(goal) or np.arange(25, dtype=float).reshape(5, 5)
        ),
    )
    for name in runner.METHODS:
        for value in calls.values():
            value.clear()
        for repeat in range(2):
            timing, results = runner.measure(name, grid, pairs, object())
            assert len(calls["graph"]) == repeat + 1
            assert len(calls["landmark"]) == (repeat + 1) * int(name.startswith("landmark"))
            assert len(calls["policy"]) == (repeat + 1) * len(goals) * int(name == "bfs_policy")
            assert len(calls["prediction"]) == (repeat + 1) * len(goals) * int(
                name.endswith("_cnn")
            )
            assert timing["goals_prepared"] == len(goals)
            assert timing["other_seconds"] >= 0
            assert timing["setup_seconds"] + timing["search_seconds"] <= timing["total_seconds"]
            for (start, goal), result in zip(pairs, results, strict=True):
                expected = bfs(grid, start, goal)
                assert (result["status"], result["cost"]) == (expected.status, expected.cost)
                if result["path"]:
                    validate_path(grid, result["path"], start, goal, result["cost"])


def test_workloads_keep_starts_fixed_and_balance_goals_with_exact_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_changing_goals import queries_for

    grid = Grid.from_ascii([".....", "#####", ".....", ".....", "....."])
    record = {"map_seed": 12, "layout_id": "example", "kind": "random"}
    config = {"query_seed": 13, "queries_per_session": 4, "goal_counts": [1, 2, 4]}
    workloads = queries_for(grid, record, config)
    assert workloads == queries_for(grid, record, config)
    starts = [q["start"] for q in workloads[0]["queries"]]
    assert len(set(starts)) == 4
    for workload in workloads:
        assert [q["start"] for q in workload["queries"]] == starts
        goals = [q["goal"] for q in workload["queries"]]
        assert len(set(goals)) == workload["goal_count"]
        assert set(starts).isdisjoint(goals)
        assert all(goals.count(g) == 4 // workload["goal_count"] for g in goals)
        for query in workload["queries"]:
            expected = bfs(grid, query["start"], query["goal"])
            assert query["optimal_cost"] == (expected.cost if expected.cost is not None else -1)


def test_workload_verifier_rejects_changed_labels_and_duplicate_cases(monkeypatch):
    import copy
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import report_changing_goals as report
    from run_changing_goals import queries_for

    grid = Grid(4, 4)
    record = {"layout_id": "example", "kind": "random", "map_seed": 3}
    config = {"query_seed": 4, "queries_per_session": 4, "goal_counts": [1, 2, 4]}
    workloads = json.loads(json.dumps(queries_for(grid, record, config)))
    monkeypatch.setattr(report, "grid_for", lambda record: grid)
    assert len(report.verify_workloads([record], config, workloads)[0]) == 3
    bad = copy.deepcopy(workloads)
    bad[0]["queries"][0]["optimal_cost"] += 1
    with pytest.raises(ValueError, match="BFS-label"):
        report.verify_workloads([record], config, bad)
    with pytest.raises(ValueError, match="coverage"):
        report.verify_workloads([record], config, workloads + workloads[:1])
