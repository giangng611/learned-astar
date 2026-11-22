from pathlib import Path

import numpy as np
import pytest

from learned_astar.environments import Grid
from learned_astar.research.runtime import measure_session, result_signature
from learned_astar.search import SearchLimits
from learned_astar.search.prepared import PreparedAStar, PreparedGrid


def test_session_rebuilds_setup_and_predicts_once_for_all_starts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import run_landmark_timing as runner

    grid = Grid.from_ascii([".....", ".###.", ".....", "#####", "....."])
    goal, starts = (2, 4), [(0, 0), (2, 2), (4, 0), (2, 4)]
    # Deliberately wrong predictions must not change optimal costs.
    field = np.arange(25, dtype=float).reshape(5, 5)[::-1].copy()
    calls = []
    monkeypatch.setattr(runner, "predict_distances", lambda *args: calls.append(1) or field.copy())
    monkeypatch.setattr(
        "learned_astar.research.runtime.predict_distances", lambda *args: field.copy()
    )
    model, limits = object(), SearchLimits()
    prepared = PreparedGrid.build(grid)
    for name in runner.METHODS:
        before = len(calls)
        timing, actual = runner.measure(name, grid, goal, starts, model, limits)
        assert len(calls) - before == int(name.endswith("_cnn"))
        assert timing["total_seconds"] >= timing["setup_seconds"] + timing["search_seconds"]
        assert timing["setup_seconds"] >= sum(timing[k] for k in runner.COMPONENTS)
        if name.startswith("landmark"):
            index = runner.LandmarkIndex.build(prepared)
            planner = PreparedAStar(
                prepared,
                goal,
                primary=index.primary(goal),
                ties=field.ravel() if name.endswith("_cnn") else None,
            )
            expected = [planner.search(start) for start in starts]
            assert [(r["path"], r["stats"]) for r in actual] == [
                (r.path, r.stats) for r in expected
            ]
            assert timing["preprocess_expanded"] == index.preprocess_expanded > 0
        else:
            reference = {
                "manhattan": "prepared_manhattan",
                "manhattan_cnn": "prepared_7",
                "bfs_policy": "bfs_policy",
            }[name]
            _, expected = measure_session(reference, grid, goal, starts, {7: model}, limits)
            assert result_signature(actual) == result_signature(expected)
        assert [r["cost"] for r in actual] == [6, 2, None, 0]
        # A second call cannot reuse the first session's prepared state or prediction.
        _, again = runner.measure(name, grid, goal, starts, model, limits)
        assert runner.signature(actual) == runner.signature(again)
        assert len(calls) - before == 2 * int(name.endswith("_cnn"))


def test_timing_rejects_unknown_methods_and_missing_models(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from run_landmark_timing import measure

    with pytest.raises(ValueError, match="Unknown"):
        measure("other", Grid(2, 2), (1, 1), [(0, 0)])
    with pytest.raises(ValueError, match="resident"):
        measure("landmark_cnn", Grid(2, 2), (1, 1), [(0, 0)])
