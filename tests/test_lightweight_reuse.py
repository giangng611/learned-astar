import importlib.util
from pathlib import Path

import numpy as np
import pytest

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid
from learned_astar.research.runtime import measure_session, result_signature
from learned_astar.search import SearchLimits

SPEC = importlib.util.spec_from_file_location(
    "reuse_runner", Path(__file__).parents[1] / "scripts/run_lightweight_reuse.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_same_seed_at_different_widths_uses_distinct_models_once_per_session(monkeypatch):
    config = {
        "widths": [16, 8, 4],
        "seeds": [7],
        "models": {str(w): f"width_{w}" for w in (16, 8, 4)},
    }
    specs = runner.model_specs(config)
    mapping = runner.method_map(config)
    assert len(mapping) == 7
    models = {s["id"]: object() for s in specs}
    grid = Grid.from_ascii([".....", "####.", ".....", ".####", "....."])
    goal, starts = (4, 4), [(0, 0), (2, 0), (4, 0)]
    fields = {
        model: np.maximum(reverse_bfs(grid, goal), 0).astype(float) + offset
        for offset, model in enumerate(models.values())
    }
    calls = []

    def predict(model, *_):
        calls.append(model)
        return fields[model].copy()

    monkeypatch.setattr("learned_astar.research.runtime.predict_distances", predict)
    for spec in specs:
        before = len(calls)
        _, actual = measure_session(
            mapping[spec["name"]], grid, goal, starts, models, SearchLimits()
        )
        assert calls[before:] == [models[spec["id"]]]
        _, reference = measure_session(
            f"cached_{spec['id']}", grid, goal, starts, models, SearchLimits()
        )
        assert result_signature(actual) == result_signature(reference)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.model_specs({**config, "widths": [4, 4]})
