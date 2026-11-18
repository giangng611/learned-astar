import gzip
import importlib.util
import json
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


@pytest.fixture
def reporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import report_lightweight_reuse

    return report_lightweight_reuse


def tiny_evidence():
    config = {"widths": [], "seeds": [], "batch_sizes": [1, 2], "repetitions": 3}
    manifest = [{"layout_id": "a", "split": "test"}]
    sessions, queries = [], []
    for size in config["batch_sizes"]:
        for method in runner.CONTROLS:
            for repetition, elapsed in enumerate((3.0, 1.0, 2.0)):
                sessions.append(
                    {
                        "layout_id": "a",
                        "split": "test",
                        "batch_size": size,
                        "method": method,
                        "repetition": repetition,
                        "result_sha256": "x",
                        "total_seconds": elapsed,
                        "setup_seconds": 0.1,
                        "search_seconds": elapsed - 0.2,
                        "graph_seconds": 0.05,
                        "prediction_seconds": 0,
                        "reference_seconds": 0,
                        "preprocess_expanded": 0,
                    }
                )
            for q in range(size):
                queries.append(
                    {
                        "layout_id": "a",
                        "split": "test",
                        "batch_size": size,
                        "method": method,
                        "query": q,
                        "start": [q, 0],
                        "goal": [2, 2],
                        "optimal_cost": 4 - q,
                        "cost": 4 - q,
                        "status": "found",
                        "expanded": None if method == "bfs_policy" else 5 - q,
                        "path_steps": 4 - q if method == "bfs_policy" else None,
                    }
                )
    return config, manifest, sessions, queries


def test_reuse_evidence_requires_complete_sessions_and_exact_repetitions(reporter):
    config, manifest, sessions, _ = tiny_evidence()
    result = reporter.collapse_sessions(sessions, config, manifest)
    assert len(result) == 8
    assert result["a", 2, "manhattan"]["total_seconds"] == 2
    with pytest.raises(ValueError, match="repetitions"):
        reporter.collapse_sessions(sessions[:-1], config, manifest)
    with pytest.raises(ValueError, match="coverage"):
        reporter.collapse_sessions([r for r in sessions if r["batch_size"] == 1], config, manifest)
    altered = [dict(r) for r in sessions]
    altered[0]["result_sha256"] = "changed"
    with pytest.raises(ValueError, match="paths or counts"):
        reporter.collapse_sessions(altered, config, manifest)


def test_reuse_query_verifier_rejects_missing_queries_and_changed_batch_prefixes(
    reporter, tmp_path
):
    config, manifest, _, queries = tiny_evidence()
    path = tmp_path / "queries.jsonl.gz"

    def write(values):
        with gzip.open(path, "wt") as stream:
            for row in values:
                stream.write(json.dumps(row) + "\n")

    write(queries)
    _, _, count = reporter.verify_queries(path, config, manifest)
    assert count == 12
    write(queries[:-1])
    with pytest.raises(ValueError, match="queries"):
        reporter.verify_queries(path, config, manifest)
    altered = [dict(r) for r in queries]
    altered[-2]["start"] = [1, 1]
    write(altered)
    with pytest.raises(ValueError, match="different queries"):
        reporter.verify_queries(path, config, manifest)


def test_reuse_report_rejects_raw_checksum_change(reporter, tmp_path):
    (tmp_path / "protocol.toml").write_text("widths = []\n")
    (tmp_path / "measurement.json").write_text(json.dumps({"sessions_sha256": "wrong"}))
    (tmp_path / "sessions.jsonl.gz").write_bytes(b"altered evidence")
    with pytest.raises(ValueError, match="checksum"):
        reporter.analyze(tmp_path)
