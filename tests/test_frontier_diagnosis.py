import gzip
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid
from learned_astar.search import astar
from learned_astar.search.core import best_first_search


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("diagnose_frontier")


def test_exact_labels_have_no_error_or_consistency_violation(diagnostic):
    grid = Grid.from_ascii(["...#.", "##.#.", "...#."])
    goal = (0, 0)
    truth = reverse_bfs(grid, goal)
    result = diagnostic.field_errors(grid, goal, truth.astype(float), truth)
    assert result["regions"]["detour_required"]["cells"] > 0
    assert result["regions"]["all_reachable"]["learned_mae"] == 0
    assert result["inconsistent_edge_fraction"] == 0
    assert result["max_consistency_violation"] == 0
    assert result["regions"]["all_reachable"]["cells"] == 7


def test_small_diagnosis_and_corrupted_evidence(diagnostic, tmp_path, monkeypatch):
    source, profile, output = [tmp_path / p for p in ("source", "profile", "output")]
    (source / "frontier").mkdir(parents=True)
    profile.mkdir()
    checkpoint = source / "frontier/model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    grid, start, goal = Grid(3, 3), (0, 0), (2, 2)
    truth = reverse_bfs(grid, goal)
    prediction = np.zeros((3, 3), dtype=np.float32)
    policy = [
        {
            "group_id": "tiny",
            "kind": "walls",
            "directory": "unused",
            "starts": [start],
            "goal": goal,
            "optimal_costs": [4],
        }
    ]
    diagnostic.save_json(source / "policy.json", policy)
    diagnostic.save_json(profile / "protocol.json", {"source": str(source), "repetitions": 3})
    diagnostic.save_json(
        profile / "fingerprints.json", {str(checkpoint): diagnostic.digest(checkpoint)}
    )
    rows = []
    for method in ("manhattan", "learned_reference", "learned_profiled"):
        result = (
            astar(grid, start, goal)
            if method == "manhattan"
            else best_first_search(grid, start, goal, lambda cell, _: float(prediction[cell]))
        )
        for rep in range(3):
            rows.append(
                {
                    "group_id": "tiny",
                    "query": 0,
                    "kind": "walls",
                    "method": method,
                    "repetition": rep,
                    "optimal_cost": 4,
                    "status": result.status,
                    "cost": result.cost,
                    "path": result.path,
                    "stats": asdict(result.stats),
                }
            )
    with gzip.open(profile / "queries.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    diagnostic.save_json(
        profile / "summary.json",
        {
            "raw_sha256": diagnostic.digest(profile / "queries.jsonl.gz"),
            "timed_queries": 9,
            "model_sha256": diagnostic.digest(checkpoint),
        },
    )
    monkeypatch.setattr(diagnostic, "checked", lambda _: None)
    monkeypatch.setattr(diagnostic, "configure_torch", lambda *args: None)
    monkeypatch.setattr(diagnostic, "load_model", lambda _: (None, {}))
    monkeypatch.setattr(
        diagnostic, "load_record", lambda *args: {"blocked": np.zeros((3, 3), bool)}
    )
    monkeypatch.setattr(diagnostic, "predict_distances", lambda *args: prediction)
    diagnostic.run(profile, output, 5)
    report = json.loads((output / "summary.json").read_text())
    assert report["wall_queries"] == report["more_expansions"] == 1
    assert report["cases"][0]["replay_matches_saved"]
    assert "S" in (output / "case_1_map.txt").read_text()
    np.testing.assert_array_equal(np.load(output / "case_1_fields.npz")["truth"], truth)
    with pytest.raises(FileExistsError):
        diagnostic.run(profile, output, 5)
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.rank_queries(rows[:-1], policy, 3)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs changed"):
        diagnostic.run(profile, tmp_path / "bad", 5)
