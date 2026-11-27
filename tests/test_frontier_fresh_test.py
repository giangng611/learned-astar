import copy
import importlib
import json
import random
from pathlib import Path

import pytest
import torch

from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN
from learned_astar.models.network import RESEARCH_ARCHITECTURE


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_frontier_fresh_test")


def test_new_maps_frozen_model_complete_run_and_evidence_checks(runner, tmp_path):
    torch.set_num_threads(1)
    source, exclusion = tmp_path / "source", tmp_path / "excluded"
    (source / "frontier").mkdir(parents=True)
    exclusion.mkdir()
    # Exclude the first proposed layout using its rotation-invariant identifier.
    first_seed = random.Random(19).randrange(2**32)
    first = runner.make_grid(8, "random", 0.1, first_seed)
    forbidden = layout_id(occupancy(first), rotations=True)
    model = DistanceCNN(channels=4)
    checkpoint = source / "frontier/model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": RESEARCH_ARCHITECTURE,
            "model_config": model.model_config,
            "development_layout_groups": [forbidden],
        },
        checkpoint,
    )
    sha = runner.digest(checkpoint)
    runner.save_json(source / "training.json", {"frontier": {"sha256": sha}})
    runner.save_json(source / "fingerprints.json", {str(checkpoint): sha})
    runner.save_json(exclusion / "separation.json", {"excluded_groups": [forbidden]})
    runner.save_json(exclusion / "manifest.json", [])
    runner.save_json(
        exclusion / "fingerprints.json", {str(p): runner.digest(p) for p in exclusion.iterdir()}
    )
    config = tmp_path / "config.toml"
    config.write_text(
        f'source_run = "{source}"\nexclusion_run = "{exclusion}"\n'
        f'checkpoint_sha256 = "{sha}"\nsizes = [8, 10]\nmaps_per_family = 2\n'
        "starts_per_map = 2\nrepetitions = 3\nmap_seed = 19\nquery_seed = 20\n"
        "order_seed = 21\nbootstrap_seed = 22\nmax_expansions = 1000000\n"
        "time_limit_seconds = 5.0\n"
    )
    root = tmp_path / "run"
    runner.prepare(root, config)
    _, records, _ = runner.audit_maps(root)
    assert len(records) == len({r["group_id"] for r in records}) == 8
    assert forbidden not in {r["group_id"] for r in records}
    assert records[0]["map_seed"] != first_seed
    assert runner.digest(root / "model.pt") == sha
    # Preparation reproduces the same maps and endpoints in a different destination.
    repeat = tmp_path / "repeat"
    runner.prepare(repeat, config)
    assert json.loads((repeat / "manifest.json").read_text()) == records
    runner.evaluate(root)
    runner.verify(root)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["maps"] == 8 and summary["timed_queries"] == 192
    assert set(summary["families"]) == {"random_8", "walls_8", "random_10", "walls_10"}
    assert runner.digest(checkpoint) == sha
    with pytest.raises(FileExistsError):
        runner.evaluate(root)
    summary["families"]["walls_8"]["methods"]["learned"]["mean_expanded"] += 1
    runner.save_json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="summary differs"):
        runner.verify(root)
    (root / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.audit_maps(root)


def test_repetition_and_method_coverage(runner):
    records = [{"group_id": "map", "starts": [[0, 0]]}]
    rows = [
        {
            "group_id": "map",
            "query": 0,
            "method": method,
            "repetition": rep,
            "status": "found",
            "cost": 1,
            "path": [[0, 0], [0, 1]],
            "stats": {},
            "total_ms": rep + 1,
        }
        for method in runner.METHODS
        for rep in range(3)
    ]
    result = runner.representatives(rows, records, 3)
    assert len(result) == 4 and all(r["repetition"] == 1 for r in result)
    with pytest.raises(ValueError, match="Repetition coverage"):
        runner.representatives(rows[:-1], records, 3)
    with pytest.raises(ValueError, match="method coverage"):
        runner.representatives(rows[:-3], records, 3)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["cost"] = 2
    with pytest.raises(ValueError, match="Repeated outcomes"):
        runner.representatives(corrupted, records, 3)


def test_unreachable_zero_length_and_incomplete_queries(runner):
    grid = Grid.from_ascii(["...", "###", "..."])
    record = {
        "family": "walls_3",
        "goal": [2, 2],
        "starts": [[0, 0], [2, 2]],
        "optimal_costs": [-1, 0],
    }
    row = {
        "query": 0,
        "family": "walls_3",
        "optimal_cost": -1,
        "status": "no_path",
        "cost": None,
        "path": [],
        "method": "learned",
        "total_ms": 1,
    }
    runner.check_outcome(row, record, grid)
    runner.check_outcome(
        {**row, "query": 1, "optimal_cost": 0, "status": "found", "cost": 0, "path": [[2, 2]]},
        record,
        grid,
    )
    with pytest.raises(ValueError, match="Incomplete"):
        runner.check_outcome({**row, "status": "time_limit"}, record, grid)


def test_summary_keeps_sizes_separate_and_weights_maps(runner):
    rows = []
    for family in ("walls_8", "walls_16"):
        for group, queries, delta in (("a", 1, -4), ("b", 3, 2)):
            for _ in range(queries):
                for method in runner.METHODS:
                    value = 10 + (delta if method == "learned" else 0)
                    rows.append(
                        {
                            "family": family,
                            "group_id": group,
                            "method": method,
                            "total_ms": value,
                            "stats": {"expanded": value, "reexpanded": 0},
                            "status": "found",
                            "cost": 2,
                            "optimal_cost": 2,
                        }
                    )
    result = runner.summarize(rows, 7)
    assert len(result) == 2
    for family in result.values():
        assert family["paired_differences"]["expanded"]["mean_learned_minus_manhattan"] == -1
