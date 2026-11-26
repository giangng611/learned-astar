import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from learned_astar.data import reverse_bfs
from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN, predict_distances
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.search import SearchLimits
from learned_astar.search.learned import learned_astar


@pytest.fixture
def profiler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("profile_frontier")


def test_prediction_and_search_instrumentation_preserve_behavior(profiler):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = DistanceCNN(channels=4).eval()
    grids = [
        Grid(5, 5),
        Grid.from_ascii([".....", "####.", "....."]),
        Grid.from_ascii([".....", "#####", "....."]),
    ]
    for grid in grids:
        goal = (grid.height - 1, grid.width - 1)
        distances, parts = profiler.timed_prediction(model, grid, goal)
        np.testing.assert_array_equal(distances, predict_distances(model, grid, goal))
        assert all(v >= 0 for v in parts.values())
        for start in [(0, 0), goal]:
            for limits in [SearchLimits(), SearchLimits(1), SearchLimits(time_limit_seconds=1e-12)]:
                expected = learned_astar(grid, start, goal, model, limits=limits)
                result, times = profiler.measure(
                    "learned_profiled", grid, start, goal, model, limits
                )
                assert profiler.signature(result) == profiler.signature(expected)
                assert times["total_seconds"] == pytest.approx(
                    sum(times[k] for k in profiler.PARTS)
                )
                assert all(v >= 0 for v in times.values())


def test_representative_median_preserves_component_sum_and_coverage(profiler):
    rows = []
    # Independent component medians would add to 4, unlike the actual median total (5).
    components = [(1.0, 2.0), (4.0, 1.0), (2.0, 6.0)]
    for method in profiler.METHODS:
        for rep, (a, b) in enumerate(components):
            timing = {"total_seconds": a + b}
            if method == "learned_profiled":
                timing.update(dict.fromkeys(profiler.PARTS, 0.0))
                timing.update(input_seconds=a, cnn_seconds=b)
            rows.append(
                {
                    "group_id": "map",
                    "kind": "walls",
                    "query": 0,
                    "method": method,
                    "repetition": rep,
                    "timing": timing,
                }
            )
    summary = profiler.summarize(rows, 3)["walls"]
    assert summary["mean_total_ms"]["learned_profiled"] == 5000
    assert summary["profile_components_ms"]["input"] == 4000
    assert sum(summary["profile_components_ms"].values()) == 5000
    with pytest.raises(ValueError, match="coverage"):
        profiler.summarize(rows[:-1], 3)


def test_small_complete_profile_and_checkpoint_rejection(profiler, tmp_path):
    source = tmp_path / "source"
    (source / "frontier").mkdir(parents=True)
    torch.manual_seed(7)
    model = DistanceCNN(channels=4)
    checkpoint = source / "frontier/model.pt"
    torch.save(
        {
            "architecture": RESEARCH_ARCHITECTURE,
            "model_config": model.model_config,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    checkpoint_sha = profiler.digest(checkpoint)
    (source / "training.json").write_text(json.dumps({"frontier": {"sha256": checkpoint_sha}}))
    grid = Grid.from_ascii([".....", "#####", "....."])
    blocked = occupancy(grid)
    mapfile = source / "map.npz"
    np.savez_compressed(mapfile, blocked=blocked)
    goal, starts = (2, 4), [(0, 0), (2, 0)]
    truth = reverse_bfs(grid, goal)
    policy = [
        {
            "directory": str(source),
            "file": "map.npz",
            "status": "ready",
            "sha256": profiler.digest(mapfile),
            "kind": "walls",
            "group_id": layout_id(blocked, rotations=True),
            "goal": goal,
            "starts": starts,
            "optimal_costs": [int(truth[s]) for s in starts],
        }
    ]
    (source / "policy.json").write_text(json.dumps(policy))
    (source / "protocol.toml").write_text("seed = 7\n")
    (source / "fingerprints.json").write_text(json.dumps({str(mapfile): profiler.digest(mapfile)}))
    output = tmp_path / "profile"
    profiler.run(source, output, 3)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["trained"] is False and summary["timed_queries"] == 18
    assert summary["profile_matches_reference_paths_and_counters"]
    assert summary["model_sha256"] == checkpoint_sha == profiler.digest(checkpoint)
    assert summary["families"]["walls"]["queries"] == 2
    with pytest.raises(FileExistsError):
        profiler.run(source, output, 3)
    with pytest.raises(ValueError, match="odd repetition"):
        profiler.run(source, tmp_path / "bad_reps", 2)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint"):
        profiler.run(source, tmp_path / "bad_model", 3)
