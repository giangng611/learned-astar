import numpy as np
import pytest
import torch

from learned_astar.data import (
    DatasetConfig,
    generate_dataset,
    read_manifest,
    reverse_bfs,
    verify_dataset,
)
from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN, encode_inputs, load_model
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.research.experiment import (
    error_metrics,
    execute_query,
    manhattan_field,
    parity_distances,
    select_candidate,
)
from learned_astar.search import SearchLimits


def test_residual_preserves_geometry_and_obstacle_ablation_is_exact(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    blocked = np.zeros((8, 8), bool)
    blocked[3, :6] = True
    inputs = torch.from_numpy(encode_inputs(blocked, (7, 7))).unsqueeze(0)
    hidden = inputs.clone()
    hidden[:, 0] = 0
    model = DistanceCNN(channels=16, residual=True, obstacles=False).eval()
    with torch.inference_mode():
        prediction = model(inputs)
        assert torch.all(prediction >= inputs[:, 4])
        torch.testing.assert_close(prediction, model(hidden), rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": RESEARCH_ARCHITECTURE,
            "model_config": model.model_config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = load_model(path)
    torch.testing.assert_close(restored(inputs), prediction)


def test_mixed_dataset_excludes_previous_layouts_and_preserves_validation_split(tmp_path):
    original = tmp_path / "original"
    config = DatasetConfig(
        train_maps=2,
        validation_maps=2,
        test_maps=0,
        extra_maps=0,
        train_wall_maps=2,
        validation_wall_maps=2,
        size=16,
        goals_per_map=2,
    )
    generate_dataset(original, config)
    excluded = {r["group_id"] for r in read_manifest(original)}
    fresh = tmp_path / "fresh"
    metadata = generate_dataset(fresh, config, exclude_groups=excluded)
    records = read_manifest(fresh)
    assert not excluded & {r["group_id"] for r in records}
    assert metadata["excluded_layout_groups"] == 8
    assert verify_dataset(fresh)["verified_goal_maps"] == 16
    assert {r["kind"] for r in records if r["split"] == "validation"} == {"random", "walls"}


def test_invalid_model_width_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        DistanceCNN(channels=0)


def test_rounding_preserves_manhattan_floor_and_grid_parity():
    goal = (3, 2)
    lower = manhattan_field((4, 5), goal)
    predictions = lower.astype(float)
    predictions[0, 0] -= 3
    predictions[0, 1] += 2.6
    rounded = parity_distances(predictions, goal)
    assert np.all(rounded >= lower)
    assert np.all((rounded - lower) % 2 == 0)
    assert rounded[0, 1] == lower[0, 1] + 2


def test_neighbor_diagnostic_ignores_obstacles_and_detects_inconsistent_estimates():
    labels = np.array([[2, 1, 0], [-2, -2, -2]])
    lower = manhattan_field(labels.shape, (0, 2))
    assert error_metrics(labels, labels, lower)["neighbor_violation_fraction"] == 0
    prediction = np.array([[4, 1, 0], [999, 999, 999]])
    metrics = error_metrics(prediction, labels, lower)
    assert metrics["neighbor_violation_fraction"] == 0.5
    assert metrics["mean_neighbor_violation"] == 1.0


def test_exact_reference_requires_only_route_expansions_and_counts_preprocessing():
    grid = Grid(7, 7, frozenset((r, 3) for r in range(6)))
    goal, start = (0, 6), (0, 0)
    labels = reverse_bfs(grid, goal)
    for mode in ("oracle", "exact"):
        result, parts = execute_query(
            (None, mode), grid, start, goal, labels, SearchLimits(time_limit_seconds=5)
        )
        assert result.cost == result.stats.expanded == 18
        assert result.elapsed_seconds >= parts["prediction_seconds"] + parts["search_seconds"]


def test_demo_rounding_recovers_an_exact_heuristic_with_small_prediction_errors(monkeypatch):
    from learned_astar.search.learned import learned_astar

    grid = Grid(5, 5, frozenset((r, 2) for r in range(4)))
    start, goal = (0, 0), (0, 4)
    labels = reverse_bfs(grid, goal)
    prediction = np.maximum(labels, 0).astype(float) + 0.75
    monkeypatch.setattr("learned_astar.search.learned.predict_distances", lambda *args: prediction)
    result = learned_astar(grid, start, goal, None, heuristic_mode="parity")
    assert result.cost == result.stats.expanded == 12


def test_selection_uses_quality_gate_and_rejects_test_evidence(tmp_path):
    import json
    from copy import deepcopy

    method = {
        "reachable_success_rate": 1,
        "statuses": {"found": 4},
        "queries": 4,
        "path_ratio": {"mean": 1.0},
        "optimal_fraction_of_found": 1.0,
        "paired_95ci": {"reachable_expansion_ratio": {"mean": 0.8}},
    }
    names = [
        f"{variant}_seed_7_{mode}"
        for variant in ("direct", "residual")
        for mode in ("raw", "parity")
    ]
    summary = {
        "settings": {"validation": True},
        "splits": {
            kind: {"methods": {name: deepcopy(method) for name in names}}
            for kind in ("random", "walls")
        },
    }
    for kind in summary["splits"].values():
        kind["methods"]["direct_seed_7_raw"]["path_ratio"]["mean"] = 1.2
        kind["methods"]["residual_seed_7_parity"]["paired_95ci"]["reachable_expansion_ratio"][
            "mean"
        ] = 0.5
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    criteria = {"maximum_mean_path_ratio": 1.02, "minimum_optimal_fraction": 0.9}
    chosen = select_candidate(path, criteria)
    assert (chosen["variant"], chosen["mode"], chosen["quality_gate_passed"]) == (
        "residual",
        "parity",
        True,
    )
    summary["settings"]["validation"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="test results"):
        select_candidate(path, criteria)


def test_research_pipeline_and_fresh_dataset_leakage_guard(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "smoke.py"
    script.write_text("""
import json
from pathlib import Path
import torch
from learned_astar.data import DatasetConfig, generate_dataset, read_manifest
from learned_astar.models import DistanceCNN
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.research.experiment import evaluate_research, digest

root = Path(__file__).parent
development = root / "development"
generate_dataset(development, DatasetConfig(train_maps=2, validation_maps=2,
    validation_wall_maps=2, test_maps=0, extra_maps=0, size=8))
groups = [r["group_id"] for r in read_manifest(development)]
model = DistanceCNN(channels=4, residual=True)
payload = {"architecture": RESEARCH_ARCHITECTURE, "model_config": model.model_config,
    "state_dict": model.state_dict(), "seed": 7, "epoch": 1, "validation_loss": 1.0,
    "development_layout_groups": groups,
    "dataset_manifest_sha256": digest(development / "manifest.jsonl")}
checkpoint = root / "model.pt"
torch.save(payload, checkpoint)
spec = {"models": {"residual_seed_7": str(checkpoint)}, "selection": None}
config = {"evaluation": {"order_seed": 1, "cpu_threads": 1, "repetitions": 2,
    "max_expansions": 100000, "time_limit_seconds": 5,
    "bootstrap_seed": 2, "bootstrap_samples": 20}}
validation = evaluate_research(development, spec, root / "validation", config, validation=True)
assert set(validation["splits"]) == {"random", "walls"}
fresh = root / "fresh"
generate_dataset(fresh, DatasetConfig(train_maps=0, validation_maps=0, test_maps=2,
    extra_maps=0, size=8, seed=987), exclude_groups=groups)
test = evaluate_research(fresh, spec, root / "test", config, validation=False,
                         exclude_groups=groups)
assert test["splits"]["test"]["methods"]["oracle_lookup"]["optimal_fraction_of_found"] == 1
assert test["settings"]["raw_runs"] == 2 * 4 * 8 * 2
payload["development_layout_groups"].append(read_manifest(fresh)[0]["group_id"])
torch.save(payload, checkpoint)
try:
    evaluate_research(fresh, spec, root / "bad", config, validation=False)
except ValueError as error:
    assert "overlaps model development" in str(error)
else:
    raise AssertionError("Accepted a leaked test map")
""")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
