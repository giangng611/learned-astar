from itertools import permutations

import numpy as np
import pytest
import torch

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid
from learned_astar.models import distance_scale
from learned_astar.search import SearchLimits, SearchStatus, astar, best_first_search, bfs
from learned_astar.search.core import manhattan
from learned_astar.search.learned import learned_astar
from learned_astar.training.routes import route_loss


def test_learned_ties_preserve_optimality_on_every_small_layout_and_endpoint_pair():
    cells = [(r, c) for r in range(3) for c in range(3)]
    for mask in range(512):
        grid = Grid(3, 3, frozenset(cell for i, cell in enumerate(cells) if mask & (1 << i)))
        # Deliberately unrelated, inconsistent scores must not change primary ordering.
        scores = {cell: ((17 * cell[0] + 43 * cell[1] + mask) % 13) * 1000.0 for cell in cells}
        for start, goal in permutations(grid.free_cells(), 2):
            reference = bfs(grid, start, goal)
            result = best_first_search(
                grid,
                start,
                goal,
                manhattan,
                tie_breaker=lambda cell, _, values=scores: values[cell],
            )
            assert (result.status, result.cost) == (reference.status, reference.cost)


def test_fifo_is_an_explicit_control_and_default_ties_stay_unchanged():
    grid = Grid(4, 4)
    baseline = astar(grid, (0, 0), (3, 3))
    fifo = best_first_search(grid, (0, 0), (3, 3), manhattan, tie_breaker="fifo")
    assert baseline.cost == fifo.cost == 6
    assert baseline.stats.expanded == 6
    assert fifo.stats.expanded == 15


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_invalid_tie_scores_are_rejected(score):
    with pytest.raises(ValueError, match="tie-breaker"):
        best_first_search(Grid(2, 2), (0, 0), (1, 1), manhattan, tie_breaker=lambda *_: score)


def test_learned_tie_mode_keeps_primary_order_and_obeys_limits(monkeypatch):
    grid = Grid(3, 5)
    field = np.full((3, 5), 1e6)
    field[2] = 0
    field[0, 4] = 0
    calls = []
    monkeypatch.setattr(
        "learned_astar.search.learned.predict_distances", lambda *args: calls.append(1) or field
    )
    result = learned_astar(grid, (0, 0), (0, 4), None, heuristic_mode="tie")
    assert result.cost == 4 and calls == [1]
    limited = learned_astar(
        grid, (0, 0), (0, 4), None, heuristic_mode="tie", limits=SearchLimits(max_expansions=2)
    )
    assert limited.status == SearchStatus.EXPANSION_LIMIT
    assert limited.cost is None


def test_route_loss_prefers_good_moves_and_masks_obstacles():
    grid = Grid(4, 4, frozenset({(1, 1)}))
    labels = reverse_bfs(grid, (3, 3))
    target = torch.from_numpy(np.maximum(labels, 0).astype(np.float32)).unsqueeze(0)
    target /= distance_scale(4, 4)
    mask = torch.from_numpy(labels >= 0).unsqueeze(0)
    good = target.clone().requires_grad_()
    wrong = (1 - target).requires_grad_()
    assert route_loss(good, target, mask) < route_loss(wrong, target, mask)
    loss = route_loss(wrong, target, mask)
    loss.backward()
    assert torch.isfinite(wrong.grad).all()
    assert wrong.grad[0, 1, 1] == 0
    changed = good.detach().clone()
    changed[0, 1, 1] = 1e6
    torch.testing.assert_close(route_loss(changed, target, mask), route_loss(good, target, mask))


def test_route_loss_accepts_multiple_good_moves_and_isolated_goal():
    # At the top-left, both down and right lead to optimal routes.
    target = torch.tensor([[[1.0, 0.5], [0.5, 0.0]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    expected = 2 * torch.log1p(torch.exp(torch.tensor(-2.0))) / 3
    torch.testing.assert_close(route_loss(target, target, mask), expected)
    single = torch.tensor([[[0.2]]], requires_grad=True)
    loss = route_loss(single, torch.zeros_like(single), torch.ones_like(single, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert single.grad.item() == 0


def test_route_training_saves_finite_components_and_loadable_checkpoint(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "train_smoke.py"
    script.write_text("""
import json
from pathlib import Path
import math
from learned_astar.data import DatasetConfig, generate_dataset
from learned_astar.training.train import TrainingConfig, train_models
from learned_astar.models import load_model
root = Path(__file__).parent
generate_dataset(root / "data", DatasetConfig(train_maps=2, validation_maps=2,
    test_maps=0, extra_maps=0, size=8, goals_per_map=1))
config = TrainingConfig(seeds=(7,), epochs=2, batch_size=2, cpu_threads=1,
                        channels=4, route_loss_weight=0.02)
train_models(root / "data", root / "models", config)
rows = json.loads((root / "models/history_7.json").read_text())
assert len(rows) == 2
for row in rows:
    assert all(math.isfinite(value) for value in row.values())
    assert row["train_route_loss"] >= 0
    assert abs(row["train_loss"] - row["train_distance_loss"] - 0.02*row["train_route_loss"]) < 1e-6
model, meta = load_model(root / "models/seed_7.pt")
assert meta["training_config"]["route_loss_weight"] == 0.02
""")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_move_diagnostic_uses_legal_neighbors_and_accepts_all_optimal_directions():
    from learned_astar.models.transforms import manhattan_field
    from learned_astar.research.experiment import error_metrics

    grid = Grid(4, 4, frozenset({(1, 1)}))
    labels = reverse_bfs(grid, (3, 3))
    lower = manhattan_field(labels.shape, (3, 3))
    perfect = error_metrics(labels, labels, lower)
    assert perfect["optimal_move_fraction"] == perfect["branch_move_fraction"] == 1
    reversed_scores = error_metrics(10 - labels, labels, lower)
    assert reversed_scores["optimal_move_fraction"] < 1


def test_policy_evaluation_selection_and_strict_validation_holdout(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "evaluate_smoke.py"
    script.write_text("""
import json
from pathlib import Path
import torch
from learned_astar.data import DatasetConfig, generate_dataset, read_manifest
from learned_astar.models import DistanceCNN
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.research.experiment import digest
from learned_astar.research.search_learning import make_spec, evaluate_stage, choose_tie_family

root = Path(__file__).parent
development = root / "development"
generate_dataset(development, DatasetConfig(train_maps=2, validation_maps=2,
    test_maps=0, extra_maps=0, size=8))
groups = [r["group_id"] for r in read_manifest(development)]
old = root / "old"
run = root / "run"
for family in ("distance", "route", "no_obstacles"):
    for seed in ((7,) if family=="no_obstacles" else (7,17,27)):
        model = DistanceCNN(channels=4, obstacles=family!="no_obstacles")
        label = "no_obstacles" if family=="no_obstacles" else "direct" if seed==7 else "replicates"
        folder = run/"route" if family=="route" else old/label
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": RESEARCH_ARCHITECTURE, "model_config": model.model_config,
            "state_dict": model.state_dict(), "seed": seed, "epoch": 1,
            "validation_loss": 1.0, "development_layout_groups": groups,
            "dataset_manifest_sha256": digest(development/"manifest.jsonl")},
            folder/f"seed_{seed}.pt")
config = {"sources": {"distance_models": str(old)}, "evaluation": {
    "order_seed": 1, "cpu_threads": 1, "repetitions": 1, "max_expansions": 100000,
    "time_limit_seconds": 5, "bootstrap_seed": 2, "bootstrap_samples": 20}}
validation = run / "policy_validation"
generate_dataset(validation, DatasetConfig(train_maps=0, validation_maps=2,
    validation_wall_maps=2, test_maps=0, extra_maps=0, size=8, seed=567), exclude_groups=groups)
spec = make_spec(run, config)
summary = evaluate_stage(validation, spec, run/"validation", config, validation=True)
assert summary["settings"]["raw_runs"] == 4*4*19
for group in summary["splits"].values():
    assert group["methods"]["route_tie_mean"]["optimal_fraction_of_found"] == 1
selection = choose_tie_family(run/"validation/summary.json")
assert selection["quality_gate_passed"]
summary["settings"]["validation"] = False
fake = run/"test_summary.json"
fake.write_text(json.dumps(summary))
try:
    choose_tie_family(fake)
except ValueError as error:
    assert "not test" in str(error)
else:
    raise AssertionError("Selected using test evidence")
path = Path(spec["models"]["distance_seed_7"])
payload = torch.load(path, weights_only=True)
payload["development_layout_groups"].append(read_manifest(validation)[0]["group_id"])
torch.save(payload, path)
try:
    evaluate_stage(validation, spec, run/"bad", config, validation=True)
except ValueError as error:
    assert "overlaps model development" in str(error)
else:
    raise AssertionError("Accepted leaked policy-validation maps")
""")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
