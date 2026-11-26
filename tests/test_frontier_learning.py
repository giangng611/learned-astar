import importlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from learned_astar.data import DatasetConfig, generate_dataset, read_manifest, reverse_bfs
from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN, distance_scale, encode_inputs
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.search import astar


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("frontier_learning"), importlib.import_module(
        "run_frontier_pilot"
    )


def test_traces_match_search_and_teacher_pairs_are_strict_on_small_grids(modules):
    learning, _ = modules
    cells = [(r, c) for r in range(3) for c in range(3)]
    comparisons = 0
    for mask in range(512):
        grid = Grid(3, 3, frozenset(c for i, c in enumerate(cells) if mask & (1 << i)))
        free = list(grid.free_cells())
        for goal in free:
            truth = reverse_bfs(grid, goal)
            for start in free:
                pairs, expanded, cost = learning.frontier_pairs(grid, start, goal, truth, seed=7)
                expected = astar(grid, start, goal)
                assert (expanded, cost) == (expected.stats.expanded, expected.cost)
                for positive, negative, gp, gn in pairs:
                    p, n = divmod(positive, grid.width), divmod(negative, grid.width)
                    assert truth[p] >= 0 and truth[n] >= 0
                    assert gp + truth[p] < gn + truth[n]
                    comparisons += 1
    assert comparisons > 0


def test_pair_loss_uses_g_goal_override_and_empty_masks(modules):
    learning, _ = modules
    inputs = torch.zeros(1, 5, 2, 2)
    inputs[0, 1, 0, 0] = 1
    prediction = torch.tensor([[[100.0, 1.0], [2.0, 3.0]]], requires_grad=True)
    # Goal score is zero at inference, despite the model's large raw goal output.
    pairs = torch.tensor([[[0, 1, 0, 0], [1, 2, 10, 0]]], dtype=torch.float32)
    valid = torch.tensor([[True, False]])
    loss = learning.pair_loss(prediction, inputs, pairs, valid)
    assert float(loss.detach()) == pytest.approx(
        float(torch.nn.functional.softplus(torch.tensor(-2.0)))
    )
    loss.backward()
    assert prediction.grad[0, 0, 0] == 0
    assert prediction.grad[0, 0, 1] < 0
    # The second comparison becomes wrong because queued g reverses the h ordering.
    assert learning.pair_accuracy(prediction, inputs, pairs, torch.ones_like(valid)) == 0.5
    assert learning.pair_loss(prediction, inputs, pairs, torch.zeros_like(valid)) == 0
    assert learning.pair_accuracy(prediction, inputs, pairs, torch.zeros_like(valid)) is None
    padded, mask = learning.padded_pairs([], 4)
    assert padded.shape == (4, 4) and not mask.any()


def test_seeded_wall_comparisons_are_bounded_and_exact_field_orders_them(modules):
    learning, _ = modules
    grid = Grid.from_ascii([".........", "#######..", ".........", "..#######", "........."])
    goal, start = (0, 0), (4, 8)
    truth = reverse_bfs(grid, goal)
    args = dict(seed=41, snapshots=2, negatives=3)
    pairs, expanded, cost = learning.frontier_pairs(grid, start, goal, truth, **args)
    assert learning.frontier_pairs(grid, start, goal, truth, **args) == (pairs, expanded, cost)
    assert 0 < len(pairs) <= 6
    values, valid = learning.padded_pairs(pairs, 6)
    inputs = torch.from_numpy(encode_inputs(occupancy(grid), goal))[None]
    prediction = torch.from_numpy(np.maximum(truth, 0).astype(np.float32))[None] / distance_scale(
        *truth.shape
    )
    assert (
        learning.pair_accuracy(
            prediction, inputs, torch.from_numpy(values)[None], torch.from_numpy(valid)[None]
        )
        == 1
    )


def test_complete_small_pilot_preserves_parent_and_rejects_changes(modules, tmp_path):
    _, runner = modules
    source = tmp_path / "source"
    generate_dataset(
        source,
        DatasetConfig(
            seed=501,
            size=8,
            train_maps=2,
            train_wall_maps=2,
            validation_maps=2,
            validation_wall_maps=2,
            test_maps=0,
            extra_maps=0,
            goals_per_map=1,
        ),
    )
    model = DistanceCNN(channels=4)
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "architecture": RESEARCH_ARCHITECTURE,
            "model_config": model.model_config,
            "state_dict": model.state_dict(),
            "development_layout_groups": [r["group_id"] for r in read_manifest(source)],
            "seed": 7,
            "epoch": 1,
        },
        checkpoint,
    )
    parent_hash = runner.digest(checkpoint)
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    records = []
    for index, kind in enumerate(("random", "walls")):
        # Policy generation is independent of training and uses the runner's 32x32 scope.
        blocked = np.zeros((32, 32), dtype=np.bool_)
        if kind == "walls":
            blocked[15, :31] = True
        else:
            rng = random.Random(777)
            for cell in rng.sample([(r, c) for r in range(32) for c in range(32)], 100):
                blocked[cell] = True
        path = policy_root / f"map{index}.npz"
        np.savez_compressed(path, blocked=blocked)
        records.append(
            {
                "width": 32,
                "height": 32,
                "kind": kind,
                "map_seed": 100 + index,
                "group_id": layout_id(blocked, rotations=True),
                "layout_id": layout_id(blocked),
                "status": "ready",
                "file": path.name,
                "sha256": runner.digest(path),
            }
        )
    (policy_root / "manifest.json").write_text(json.dumps(records))
    config = tmp_path / "config.toml"
    config.write_text(
        f"source = {json.dumps(str(source))}\ncheckpoint = {json.dumps(str(checkpoint))}\n"
        f"policy_sources = [{json.dumps(str(policy_root))}]\n"
        "train_per_kind = 2\nvalidation_per_kind = 2\nseed = 7\nquery_seed = 149501\n"
        "trace_starts = 2\nsnapshots = 2\nnegatives = 2\nepochs = 2\nbatch_size = 4\n"
        "learning_rate = 0.0002\nroute_weight = 0.02\nfrontier_weight = 0.02\n"
        "cpu_threads = 1\nevaluation_starts = 2\nrepetitions = 3\n"
    )
    root = tmp_path / "run"
    runner.prepare(root, config)
    prep = json.loads((root / "preparation.json").read_text())
    assert prep["train"]["maps"] == prep["validation"]["maps"] == 4
    assert prep["policy_maps"] == 2
    runner.train(root)
    runner.evaluate(root)
    assert runner.digest(checkpoint) == parent_hash
    summary = json.loads((root / "summary.json").read_text())
    assert summary["maps"] == 2 and summary["queries"] == 4
    assert summary["fresh_inference_per_query"]
    for family in summary["families"].values():
        assert family["manhattan"]["optimal_fraction"] == 1
        assert family["bfs"]["optimal_fraction"] == 1
    assert len(json.loads((root / "control/history.json").read_text())) == 2
    assert len(json.loads((root / "frontier/history.json").read_text())) == 2
    child = torch.load(root / "frontier/model.pt", weights_only=True)
    assert child["initial_checkpoint_sha256"] == parent_hash
    assert child["checkpoint_rule"] == "fixed_final_epoch"
    with pytest.raises(FileExistsError):
        runner.evaluate(root)
    (root / "initial.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Frozen input"):
        runner.checked(root)
