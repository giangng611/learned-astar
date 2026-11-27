import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from learned_astar.data import reverse_bfs
from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN, distance_scale, encode_inputs
from learned_astar.models.network import RESEARCH_ARCHITECTURE


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("run_consistency_pilot")


def test_consistency_exact_labels_masks_and_goal(pilot):
    grid = Grid.from_ascii(["...#.", "##.#.", "...#."])
    goal = (0, 0)
    truth = reverse_bfs(grid, goal)
    inputs = torch.from_numpy(encode_inputs(occupancy(grid), goal))[None]
    mask = torch.from_numpy(truth >= 0)[None]
    prediction = torch.from_numpy(np.maximum(truth, 0).astype(np.float32))[None]
    prediction /= distance_scale(grid.height, grid.width)
    prediction[0, 0, 0] = 99  # Inference clamps this goal to zero.
    prediction[~mask] = 999  # Walls and disconnected cells must not affect the penalty.
    loss, fraction = pilot.edge_penalty(prediction, inputs, mask)
    assert float(loss) == pytest.approx(0, abs=1e-10)
    assert float(fraction) == 0
    prediction = prediction.clone().requires_grad_()
    empty_loss, _ = pilot.edge_penalty(prediction, inputs, torch.zeros_like(mask))
    empty_loss.backward()
    assert float(empty_loss.detach()) == 0
    assert torch.count_nonzero(prediction.grad) == 0


def test_penalty_gradient_and_directed_fraction(pilot):
    inputs = torch.zeros((1, 5, 1, 2))
    inputs[0, 1, 0, 1] = 1
    prediction = torch.tensor([[[4.0, 8.0]]], requires_grad=True)
    loss, fraction = pilot.edge_penalty(prediction, inputs, torch.ones((1, 1, 2), dtype=torch.bool))
    assert float(loss.detach()) == 9
    assert float(fraction) == 0.5
    loss.backward()
    assert prediction.grad.tolist() == [[[6.0, 0.0]]]


def test_tiny_matched_run_and_fingerprints(pilot, tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "source"
    (source / "frontier").mkdir(parents=True)
    model = DistanceCNN(channels=4)
    checkpoint = source / "frontier/model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.model_config,
            "architecture": RESEARCH_ARCHITECTURE,
        },
        checkpoint,
    )
    pilot.save_json(source / "training.json", {"frontier": {"sha256": pilot.digest(checkpoint)}})
    (source / "protocol.toml").write_text("seed = 7\n")
    grid, goal = Grid.from_ascii(["....", "####", "....", "...."]), (3, 3)
    blocked, truth = occupancy(grid), reverse_bfs(grid, goal)
    np.savez_compressed(source / "map.npz", blocked=blocked)
    record = {
        "directory": str(source),
        "file": "map.npz",
        "status": "ready",
        "sha256": pilot.digest(source / "map.npz"),
        "kind": "walls",
        "group_id": layout_id(blocked, rotations=True),
        "goal": goal,
        "starts": [(0, 0), goal, (2, 0)],
        "optimal_costs": [-1, 0, 4],
    }
    pilot.save_json(source / "policy.json", [record])
    pilot.save_json(source / "selected_records.json", {})
    for split in ("train", "validation"):
        np.savez_compressed(
            source / f"{split}.npz",
            inputs=encode_inputs(blocked, goal)[None],
            targets=np.maximum(truth, 0).astype(np.float32)[None] / 6,
            mask=(truth >= 0)[None],
            pairs=np.zeros((1, 1, 4), np.float32),
            pair_mask=np.zeros((1, 1), bool),
        )
    pilot.save_json(
        source / "fingerprints.json",
        {str(p): pilot.digest(p) for p in source.iterdir() if p.is_file()},
    )
    config = tmp_path / "config.toml"
    config.write_text(
        f'source_run = "{source}"\nseed = 7\nepochs = 1\nbatch_size = 1\n'
        "learning_rate = 0.0002\nroute_weight = 0.02\nfrontier_weight = 0.02\n"
        "consistency_weight = 0.02\ncpu_threads = 1\nrepetitions = 3\n"
    )
    root = tmp_path / "run"
    pilot.prepare(root, config)
    assert pilot.digest(root / "initial.pt") == pilot.digest(checkpoint)
    pilot.train(root)
    for branch in pilot.BRANCHES:
        _, metadata = pilot.load_model(root / branch / "model.pt")
        assert metadata["initial_checkpoint_sha256"] == pilot.digest(checkpoint)
        assert metadata["checkpoint_rule"] == "fixed_final_epoch"
    pilot.evaluate(root)
    report = json.loads((root / "summary.json").read_text())
    assert report["timed_queries"] == 45
    family = report["families"]["walls"]
    assert family["methods"]["manhattan"]["reachable_queries"] == 2
    assert family["methods"]["manhattan"]["optimal_fraction"] == 1
    assert set(family["paired_differences"]) == {"total_ms", "expanded", "reexpanded"}
    with pytest.raises(FileExistsError):
        pilot.train(root)
    with pytest.raises(FileExistsError):
        pilot.evaluate(root)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        pilot.checked(root)


def test_summary_weights_maps_and_preserves_difference_sign(pilot):
    rows = []
    for group, count, difference in (("a", 1, -4), ("b", 3, 2)):
        for q in range(count):
            for method in ("control", "consistency"):
                value = 10 + (difference if method == "consistency" else 0)
                rows.append(
                    {
                        "group_id": group,
                        "kind": "walls",
                        "method": method,
                        "total_ms": value,
                        "expanded": value,
                        "reexpanded": value,
                        "optimal_cost": 1,
                        "cost": 1,
                        "query": q,
                    }
                )
    report = pilot.summarize(rows)["walls"]
    assert report["paired_differences"]["expanded"]["mean_candidate_minus_control"] == -1
