import numpy as np
import pytest
import torch

from learned_astar.environments import Grid
from learned_astar.models import DistanceCNN, encode_inputs, load_model, predict_distances
from learned_astar.models.network import ARCHITECTURE
from learned_astar.training.train import load_training_tensors, masked_loss


def test_features_preserve_obstacles_goal_and_signed_offsets():
    blocked = np.zeros((3, 5), dtype=bool)
    blocked[0, 0] = True
    channels = encode_inputs(blocked, (1, 2))
    assert channels.shape == (5, 3, 5)
    assert channels[0, 0, 0] == 1
    assert channels[1].sum() == channels[1, 1, 2] == 1
    assert channels[2, 0, 0] == -0.5
    assert channels[3, 0, 0] == -0.5
    assert channels[4, 0, 0] == 0.5
    assert np.isfinite(encode_inputs(np.zeros((1, 1), bool), (0, 0))).all()


def test_masked_loss_weights_examples_equally_and_ignores_blocked_cells():
    prediction = torch.tensor([[[0.0, 0.0]], [[0.0, 1000.0]]], requires_grad=True)
    target = torch.tensor([[[2.0, 2.0]], [[1.0, 0.0]]])
    mask = torch.tensor([[[True, True]], [[True, False]]])
    loss = masked_loss(prediction, target, mask)
    assert loss.item() == 1.0  # (1.5 + 0.5) / 2
    loss.backward()
    assert prediction.grad[1, 0, 1] == 0


def test_model_supports_grid_sizes_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = DistanceCNN().eval()
    path = tmp_path / "model.pt"
    torch.save({"architecture": ARCHITECTURE, "state_dict": model.state_dict(), "seed": 4}, path)
    restored, meta = load_model(path)
    assert meta["seed"] == 4
    assert sum(p.numel() for p in model.parameters()) == 38497
    for height, width in [(1, 1), (16, 16), (32, 32), (64, 64), (7, 11)]:
        grid = Grid(height, width)
        predicted = predict_distances(restored, grid, (0, 0))
        assert predicted.shape == (height, width)
        assert predicted[0, 0] == 0
        assert np.isfinite(predicted).all() and (predicted >= 0).all()
        np.testing.assert_array_equal(predicted, predict_distances(model, grid, (0, 0)))


def test_training_loader_cannot_load_test_split(tmp_path):
    with pytest.raises(ValueError, match="train or validation"):
        load_training_tensors(tmp_path, "test")
