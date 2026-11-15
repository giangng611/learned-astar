"""Predict normalized distances with the five-channel CNN from the design."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from learned_astar.data.dataset import occupancy
from learned_astar.environments import Cell, Grid

ARCHITECTURE = "dilated-cnn-32-1-2-4-8-16-v1"


def distance_scale(height: int, width: int) -> int:
    return max(height + width - 2, 1)


def encode_inputs(blocked: np.ndarray, goal: Cell) -> np.ndarray:
    if blocked.ndim != 2 or min(blocked.shape) < 1:
        raise ValueError("Occupancy must be a nonempty two-dimensional array")
    height, width = blocked.shape
    row, column = goal
    if not (0 <= row < height and 0 <= column < width) or blocked[goal]:
        raise ValueError("Goal must be a free cell inside the map")
    rows, columns = np.indices(blocked.shape, dtype=np.float32)
    dr, dc = rows - row, columns - column
    marker = np.zeros(blocked.shape, dtype=np.float32)
    marker[goal] = 1
    return np.stack(
        [
            blocked.astype(np.float32),
            marker,
            dr / max(height - 1, 1),
            dc / max(width - 1, 1),
            (np.abs(dr) + np.abs(dc)) / distance_scale(height, width),
        ]
    )


class DistanceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 5
        for dilation in (1, 2, 4, 8, 16):
            layers.extend(
                [nn.Conv2d(channels, 32, 3, padding=dilation, dilation=dilation), nn.ReLU()]
            )
            channels = 32
        layers.extend([nn.Conv2d(32, 1, 1), nn.Softplus()])
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs).squeeze(1)


def load_model(path: str | Path) -> tuple[DistanceCNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported checkpoint architecture")
    model = DistanceCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, {k: v for k, v in checkpoint.items() if k != "state_dict"}


@torch.inference_mode()
def predict_distances(model: DistanceCNN, grid: Grid, goal: Cell) -> np.ndarray:
    grid.require_cell(goal, free=True)
    inputs = torch.from_numpy(encode_inputs(occupancy(grid), goal)).unsqueeze(0)
    prediction = model(inputs).squeeze(0).numpy() * distance_scale(grid.height, grid.width)
    if prediction.shape != (grid.height, grid.width) or not np.isfinite(prediction).all():
        raise ValueError("Model must produce one finite distance per cell")
    if np.any(prediction < 0):
        raise ValueError("Model distances must be nonnegative")
    prediction[goal] = 0.0
    return prediction
