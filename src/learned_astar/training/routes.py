"""Teach local move preferences while retaining distance supervision."""

import torch
from torch.nn import functional as F


def neighbors(values):
    padded = F.pad(values, (1, 1, 1, 1), value=0)
    return torch.stack(
        [
            padded[:, :-2, 1:-1],
            padded[:, 1:-1, 2:],
            padded[:, 2:, 1:-1],
            padded[:, 1:-1, :-2],
        ],
        dim=1,
    )


def route_loss(prediction, target, mask):
    """Negative log probability assigned to any optimal neighboring move.

    Predictions and targets are normalized distances. A one-step temperature
    converts neighbor scores back to steps. All equally good moves are accepted.
    Goals, blocked cells, and unreachable cells do not supply move labels.
    """
    scale = max(prediction.shape[-2] + prediction.shape[-1] - 2, 1)
    valid_neighbors = neighbors(mask)
    good = valid_neighbors & (neighbors(target) < target.unsqueeze(1) - 0.5 / scale)
    active = mask & (target > 0) & good.any(dim=1)
    logits = -neighbors(prediction) * scale
    all_moves = torch.logsumexp(logits.masked_fill(~valid_neighbors, -1e9), dim=1)
    best_moves = torch.logsumexp(logits.masked_fill(~good, -1e9), dim=1)
    values = (all_moves - best_moves) * active
    sizes = active.flatten(1).sum(1).clamp_min(1)
    return (values.flatten(1).sum(1) / sizes).mean()
