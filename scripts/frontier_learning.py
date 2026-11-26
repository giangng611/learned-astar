"""Collect comparisons from live Manhattan frontiers and supervise their ordering."""

import heapq
import random
from itertools import count

import numpy as np
import torch
from torch.nn import functional as F

from learned_astar.models import distance_scale
from learned_astar.search.core import manhattan


def frontier_pairs(grid, start, goal, distances, *, seed, snapshots=8, negatives=8):
    """Reservoir-sample informative frontiers, accepting all tied teacher minima.

    Teacher scores are queued g plus exact remaining distance. They rank route
    cost, not the unknown number of future search expansions. Unreachable cells
    and equal teacher scores never form a training comparison.
    """
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    if snapshots < 1 or negatives < 1:
        raise ValueError("Positive sample budgets required")
    rng, serial = random.Random(seed), count()
    heap = [(manhattan(start, goal), 0, next(serial), start)]
    best, opened = {start: 0}, {start: 0}
    retained, seen, expanded, last_snapshot = [], 0, 0, -1
    while heap:
        # Every fourth expansion, inspect active entries only (not stale heap copies).
        if expanded % 4 == 0 and expanded != last_snapshot:
            last_snapshot = expanded
            candidates = [
                (c, g, g + int(distances[c])) for c, g in opened.items() if distances[c] >= 0
            ]
            if candidates:
                optimum = min(score for _, _, score in candidates)
                good = [(c, g) for c, g, score in candidates if score == optimum]
                bad = [(c, g) for c, g, score in candidates if score > optimum]
                if bad:
                    positive, gp = rng.choice(good)
                    pairs = [
                        (
                            positive[0] * grid.width + positive[1],
                            cell[0] * grid.width + cell[1],
                            gp,
                            gn,
                        )
                        for cell, gn in rng.sample(bad, min(negatives, len(bad)))
                    ]
                    seen += 1
                    if len(retained) < snapshots:
                        retained.append(pairs)
                    else:
                        index = rng.randrange(seen)
                        if index < snapshots:
                            retained[index] = pairs
        _, negative_g, _, cell = heapq.heappop(heap)
        g = -negative_g
        if opened.get(cell) != g:
            continue
        del opened[cell]
        if cell == goal:
            return [pair for batch in retained for pair in batch], expanded, g
        expanded += 1
        for neighbor in grid.neighbors(cell):
            ng = g + 1
            if ng < best.get(neighbor, float("inf")):
                best[neighbor] = ng
                opened[neighbor] = ng
                heapq.heappush(heap, (ng + manhattan(neighbor, goal), -ng, next(serial), neighbor))
    return [pair for batch in retained for pair in batch], expanded, None


def pair_loss(prediction, inputs, pairs, valid):
    """Match inference's zero goal score; average within examples before the batch."""
    scores = prediction.masked_fill(inputs[:, 1].bool(), 0).flatten(1)
    scores = scores * distance_scale(*prediction.shape[-2:])
    positive = scores.gather(1, pairs[:, :, 0].long()) + pairs[:, :, 2]
    negative = scores.gather(1, pairs[:, :, 1].long()) + pairs[:, :, 3]
    counts = valid.sum(1)
    loss = (F.softplus(positive - negative) * valid).sum(1) / counts.clamp_min(1)
    return loss.mean()


@torch.inference_mode()
def pair_accuracy(prediction, inputs, pairs, valid):
    scores = prediction.masked_fill(inputs[:, 1].bool(), 0).flatten(1)
    scores = scores * distance_scale(*prediction.shape[-2:])
    delta = (
        scores.gather(1, pairs[:, :, 0].long())
        + pairs[:, :, 2]
        - scores.gather(1, pairs[:, :, 1].long())
        - pairs[:, :, 3]
    )
    return float(((delta < 0) * valid).sum() / valid.sum()) if valid.sum() else None


def padded_pairs(values, maximum):
    array = np.zeros((maximum, 4), dtype=np.float32)
    mask = np.zeros(maximum, dtype=np.bool_)
    if values:
        array[: len(values)] = values
    mask[: len(values)] = True
    return array, mask
