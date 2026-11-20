"""Exact expansion floor for consistent Manhattan A* on unit-cost grids."""

import numpy as np

from learned_astar.data import reverse_bfs


def expansion_floor(grid, start, goal, goal_distances=None):
    """Return a lower bound and a start-dependent tie field attaining it.

    Goal pops are not expansions. This diagnostic uses BFS from both endpoints;
    it is neither a deployable learned method nor a free runtime baseline.
    """
    ds = reverse_bfs(grid, start)
    dg = reverse_bfs(grid, goal) if goal_distances is None else goal_distances
    cost = int(dg[start])
    if cost < 0:
        return int((ds >= 0).sum()), int((ds >= 0).sum()), np.zeros(ds.shape)
    rr, cc = np.indices(ds.shape)
    h = np.abs(rr - goal[0]) + np.abs(cc - goal[1])
    strict = (ds >= 0) & (ds + h < cost)
    optimal = (ds >= 0) & (dg >= 0) & (ds + dg == cost)
    last_strict = int(ds[strict & optimal].max()) if (strict & optimal).any() else -1
    floor = int(strict.sum()) + cost - last_strict - 1
    scores = np.where(optimal, np.maximum(dg, 0), grid.height * grid.width + cost + 1)
    return floor, int(strict.sum()), scores.astype(float)
