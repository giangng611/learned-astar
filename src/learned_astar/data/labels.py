"""Exact labels, with different markers for walls and unreachable free cells."""

from collections import deque

import numpy as np

from learned_astar.environments import Cell, Grid

BLOCKED = -2
UNREACHABLE = -1


def reverse_bfs(grid: Grid, goal: Cell) -> np.ndarray:
    grid.require_cell(goal, free=True)
    distances = np.full((grid.height, grid.width), UNREACHABLE, dtype=np.int32)
    for cell in grid.blocked:
        distances[cell] = BLOCKED
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for neighbor in grid.neighbors(cell):
            if distances[neighbor] == UNREACHABLE:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def validate_labels(grid: Grid, goal: Cell, labels: np.ndarray) -> None:
    """Check shortest-distance conditions without rerunning the label generator."""
    grid.require_cell(goal, free=True)
    if labels.shape != (grid.height, grid.width) or labels.dtype.kind not in "iu":
        raise ValueError("Distance labels must be an integer grid of the expected shape")
    if labels[goal] != 0:
        raise ValueError("Goal distance must be zero")
    for row in range(grid.height):
        for column in range(grid.width):
            cell = (row, column)
            value = int(labels[cell])
            if cell in grid.blocked:
                if value != BLOCKED:
                    raise ValueError("Blocked-cell marker is wrong")
                continue
            if value < UNREACHABLE or (value == 0 and cell != goal):
                raise ValueError("Invalid free-cell label")
            neighbors = [int(labels[n]) for n in grid.neighbors(cell)]
            if value == UNREACHABLE:
                if any(n >= 0 for n in neighbors):
                    raise ValueError("Unreachable cell touches a reachable cell")
            else:
                if any(n < 0 or abs(n - value) > 1 for n in neighbors):
                    raise ValueError("Adjacent reachable distances are inconsistent")
                if value > 0 and value - 1 not in neighbors:
                    raise ValueError("Distance has no shorter neighbor leading to the goal")
