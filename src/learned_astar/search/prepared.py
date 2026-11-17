"""Reusable map-goal search state; every setup cost belongs to the caller's timer."""

import heapq
import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter

from learned_astar.search.core import SearchLimits, SearchResult, SearchStats, SearchStatus


@dataclass(frozen=True)
class PreparedGrid:
    grid: object
    cells: tuple
    neighbors: tuple

    @classmethod
    def build(cls, grid):
        cells = tuple(divmod(i, grid.width) for i in range(grid.height * grid.width))
        neighbors = tuple(
            tuple(r * grid.width + c for r, c in grid.neighbors(cell))
            if cell not in grid.blocked
            else ()
            for cell in cells
        )
        return cls(grid, cells, neighbors)

    def index(self, cell):
        self.grid.require_cell(cell, free=True)
        return cell[0] * self.grid.width + cell[1]


def checked_scores(values, size):
    result = tuple(float(v) for v in values)
    if len(result) != size or any(not math.isfinite(v) or v < 0 for v in result):
        raise ValueError("Scores must contain one finite nonnegative value per cell")
    return result


class PreparedAStar:
    """Same ordering and counters as the reference search, using integer cell IDs.

    A planner is tied to one immutable grid, goal, and frozen score field.
    Construct a new planner when any of them changes. No global cache is used.
    """

    def __init__(self, prepared, goal, *, primary=None, ties=None):
        self.prepared = prepared
        self.goal = prepared.index(goal)
        size = len(prepared.cells)
        self.primary = (
            checked_scores(primary, size)
            if primary is not None
            else tuple(abs(r - goal[0]) + abs(c - goal[1]) for r, c in prepared.cells)
        )
        self.ties = checked_scores(ties, size) if ties is not None else None

    def search(self, start, *, limits=None):
        source = self.prepared.index(start)
        tick = perf_counter()
        limits = limits or SearchLimits()
        n = len(self.prepared.cells)
        best = [n + 1] * n
        parent = [-1] * n
        seen = bytearray(n)
        best[source] = 0
        h, ties, goal = self.primary, self.ties, self.goal
        secondary = ties[source] if ties is not None else 0
        queue = [(h[source], secondary, 0, 0, source)]
        expanded = unique = generated = stale = 0
        inserted = maximum = 1
        deadline = None if limits.time_limit_seconds is None else tick + limits.time_limit_seconds
        status = SearchStatus.NO_PATH
        if source == goal:
            status = SearchStatus.FOUND
        else:
            while queue:
                if deadline is not None and perf_counter() >= deadline:
                    status = SearchStatus.TIME_LIMIT
                    break
                _, _, negative_g, _, cell = heapq.heappop(queue)
                distance = -negative_g
                if best[cell] != distance:
                    stale += 1
                    continue
                if cell == goal:
                    status = SearchStatus.FOUND
                    break
                if limits.max_expansions is not None and expanded >= limits.max_expansions:
                    status = SearchStatus.EXPANSION_LIMIT
                    break
                expanded += 1
                if not seen[cell]:
                    unique += 1
                    seen[cell] = 1
                for neighbor in self.prepared.neighbors[cell]:
                    generated += 1
                    next_g = distance + 1
                    if next_g < best[neighbor]:
                        best[neighbor] = next_g
                        parent[neighbor] = cell
                        secondary = ties[neighbor] if ties is not None else -next_g
                        heapq.heappush(
                            queue, (next_g + h[neighbor], secondary, -next_g, inserted, neighbor)
                        )
                        inserted += 1
                        maximum = max(maximum, len(queue))
        path = []
        if status == SearchStatus.FOUND:
            cell = goal
            while cell != -1:
                path.append(self.prepared.cells[cell])
                cell = parent[cell]
            path.reverse()
        return SearchResult(
            status,
            tuple(path),
            len(path) - 1 if path else None,
            perf_counter() - tick,
            SearchStats(expanded, unique, generated, inserted, stale, expanded - unique, maximum),
        )


class ReversePolicy:
    """Build exact distances and next steps once, then return paths by following them."""

    def __init__(self, prepared, goal):
        self.prepared = prepared
        self.goal = prepared.index(goal)
        distances = [-1] * len(prepared.cells)
        next_step = [-1] * len(prepared.cells)
        distances[self.goal] = 0
        queue = deque([self.goal])
        self.preprocess_expanded = 0
        while queue:
            cell = queue.popleft()
            self.preprocess_expanded += 1
            for neighbor in prepared.neighbors[cell]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[cell] + 1
                    next_step[neighbor] = cell
                    queue.append(neighbor)
        self.distances, self.next_step = tuple(distances), tuple(next_step)

    def path(self, start):
        cell = self.prepared.index(start)
        if self.distances[cell] < 0:
            return ()
        path = [self.prepared.cells[cell]]
        while cell != self.goal:
            cell = self.next_step[cell]
            path.append(self.prepared.cells[cell])
        return tuple(path)
