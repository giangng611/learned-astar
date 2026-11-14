"""BFS and a shared priority queue search for Dijkstra and A* variants."""

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import count
from time import perf_counter

from learned_astar.config import require_finite, require_int
from learned_astar.environments import Cell, Grid

Heuristic = Callable[[Cell, Cell], float]


class SearchStatus(StrEnum):
    FOUND = "found"
    NO_PATH = "no_path"
    TIME_LIMIT = "time_limit"
    EXPANSION_LIMIT = "expansion_limit"


@dataclass(frozen=True, slots=True)
class SearchLimits:
    max_expansions: int | None = None
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_expansions is not None:
            require_int(self.max_expansions, "max_expansions")
        if self.time_limit_seconds is not None:
            require_finite(self.time_limit_seconds, "time_limit_seconds")


@dataclass(frozen=True, slots=True)
class SearchStats:
    expanded: int
    unique_expanded: int
    generated: int
    queue_insertions: int
    stale_pops: int
    reexpanded: int
    max_frontier: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    status: SearchStatus
    path: tuple[Cell, ...]
    cost: int | None
    elapsed_seconds: float
    stats: SearchStats


class _Run:
    """Keep accounting identical while BFS and heap search use different queues."""

    def __init__(self, start: Cell, limits: SearchLimits) -> None:
        self.started = perf_counter()
        self.limits = limits
        self.parents: dict[Cell, Cell | None] = {start: None}
        self.expanded_cells: set[Cell] = set()
        self.expanded = 0
        self.generated = 0
        self.queue_insertions = 1
        self.stale_pops = 0
        self.max_frontier = 1

    def timed_out(self) -> bool:
        limit = self.limits.time_limit_seconds
        return limit is not None and perf_counter() - self.started >= limit

    def expansion_limit_reached(self) -> bool:
        limit = self.limits.max_expansions
        return limit is not None and self.expanded >= limit

    def expand(self, cell: Cell) -> None:
        self.expanded += 1
        self.expanded_cells.add(cell)

    def inserted(self, frontier_size: int) -> None:
        self.queue_insertions += 1
        self.max_frontier = max(self.max_frontier, frontier_size)

    def finish(self, status: SearchStatus, goal: Cell | None = None) -> SearchResult:
        path: list[Cell] = []
        if status == SearchStatus.FOUND:
            cell = goal
            while cell is not None:
                path.append(cell)
                cell = self.parents[cell]
            path.reverse()
        # Parents can improve after a node is queued. Report the cost of the
        # actual returned route, which can be shorter than that queued g value.
        cost = len(path) - 1 if status == SearchStatus.FOUND else None
        return SearchResult(
            status=status,
            path=tuple(path),
            cost=cost,
            elapsed_seconds=perf_counter() - self.started,
            stats=SearchStats(
                expanded=self.expanded,
                unique_expanded=len(self.expanded_cells),
                generated=self.generated,
                queue_insertions=self.queue_insertions,
                stale_pops=self.stale_pops,
                reexpanded=self.expanded - len(self.expanded_cells),
                max_frontier=self.max_frontier,
            ),
        )


def manhattan(cell: Cell, goal: Cell) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def bfs(grid: Grid, start: Cell, goal: Cell, *, limits: SearchLimits | None = None) -> SearchResult:
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    run = _Run(start, limits or SearchLimits())
    if start == goal:
        return run.finish(SearchStatus.FOUND, goal)
    frontier = deque([start])
    while frontier:
        if run.timed_out():
            return run.finish(SearchStatus.TIME_LIMIT)
        cell = frontier.popleft()
        if cell == goal:
            return run.finish(SearchStatus.FOUND, goal)
        if run.expansion_limit_reached():
            return run.finish(SearchStatus.EXPANSION_LIMIT)
        run.expand(cell)
        for neighbor in grid.neighbors(cell):
            run.generated += 1
            if neighbor not in run.parents:
                run.parents[neighbor] = cell
                frontier.append(neighbor)
                run.inserted(len(frontier))
    return run.finish(SearchStatus.NO_PATH)


def best_first_search(
    grid: Grid,
    start: Cell,
    goal: Cell,
    heuristic: Heuristic,
    *,
    weight: float = 1.0,
    limits: SearchLimits | None = None,
) -> SearchResult:
    """Use (f, -g, insertion order), skip stale entries, and allow reopening."""
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    require_finite(weight, "weight", minimum=1.0)
    run = _Run(start, limits or SearchLimits())
    if start == goal:
        return run.finish(SearchStatus.FOUND, goal)

    def priority(cell: Cell, distance: int) -> float:
        estimate = heuristic(cell, goal)
        require_finite(estimate, "heuristic")
        value = distance + weight * estimate
        require_finite(value, "priority")
        return value

    order = count()
    frontier = [(priority(start, 0), 0, next(order), start)]
    best_g = {start: 0}
    while frontier:
        if run.timed_out():
            return run.finish(SearchStatus.TIME_LIMIT)
        _, negative_g, _, cell = heapq.heappop(frontier)
        distance = -negative_g
        if distance != best_g[cell]:
            run.stale_pops += 1
            continue
        if cell == goal:
            return run.finish(SearchStatus.FOUND, goal)
        if run.expansion_limit_reached():
            return run.finish(SearchStatus.EXPANSION_LIMIT)
        run.expand(cell)
        for neighbor in grid.neighbors(cell):
            run.generated += 1
            next_g = distance + 1
            if neighbor not in best_g or next_g < best_g[neighbor]:
                entry = (priority(neighbor, next_g), -next_g, next(order), neighbor)
                best_g[neighbor] = next_g
                run.parents[neighbor] = cell
                heapq.heappush(frontier, entry)
                run.inserted(len(frontier))
    return run.finish(SearchStatus.NO_PATH)


def dijkstra(
    grid: Grid,
    start: Cell,
    goal: Cell,
    *,
    limits: SearchLimits | None = None,
) -> SearchResult:
    return best_first_search(grid, start, goal, lambda cell, goal: 0, limits=limits)


def astar(
    grid: Grid,
    start: Cell,
    goal: Cell,
    *,
    limits: SearchLimits | None = None,
) -> SearchResult:
    return best_first_search(grid, start, goal, manhattan, limits=limits)


def weighted_astar(
    grid: Grid,
    start: Cell,
    goal: Cell,
    weight: float = 1.5,
    *,
    limits: SearchLimits | None = None,
) -> SearchResult:
    return best_first_search(grid, start, goal, manhattan, weight=weight, limits=limits)
