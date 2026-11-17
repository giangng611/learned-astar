"""Include feature preparation and a fresh model prediction in each query timer."""

from dataclasses import replace
from time import perf_counter

from learned_astar.models import predict_distances
from learned_astar.models.transforms import parity_distances
from learned_astar.search.core import (
    SearchLimits,
    SearchResult,
    SearchStats,
    SearchStatus,
    best_first_search,
)


def learned_astar(grid, start, goal, model, *, limits=None, heuristic_mode="raw"):
    if heuristic_mode not in ("raw", "parity"):
        raise ValueError("Heuristic mode must be raw or parity")
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    limits = limits or SearchLimits()
    started = perf_counter()
    distances = predict_distances(model, grid, goal)
    if heuristic_mode == "parity":
        distances = parity_distances(distances, goal)
    if limits.time_limit_seconds is not None:
        remaining = limits.time_limit_seconds - (perf_counter() - started)
        if remaining <= 0:
            return SearchResult(
                SearchStatus.TIME_LIMIT,
                (),
                None,
                perf_counter() - started,
                SearchStats(0, 0, 0, 0, 0, 0, 0),
            )
        limits = replace(limits, time_limit_seconds=remaining)
    result = best_first_search(
        grid, start, goal, lambda cell, _: float(distances[cell]), limits=limits
    )
    return replace(result, elapsed_seconds=perf_counter() - started)
