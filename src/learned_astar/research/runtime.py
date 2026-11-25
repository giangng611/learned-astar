"""Measure complete sessions with setup charged equally to all methods."""

import time

from learned_astar.models import predict_distances
from learned_astar.search import SearchStats, SearchStatus, astar, best_first_search
from learned_astar.search.core import manhattan
from learned_astar.search.learned import learned_astar
from learned_astar.search.prepared import PreparedAStar, PreparedGrid, ReversePolicy


def methods(seeds, batch_size):
    result = ["manhattan", "prepared_manhattan", "prepared_exact", "bfs_policy"]
    for seed in seeds:
        result.extend((f"cached_{seed}", f"prepared_{seed}"))
        if batch_size == 1:
            result.append(f"fresh_{seed}")
    return result


def measure_session(name, grid, goal, starts, models, limits):
    """No prepared state enters this function. Return validation data outside the timer."""
    started = time.perf_counter()
    graph_seconds = prediction_seconds = reference_seconds = 0.0
    preprocess_expanded = 0
    planner = policy = field = None
    prepared = None
    if name.startswith("prepared_") or name == "bfs_policy":
        tick = time.perf_counter()
        prepared = PreparedGrid.build(grid)
        graph_seconds = time.perf_counter() - tick
    if name.startswith("cached_") or (
        name.startswith("prepared_") and name.rsplit("_", 1)[1].isdigit()
    ):
        tick = time.perf_counter()
        field = predict_distances(models[int(name.rsplit("_", 1)[1])], grid, goal)
        prediction_seconds = time.perf_counter() - tick
    if name in ("prepared_exact", "bfs_policy"):
        tick = time.perf_counter()
        policy = ReversePolicy(prepared, goal)
        reference_seconds = time.perf_counter() - tick
        preprocess_expanded = policy.preprocess_expanded
    if name.startswith("prepared_"):
        tick = time.perf_counter()
        planner = PreparedAStar(
            prepared,
            goal,
            primary=[max(v, 0) for v in policy.distances] if policy else None,
            ties=field.ravel().tolist() if field is not None else None,
        )
        graph_seconds += time.perf_counter() - tick
    setup_seconds = time.perf_counter() - started
    results = []
    search_started = time.perf_counter()
    for start in starts:
        if name == "bfs_policy":
            path = policy.path(start)
            results.append(
                {
                    "status": SearchStatus.FOUND if path else SearchStatus.NO_PATH,
                    "cost": len(path) - 1 if path else None,
                    "path": path,
                    "stats": None,
                    "path_steps": max(len(path) - 1, 0),
                }
            )
            continue
        if name == "manhattan":
            result = astar(grid, start, goal, limits=limits)
        elif name.startswith("fresh_"):
            result = learned_astar(
                grid,
                start,
                goal,
                models[int(name.rsplit("_", 1)[1])],
                limits=limits,
                heuristic_mode="tie",
            )
        elif name.startswith("cached_"):
            result = best_first_search(
                grid,
                start,
                goal,
                manhattan,
                limits=limits,
                tie_breaker=lambda cell, _: float(field[cell]),
            )
        else:
            result = planner.search(start, limits=limits)
        results.append(
            {
                "status": result.status,
                "cost": result.cost,
                "path": result.path,
                "stats": result.stats,
                "path_steps": None,
            }
        )
    search_seconds = time.perf_counter() - search_started
    total_seconds = time.perf_counter() - started
    return {
        "total_seconds": total_seconds,
        "setup_seconds": setup_seconds,
        "search_seconds": search_seconds,
        "graph_seconds": graph_seconds,
        "prediction_seconds": prediction_seconds,
        "reference_seconds": reference_seconds,
        "preprocess_expanded": preprocess_expanded,
    }, results


def result_signature(results):
    # SearchStats is a dataclass; use fixed field order to make implementation checks exact.
    from dataclasses import astuple

    return [
        (
            r["status"],
            r["cost"],
            r["path"],
            astuple(r["stats"]) if isinstance(r["stats"], SearchStats) else None,
        )
        for r in results
    ]
