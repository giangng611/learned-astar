"""Run a small, repeatable comparison before building the full experiments."""

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

from learned_astar import __version__
from learned_astar.config import ExperimentConfig, load_config
from learned_astar.environments import Cell, Grid, random_grid, sample_query, validate_path
from learned_astar.reproducibility import make_rng
from learned_astar.search import (
    SearchLimits,
    SearchResult,
    SearchStatus,
    astar,
    bfs,
    dijkstra,
    weighted_astar,
)


def run_demo(config: ExperimentConfig, limits: SearchLimits) -> dict:
    rng = make_rng(config.seed)
    grid = random_grid(config.height, config.width, config.obstacle_fraction, rng)
    start, goal = sample_query(grid, rng)
    results = {
        "bfs": bfs(grid, start, goal, limits=limits),
        "dijkstra": dijkstra(grid, start, goal, limits=limits),
        "astar_manhattan": astar(grid, start, goal, limits=limits),
    }
    weights = {f"weighted_astar_{weight}": weight for weight in config.weights}
    for name, weight in weights.items():
        results[name] = weighted_astar(grid, start, goal, weight, limits=limits)

    reference = results["bfs"]
    records = {}
    for name, result in results.items():
        _check_result(grid, start, goal, name, result, reference, weights.get(name, 1))
        records[name] = asdict(result)
        records[name]["path_cost_ratio"] = (
            result.cost / reference.cost
            if result.cost is not None and reference.cost is not None and reference.cost > 0
            else None
        )
    return {
        "schema_version": 1,
        "purpose": "Single-query smoke check; not a performance benchmark.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_version": __version__,
        },
        "config": asdict(config),
        "limits": asdict(limits),
        "grid": {
            "height": grid.height,
            "width": grid.width,
            "blocked": sorted(grid.blocked),
            "actual_obstacle_fraction": grid.obstacle_fraction,
        },
        "query": {"start": start, "goal": goal},
        "optimal_cost": reference.cost,
        "results": records,
    }


def _check_result(
    grid: Grid,
    start: Cell,
    goal: Cell,
    name: str,
    result: SearchResult,
    reference: SearchResult,
    weight: float,
) -> None:
    # These checks happen after each search timer has stopped.
    if result.status == SearchStatus.FOUND:
        validate_path(grid, result.path, start, goal, result.cost)
    if reference.status == SearchStatus.NO_PATH and result.status == SearchStatus.FOUND:
        raise RuntimeError(f"{name} found a path where BFS found none")
    if reference.status == SearchStatus.FOUND:
        if result.status == SearchStatus.NO_PATH:
            raise RuntimeError(f"{name} missed a path found by BFS")
        if result.status == SearchStatus.FOUND:
            if not reference.cost <= result.cost <= weight * reference.cost:
                raise RuntimeError(f"{name} returned a cost outside its expected bound")


def _print_report(report: dict, show_map: bool) -> None:
    grid, query = report["grid"], report["query"]
    print(
        f"Grid: {grid['height']}x{grid['width']}, blocked: {len(grid['blocked'])}, "
        f"seed: {report['config']['seed']}"
    )
    print(f"Start: {query['start']}, goal: {query['goal']}")
    print("One query only. Timings are a smoke check, not benchmark results.\n")
    print(f"{'algorithm':<22} {'status':<16} {'cost':>5} {'expanded':>9} {'time (ms)':>10}")
    for name, result in report["results"].items():
        cost = "-" if result["cost"] is None else str(result["cost"])
        print(
            f"{name:<22} {result['status']:<16} {cost:>5} "
            f"{result['stats']['expanded']:>9} {result['elapsed_seconds'] * 1000:>10.3f}"
        )
    if show_map:
        blocked = set(grid["blocked"])
        route = set(report["results"]["astar_manhattan"]["path"])
        print("\nS: start, G: goal, #: blocked, *: Manhattan A* path, .: free")
        for row in range(grid["height"]):
            symbols = []
            for column in range(grid["width"]):
                cell = (row, column)
                if cell == query["start"]:
                    symbol = "S"
                elif cell == query["goal"]:
                    symbol = "G"
                elif cell in blocked:
                    symbol = "#"
                elif cell in route:
                    symbol = "*"
                else:
                    symbol = "."
                symbols.append(symbol)
            print("".join(symbols))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Compare the baselines on one generated query")
    demo.add_argument("--config", type=Path, help="TOML settings; otherwise use built-in defaults")
    demo.add_argument("--seed", type=int, help="Override the configured seed")
    demo.add_argument("--max-expansions", type=int)
    demo.add_argument("--time-limit", type=float, help="Soft time limit in seconds per search")
    demo.add_argument(
        "--output", type=Path, help="Save the map, paths, settings, and counts as JSON"
    )
    demo.add_argument(
        "--show-map", action="store_true", help="Print the map and Manhattan A* route"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else ExperimentConfig()
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        report = run_demo(config, SearchLimits(args.max_expansions, args.time_limit))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    _print_report(report, args.show_map)
    if args.output:
        print(f"\nSaved results to {args.output}")
    return 0
