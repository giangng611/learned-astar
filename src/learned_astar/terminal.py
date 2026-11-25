"""Draw ASCII maps and compare frozen models in a small command shell."""

import json
import shlex
from dataclasses import asdict
from pathlib import Path

from learned_astar.environments import Grid, random_grid, sample_query
from learned_astar.reproducibility import make_rng


def parse_map(text):
    rows = text.splitlines()
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Map rows must be nonempty and have the same width")
    if max(len(rows), len(rows[0])) > 128:
        raise ValueError("Terminal maps may be at most 128 by 128 cells")
    if any(c not in ".#SG" for row in rows for c in row):
        raise ValueError("Use only ., #, S, and G")
    starts = [(r, c) for r, row in enumerate(rows) for c, v in enumerate(row) if v == "S"]
    goals = [(r, c) for r, row in enumerate(rows) for c, v in enumerate(row) if v == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("A map needs exactly one S and one G")
    grid = Grid.from_ascii([row.replace("S", ".").replace("G", ".") for row in rows])
    return grid, starts[0], goals[0]


def render_map(grid, start, goal, path=()):
    route = set(map(tuple, path))
    rows = []
    for row in range(grid.height):
        symbols = []
        for column in range(grid.width):
            cell = row, column
            symbol = (
                "S"
                if cell == start
                else "G"
                if cell == goal
                else "#"
                if cell in grid.blocked
                else "*"
                if cell in route
                else "."
            )
            symbols.append(symbol)
        rows.append("".join(symbols))
    return "\n".join(rows)


def prepare_models(paths):
    from learned_astar.models import load_model
    from learned_astar.reproducibility import configure_torch

    configure_torch(4550, 1)
    if paths is None:
        paths = sorted(Path("artifacts/checkpoints").glob("seed_*.pt"))
        if not paths:
            paths = sorted(Path("checkpoints/main").glob("seed_*.pt"))
    models = {}
    for path in paths:
        model, metadata = load_model(path)
        name = f"learned_seed_{metadata['seed']}"
        if name in models:
            raise ValueError("Checkpoint seeds must be distinct")
        models[name] = model
    print(f"Loaded {len(models)} learned models. Model loading is outside query timing.")
    return models


def solve_query(grid, start, goal, models, limits, *, heuristic_mode="raw"):
    from learned_astar.data import reverse_bfs
    from learned_astar.evaluation.run import check_result, methods_for_models

    optimal = int(reverse_bfs(grid, goal)[start])
    results = {}
    for name, method in methods_for_models(models, heuristic_mode=heuristic_mode).items():
        # A warm-up removes the first-call setup from the one-query demonstration.
        method(grid, start, goal, limits=limits)
        result = method(grid, start, goal, limits=limits)
        check_result(grid, start, goal, result, optimal, name)
        results[name] = asdict(result)
    return {
        "purpose": "Interactive demonstration; use the full benchmark for runtime conclusions.",
        "map": render_map(grid, start, goal),
        "start": start,
        "goal": goal,
        "optimal_cost": optimal if optimal >= 0 else None,
        "heuristic_mode": heuristic_mode,
        "results": results,
    }


def print_results(report):
    print("\nSame map and endpoints for every method. Times are one-query measurements.")
    print(f"Learned heuristic mode: {report['heuristic_mode']}")
    print(f"{'method':<23} {'status':<16} {'cost':>5} {'ratio':>7} {'expanded':>9} {'ms':>10}")
    for name, result in report["results"].items():
        cost = result["cost"]
        ratio = (
            f"{cost / report['optimal_cost']:.3f}"
            if cost is not None and report["optimal_cost"]
            else "-"
        )
        print(
            f"{name:<23} {result['status']:<16} {str(cost) if cost is not None else '-':>5} "
            f"{ratio:>7} {result['stats']['expanded']:>9} {1000 * result['elapsed_seconds']:>10.3f}"
        )
    print("S=start, G=goal, #=obstacle, *=returned path, .=free")


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_solve(args):
    from learned_astar.search import SearchLimits

    grid, start, goal = parse_map(args.map.read_text())
    models = prepare_models(args.checkpoints)
    report = solve_query(
        grid,
        start,
        goal,
        models,
        SearchLimits(args.max_expansions, args.time_limit),
        heuristic_mode=args.heuristic_mode,
    )
    print(render_map(grid, start, goal))
    print_results(report)
    if args.show_paths:
        for name, result in report["results"].items():
            print(f"\n{name}\n{render_map(grid, start, goal, result['path'])}")
    if args.output:
        write_report(report, args.output)
    return 0


def run_shell(args):
    from learned_astar.search import SearchLimits

    models = prepare_models(args.checkpoints)
    limits = SearchLimits(args.max_expansions, args.time_limit)
    current, report = None, None
    help_text = (
        "draw                  Paste a map; finish with an empty line or .end\n"
        "load FILE             Load an ASCII map\n"
        "generate H W P SEED   Generate a map (P is obstacle fraction)\n"
        "show                  Show the current map\n"
        "run                   Compare all methods\n"
        "path METHOD           Show a returned path\n"
        "save FILE             Save the last comparison as JSON\n"
        "help / quit"
    )
    print("Learned A* shell\n" + help_text)
    while True:
        try:
            command = shlex.split(input("astar> "))
            if not command:
                continue
            name, *values = command
            if name in ("quit", "exit"):
                return 0
            if name == "help":
                print(help_text)
            elif name == "draw" and not values:
                print("Enter equal-width rows using . # S G:")
                rows = []
                while True:
                    row = input("map> ")
                    if row in ("", ".end"):
                        break
                    rows.append(row)
                current = parse_map("\n".join(rows))
                report = None
                print(render_map(*current))
            elif name == "load" and len(values) == 1:
                current = parse_map(Path(values[0]).read_text())
                report = None
                print(render_map(*current))
            elif name == "generate" and len(values) == 4:
                height, width, fraction, seed = (
                    int(values[0]),
                    int(values[1]),
                    float(values[2]),
                    int(values[3]),
                )
                if max(height, width) > 128:
                    raise ValueError("Terminal maps may be at most 128 by 128 cells")
                rng = make_rng(seed)
                grid = random_grid(height, width, fraction, rng)
                current = (grid, *sample_query(grid, rng))
                report = None
                print(render_map(*current))
            elif name == "show" and not values and current:
                print(render_map(*current))
            elif name == "run" and not values and current:
                report = solve_query(*current, models, limits, heuristic_mode=args.heuristic_mode)
                print_results(report)
            elif name == "path" and len(values) == 1 and report:
                if values[0] not in report["results"]:
                    raise ValueError("Use a method name from the comparison table")
                print(render_map(*current, report["results"][values[0]]["path"]))
            elif name == "save" and len(values) == 1 and report:
                write_report(report, values[0])
                print(f"Saved {values[0]}")
            else:
                print("Command unavailable. Load or draw a map first; type help for syntax.")
        except (ValueError, OSError) as error:
            print(f"Error: {error}")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0
