"""Compare full-field and on-demand scores with frozen changing-goal workloads."""

import argparse
import gzip
import json
import random
import shutil
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from demand_heuristic import DemandAStar
from landmark_probe import LandmarkIndex
from run_landmark_timing import COMPONENTS, choose_records, signature

from learned_astar.data import load_record, read_manifest, reverse_bfs
from learned_astar.data.dataset import layout_id, occupancy, wall_grid
from learned_astar.environments import Grid, random_grid, validate_path
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, SearchStatus
from learned_astar.search.prepared import PreparedAStar, PreparedGrid, ReversePolicy

METHODS = ("manhattan", "manhattan_demand", "landmark", "landmark_demand", "bfs_policy")


def grid_for(record):
    data = load_record(record["directory"], record)
    return Grid(
        *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
    )


def queries_for(grid, record, config):
    """Same unique starts in every workload; goals and starts are disjoint."""
    rng = random.Random(record["map_seed"] + config["query_seed"])
    n, maximum = config["queries_per_session"], max(config["goal_counts"])
    goals = rng.sample(list(grid.free_cells()), maximum)
    excluded = set(goals)
    starts = rng.sample([c for c in grid.free_cells() if c not in excluded], n)
    labels = [reverse_bfs(grid, goal) for goal in goals]
    return [
        {
            "layout_id": record["layout_id"],
            "family": f"{record['kind']}_{grid.width}",
            "goal_count": count,
            "queries": [
                {
                    "start": start,
                    "goal": goals[q % count],
                    "optimal_cost": int(labels[q % count][start]),
                }
                for q, start in enumerate(starts)
            ],
        }
        for count in config["goal_counts"]
    ]


def fingerprints(config, records):
    paths = list(Path("src").rglob("*.py"))
    paths += [
        Path("scripts") / name
        for name in (
            "demand_heuristic.py",
            "run_demand_heuristic.py",
            "landmark_probe.py",
            "run_landmark_timing.py",
            "run_order_pilot.py",
            "order_limit.py",
        )
    ]
    paths += [Path(r["directory"]) / r["file"] for r in records]
    return {str(p): digest(p) for p in sorted(set(paths))}


def measure(name, grid, pairs, model=None, limits=None):
    """Charge graph and landmark setup once, and goal-specific setup on each cache miss."""
    if name not in METHODS:
        raise ValueError("Unknown method")
    begun = time.perf_counter()
    timing = dict.fromkeys((*COMPONENTS, "search_seconds"), 0.0)
    tick = time.perf_counter()
    prepared = PreparedGrid.build(grid)
    timing["graph_seconds"] = time.perf_counter() - tick
    index = None
    if name.startswith("landmark"):
        tick = time.perf_counter()
        index = LandmarkIndex.build(prepared)
        timing["landmark_seconds"] = time.perf_counter() - tick
    visits = index.preprocess_expanded if index is not None else 0
    cache, results = {}, []
    for start, goal in pairs:
        if goal not in cache:
            if name == "bfs_policy":
                tick = time.perf_counter()
                cache[goal] = ReversePolicy(prepared, goal)
                timing["reference_seconds"] += time.perf_counter() - tick
                visits += cache[goal].preprocess_expanded
            else:
                tick = time.perf_counter()
                if name.endswith("_demand"):
                    cache[goal] = DemandAStar(prepared, goal, index=index)
                else:
                    cache[goal] = PreparedAStar(
                        prepared, goal, primary=index.primary(goal) if index else None
                    )
                timing["planner_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        if name == "bfs_policy":
            path = cache[goal].path(start)
            results.append(
                {
                    "status": SearchStatus.FOUND if path else SearchStatus.NO_PATH,
                    "cost": len(path) - 1 if path else None,
                    "path": path,
                    "stats": None,
                }
            )
        else:
            result = cache[goal].search(start, limits=limits)
            results.append(
                {
                    "status": result.status,
                    "cost": result.cost,
                    "path": result.path,
                    "stats": result.stats,
                }
            )
        timing["search_seconds"] += time.perf_counter() - tick
    timing["score_entries"] = (
        sum(len(p.primary) for p in cache.values()) if name != "bfs_policy" else 0
    )
    timing["total_seconds"] = time.perf_counter() - begun
    timing["setup_seconds"] = sum(timing[k] for k in COMPONENTS)
    timing["other_seconds"] = (
        timing["total_seconds"] - timing["setup_seconds"] - timing["search_seconds"]
    )
    timing["preprocess_expanded"] = visits
    timing["goals_prepared"] = len(cache)
    return timing, results


def fresh_records(root, config):
    settings = config["fresh_test"]
    previous = tomllib.loads(Path(settings["exclusion_config"]).read_text())
    sources = previous["sources"] + settings["additional_sources"]
    excluded = {r["group_id"] for source in sources for r in read_manifest(source)}
    initial = sorted(excluded)
    rng = random.Random(settings["seed"])
    records = []
    (root / "maps").mkdir()
    for size in settings["sizes"]:
        for kind in ("random", "walls"):
            for n in range(settings["maps_per_family"]):
                density = (0.1 if n % 2 == 0 else 0.25) if kind == "random" else None
                for _ in range(10000):
                    seed = rng.randrange(2**32)
                    local = random.Random(seed)
                    grid = (
                        wall_grid(size, local)
                        if kind == "walls"
                        else random_grid(size, size, density, local)
                    )
                    blocked = occupancy(grid)
                    group = layout_id(blocked, rotations=True)
                    if group not in excluded:
                        break
                else:
                    raise ValueError("Cannot generate a distinct test map")
                excluded.add(group)
                identifier = layout_id(blocked)
                filename = f"maps/{identifier}.npz"
                np.savez_compressed(root / filename, blocked=blocked)
                records.append(
                    {
                        "layout_id": identifier,
                        "group_id": group,
                        "split": "test",
                        "map_seed": seed,
                        "kind": kind,
                        "height": size,
                        "width": size,
                        "requested_density": density,
                        "actual_density": grid.obstacle_fraction,
                        "status": "ready",
                        "file": filename,
                        "directory": str(root),
                        "sha256": digest(root / filename),
                    }
                )
    save_json(
        root / "separation.json",
        {
            "earlier_groups": len(initial),
            "excluded_groups": initial,
            "new_groups": len(records),
            "sources": {s: digest(Path(s) / "manifest.jsonl") for s in sources},
        },
    )
    return records


def prepare(root, config_path, config, validation=None):
    if validation is not None:
        from report_demand_heuristic import analyze, gate_passed

        if (validation / "protocol.toml").read_bytes() != config_path.read_bytes():
            raise ValueError("Validation protocol changed")
        summary, _ = analyze(validation)
        if not gate_passed(summary):
            raise ValueError("Validation gate failed; do not generate test maps")
    root.mkdir(parents=True, exist_ok=False)
    records = fresh_records(root, config) if validation is not None else choose_records(config)
    save_json(
        root / "scope.json",
        {
            "split": "fresh_test" if validation is not None else "policy_validation",
            "validation_measurement_sha256": digest(validation / "measurement.json")
            if validation
            else None,
        },
    )
    shutil.copy2(config_path, root / "protocol.toml")
    save_json(root / "policy_manifest.json", records)
    save_json(root / "fingerprints.json", fingerprints(config, records))
    workloads = []
    for i, record in enumerate(records, 1):
        workloads.extend(queries_for(grid_for(record), record, config))
        if i % 10 == 0:
            print(f"Prepared labels for {i}/{len(records)} maps", flush=True)
    save_json(root / "workloads.json", workloads)
    save_json(
        root / "workloads_verified.json",
        {"sha256": digest(root / "workloads.json"), "labels": "reverse BFS for every sampled goal"},
    )


def run(root, config):
    records = json.loads((root / "policy_manifest.json").read_text())
    frozen = json.loads((root / "fingerprints.json").read_text())
    if fingerprints(config, records) != frozen:
        raise ValueError("Frozen inputs changed")
    if (
        digest(root / "workloads.json")
        != json.loads((root / "workloads_verified.json").read_text())["sha256"]
    ):
        raise ValueError("Workloads changed")
    if (root / "sessions.jsonl.gz").exists():
        raise ValueError("Measurements already exist")
    configure_torch(7, config["cpu_threads"])
    model = None
    grids = {r["layout_id"]: grid_for(r) for r in records}
    workloads = json.loads((root / "workloads.json").read_text())
    indexed = {(w["layout_id"], w["goal_count"]): w for w in workloads}
    limits = SearchLimits(1000000, 5.0)
    for size in sorted({r["width"] for r in records}):
        pairs = [((0, 0), (size - 1, size - 1)), ((1, 0), (size - 1, size - 2))] * 2
        for name in METHODS:
            for _ in range(3):
                measure(name, Grid(size, size), pairs, model, limits)
    rng = random.Random(config["order_seed"])
    rng.shuffle(records)
    count = query_count = 0
    begun = time.perf_counter()
    with (
        gzip.open(root / "sessions.jsonl.gz", "wt") as sessions,
        gzip.open(root / "queries.jsonl.gz", "wt") as queries,
    ):
        for i, record in enumerate(records, 1):
            grid = grids[record["layout_id"]]
            signatures = {}
            for rep in range(config["repetitions"]):
                jobs = [(g, name) for g in config["goal_counts"] for name in METHODS]
                rng.shuffle(jobs)
                for goals, name in jobs:
                    workload = indexed[record["layout_id"], goals]
                    pairs = [(tuple(q["start"]), tuple(q["goal"])) for q in workload["queries"]]
                    timing, results = measure(name, grid, pairs, model, limits)
                    if timing["goals_prepared"] != goals:
                        raise ValueError("Goal caching disagrees with workload")
                    for q, (start, goal), result in zip(
                        workload["queries"], pairs, results, strict=True
                    ):
                        if q["optimal_cost"] >= 0:
                            validate_path(grid, result["path"], start, goal, result["cost"])
                            if result["cost"] != q["optimal_cost"] or result["status"] != "found":
                                raise ValueError("Wrong optimal path")
                        elif (
                            result["status"] != "no_path"
                            or result["cost"] is not None
                            or result["path"]
                        ):
                            raise ValueError("Wrong reachability")
                    sha = signature(results)
                    if signatures.setdefault((goals, name), sha) != sha:
                        raise ValueError("Repeated paths or counters disagree")
                    base = {
                        "layout_id": record["layout_id"],
                        "family": workload["family"],
                        "goal_count": goals,
                        "method": name,
                    }
                    sessions.write(
                        json.dumps({**base, "repetition": rep, "result_sha256": sha, **timing})
                        + "\n"
                    )
                    count += 1
                    if rep == 0:
                        for n, (q, result) in enumerate(
                            zip(workload["queries"], results, strict=True)
                        ):
                            queries.write(
                                json.dumps(
                                    {
                                        **base,
                                        "query": n,
                                        **q,
                                        "status": result["status"],
                                        "cost": result["cost"],
                                        "path": result["path"],
                                        "stats": asdict(result["stats"])
                                        if result["stats"]
                                        else None,
                                    }
                                )
                                + "\n"
                            )
                            query_count += 1
            for goals in config["goal_counts"]:
                for base in ("manhattan", "landmark"):
                    if signatures[goals, base] != signatures[goals, base + "_demand"]:
                        raise ValueError("On-demand changed paths or search counters")
            print(
                f"Measured {i}/{len(records)} maps in {time.perf_counter() - begun:.1f}s",
                flush=True,
            )
    if fingerprints(config, records) != frozen:
        raise ValueError("Sources changed during measurement")
    save_json(
        root / "measurement.json",
        {
            "maps": len(records),
            "sessions": count,
            "stored_queries": query_count,
            "timed_queries": count * config["queries_per_session"],
            "wall_seconds": time.perf_counter() - begun,
            "eager_demand_paths_and_counts_match": True,
            "all_correct": True,
            "repeated_paths_and_counts_match": True,
            "cpu_threads": config["cpu_threads"],
            "torch": str(torch.__version__),
            "sessions_sha256": digest(root / "sessions.jsonl.gz"),
            "queries_sha256": digest(root / "queries.jsonl.gz"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prepare-test", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/demand_heuristic.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/demand_validation"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/demand_validation"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if tuple(config["methods"]) != METHODS or config["repetitions"] % 2 != 1:
        raise ValueError("Use the fixed methods and odd repetitions")
    if any(config["queries_per_session"] % g for g in config["goal_counts"]):
        raise ValueError("Goal counts must divide the session size")
    if args.stage in ("prepare", "prepare-test"):
        prepare(
            args.run, args.config, config, args.validation if args.stage == "prepare-test" else None
        )
    else:
        if args.config.read_bytes() != (args.run / "protocol.toml").read_bytes():
            raise ValueError("Protocol changed")
        run(args.run, config)


if __name__ == "__main__":
    main()
