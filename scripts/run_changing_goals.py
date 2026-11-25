"""Fixed-size sessions on one map, with a controlled number of distinct goals."""

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
from landmark_probe import LandmarkIndex
from run_landmark_timing import COMPONENTS, METHODS, choose_records, signature
from run_landmark_timing import fingerprints as previous_fingerprints

from learned_astar.data import load_record, reverse_bfs
from learned_astar.environments import Grid, validate_path
from learned_astar.models import load_model, predict_distances
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, SearchStatus
from learned_astar.search.prepared import PreparedAStar, PreparedGrid, ReversePolicy


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
    result = previous_fingerprints(config, records)
    result["scripts/run_changing_goals.py"] = digest("scripts/run_changing_goals.py")
    return result


def measure(name, grid, pairs, model=None, limits=None):
    """Charge graph and landmark setup once, and goal-specific setup on each cache miss."""
    if name not in METHODS:
        raise ValueError("Unknown method")
    if name.endswith("_cnn") and model is None:
        raise ValueError("A resident model is required")
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
                field = None
                if name.endswith("_cnn"):
                    tick = time.perf_counter()
                    field = predict_distances(model, grid, goal)
                    timing["prediction_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                cache[goal] = PreparedAStar(
                    prepared,
                    goal,
                    primary=index.primary(goal) if index is not None else None,
                    ties=field.ravel().tolist() if field is not None else None,
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
    timing["total_seconds"] = time.perf_counter() - begun
    timing["setup_seconds"] = sum(timing[k] for k in COMPONENTS)
    timing["other_seconds"] = (
        timing["total_seconds"] - timing["setup_seconds"] - timing["search_seconds"]
    )
    timing["preprocess_expanded"] = visits
    timing["goals_prepared"] = len(cache)
    return timing, results


def prepare(root, config_path, config):
    records = choose_records(config)
    root.mkdir(parents=True, exist_ok=False)
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
    if records != choose_records(config) or fingerprints(config, records) != frozen:
        raise ValueError("Frozen inputs changed")
    if (
        digest(root / "workloads.json")
        != json.loads((root / "workloads_verified.json").read_text())["sha256"]
    ):
        raise ValueError("Workloads changed")
    if (root / "sessions.jsonl.gz").exists():
        raise ValueError("Measurements already exist")
    configure_torch(7, config["cpu_threads"])
    tick = time.perf_counter()
    model, _ = load_model(config["checkpoint"])
    model_load = time.perf_counter() - tick
    grids = {r["layout_id"]: grid_for(r) for r in records}
    workloads = json.loads((root / "workloads.json").read_text())
    indexed = {(w["layout_id"], w["goal_count"]): w for w in workloads}
    limits = SearchLimits(1000000, 5.0)
    for size in (32, 64):
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
            "model_load_seconds": model_load,
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
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/changing_goals.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/primary_step3"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if tuple(config["methods"]) != METHODS or config["repetitions"] % 2 != 1:
        raise ValueError("Use the fixed methods and odd repetitions")
    if any(config["queries_per_session"] % g for g in config["goal_counts"]):
        raise ValueError("Goal counts must divide the session size")
    if args.stage == "prepare":
        prepare(args.run, args.config, config)
    else:
        if args.config.read_bytes() != (args.run / "protocol.toml").read_bytes():
            raise ValueError("Protocol changed")
        run(args.run, config)


if __name__ == "__main__":
    main()
