"""Time a small fixed landmark validation comparison, including all session setup."""

import argparse
import gzip
import hashlib
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
from run_order_pilot import FAMILIES, query_map

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid, validate_path
from learned_astar.models import load_model, predict_distances
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.runtime import result_signature
from learned_astar.search import SearchLimits, SearchStatus
from learned_astar.search.prepared import PreparedAStar, PreparedGrid, ReversePolicy

METHODS = ("manhattan", "manhattan_cnn", "landmark", "landmark_cnn", "bfs_policy")
COMPONENTS = (
    "graph_seconds",
    "landmark_seconds",
    "prediction_seconds",
    "planner_seconds",
    "reference_seconds",
)
TIMINGS = ("total_seconds", "setup_seconds", "search_seconds", *COMPONENTS)


def choose_records(config):
    records = json.loads(Path(config["policy_manifest"]).read_text())
    n = config["maps_per_family"]
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError("Use a positive even map count to balance random-32 densities")
    selected = []
    for family in FAMILIES:
        available = sorted(
            [r for r in records if f"{r['kind']}_{r['width']}" == family],
            key=lambda r: r["layout_id"],
        )
        if family == "random_32":
            chosen = [r for r in available if r["requested_density"] == 0.1][: n // 2]
            chosen += [r for r in available if r["requested_density"] == 0.25][: n // 2]
        else:
            chosen = available[:n]
        if len(chosen) != n or any(r["split"] != "validation" for r in chosen):
            raise ValueError("Missing policy-validation maps")
        selected.extend(chosen)
    if len({r["group_id"] for r in selected}) != len(selected):
        raise ValueError("Repeated layout groups")
    return selected


def fingerprints(config, records):
    paths = list(Path("src").rglob("*.py"))
    paths += [
        Path("scripts") / name
        for name in (
            "run_landmark_timing.py",
            "landmark_probe.py",
            "run_order_pilot.py",
            "order_limit.py",
        )
    ]
    paths += [Path(config["checkpoint"]), Path(config["policy_manifest"])]
    paths += [Path(r["directory"]) / r["file"] for r in records]
    return {str(p): digest(p) for p in sorted(set(paths))}


def measure(name, grid, goal, starts, model=None, limits=None):
    if name not in METHODS:
        raise ValueError("Unknown timing method")
    if name.endswith("_cnn") and model is None:
        raise ValueError("Learned methods require a resident model")
    started = time.perf_counter()
    timing = dict.fromkeys(COMPONENTS, 0.0)
    tick = time.perf_counter()
    prepared = PreparedGrid.build(grid)
    timing["graph_seconds"] = time.perf_counter() - tick
    index = policy = field = None
    if name.startswith("landmark"):
        tick = time.perf_counter()
        index = LandmarkIndex.build(prepared)
        timing["landmark_seconds"] = time.perf_counter() - tick
    if name.endswith("_cnn"):
        tick = time.perf_counter()
        field = predict_distances(model, grid, goal)
        timing["prediction_seconds"] = time.perf_counter() - tick
    if name == "bfs_policy":
        tick = time.perf_counter()
        policy = ReversePolicy(prepared, goal)
        timing["reference_seconds"] = time.perf_counter() - tick
    else:
        tick = time.perf_counter()
        planner = PreparedAStar(
            prepared,
            goal,
            primary=index.primary(goal) if index is not None else None,
            ties=field.ravel().tolist() if field is not None else None,
        )
        timing["planner_seconds"] = time.perf_counter() - tick
    timing["setup_seconds"] = time.perf_counter() - started
    tick = time.perf_counter()
    results = []
    for start in starts:
        if policy is not None:
            path = policy.path(start)
            results.append(
                {
                    "status": SearchStatus.FOUND if path else SearchStatus.NO_PATH,
                    "cost": len(path) - 1 if path else None,
                    "path": path,
                    "stats": None,
                }
            )
        else:
            result = planner.search(start, limits=limits)
            results.append(
                {
                    "status": result.status,
                    "cost": result.cost,
                    "path": result.path,
                    "stats": result.stats,
                }
            )
    timing["search_seconds"] = time.perf_counter() - tick
    timing["total_seconds"] = time.perf_counter() - started
    timing["preprocess_expanded"] = (
        index.preprocess_expanded
        if index is not None
        else policy.preprocess_expanded
        if policy is not None
        else 0
    )
    return timing, results


def signature(results):
    return hashlib.sha256(json.dumps(result_signature(results)).encode()).hexdigest()


def run(root, config):
    records = json.loads((root / "policy_manifest.json").read_text())
    frozen = json.loads((root / "fingerprints.json").read_text())
    if records != choose_records(config) or fingerprints(config, records) != frozen:
        raise ValueError("Frozen source, model or policy data changed")
    if (root / "sessions.jsonl.gz").exists():
        raise ValueError("Measurements already exist")
    configure_torch(7, config["cpu_threads"])
    tick = time.perf_counter()
    model, _ = load_model(config["checkpoint"])
    model_load_seconds = time.perf_counter() - tick
    limit = SearchLimits(1000000, 5.0)
    maximum = max(config["batch_sizes"])
    prepared_data = []
    for record in records:
        if digest(Path(record["directory"]) / record["file"]) != record["sha256"]:
            raise ValueError("Dataset array hash mismatch")
        grid, goal, starts, labels = query_map(Path(record["directory"]), record, maximum)
        if not np.array_equal(reverse_bfs(grid, goal), labels):
            raise ValueError("Stored labels disagree with independent reverse BFS")
        prepared_data.append((record, grid, goal, starts, labels))
    for size in (32, 64):
        for name in METHODS:
            for _ in range(3):
                measure(name, Grid(size, size), (size - 1, size - 1), [(0, 0)], model, limit)
    rng = random.Random(config["order_seed"])
    rng.shuffle(prepared_data)
    sessions_count = query_count = timed_count = 0
    wall_started = time.perf_counter()
    with (
        gzip.open(root / "sessions.jsonl.gz", "wt") as sessions,
        gzip.open(root / "queries.jsonl.gz", "wt") as queries,
    ):
        for number, (record, grid, goal, starts, labels) in enumerate(prepared_data, 1):
            signatures = {}
            # Shuffle all batch sizes and methods within each repeat to reduce ordering bias.
            for rep in range(config["repetitions"]):
                jobs = [(size, name) for size in config["batch_sizes"] for name in METHODS]
                rng.shuffle(jobs)
                for size, name in jobs:
                    timing, results = measure(name, grid, goal, starts[:size], model, limit)
                    for start, result in zip(starts[:size], results, strict=True):
                        optimal = int(labels[start])
                        if optimal >= 0:
                            validate_path(grid, result["path"], start, goal, result["cost"])
                            if result["status"] != "found" or result["cost"] != optimal:
                                raise ValueError("Incorrect optimal cost")
                        elif (
                            result["status"] != "no_path"
                            or result["path"]
                            or result["cost"] is not None
                        ):
                            raise ValueError("Incorrect reachability")
                    sha = signature(results)
                    if signatures.setdefault((size, name), sha) != sha:
                        raise ValueError("Repeated paths or counters changed")
                    base = {
                        "layout_id": record["layout_id"],
                        "family": f"{record['kind']}_{grid.width}",
                        "batch_size": size,
                        "method": name,
                    }
                    sessions.write(
                        json.dumps({**base, "repetition": rep, "result_sha256": sha, **timing})
                        + "\n"
                    )
                    sessions_count += 1
                    timed_count += size
                    if rep == 0 and size == maximum:
                        for q, (start, result) in enumerate(zip(starts, results, strict=True)):
                            queries.write(
                                json.dumps(
                                    {
                                        **base,
                                        "query": q,
                                        "start": start,
                                        "goal": goal,
                                        "optimal_cost": int(labels[start]),
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
                f"Measured maps {number}/{len(records)}; "
                f"elapsed {time.perf_counter() - wall_started:.1f}s",
                flush=True,
            )
    if fingerprints(config, records) != frozen:
        raise ValueError("Sources changed during measurement")
    save_json(
        root / "measurement.json",
        {
            "maps": len(records),
            "sessions": sessions_count,
            "stored_queries": query_count,
            "timed_queries": timed_count,
            "wall_seconds": time.perf_counter() - wall_started,
            "model_load_seconds": model_load_seconds,
            "model_load_scope": (
                "One observed resident-weight load; excludes Python and library startup"
            ),
            "all_correct": True,
            "repeated_paths_and_counts_match": True,
            "labels_recomputed": True,
            "cpu_threads": config["cpu_threads"],
            "torch": str(torch.__version__),
            "sessions_sha256": digest(root / "sessions.jsonl.gz"),
            "queries_sha256": digest(root / "queries.jsonl.gz"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/landmark_timing.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/primary_step2"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if tuple(config["methods"]) != METHODS or config["repetitions"] % 2 != 1:
        raise ValueError("Use the fixed five methods and an odd repetition count")
    if args.stage == "prepare":
        records = choose_records(config)
        args.run.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.config, args.run / "protocol.toml")
        save_json(args.run / "policy_manifest.json", records)
        save_json(args.run / "fingerprints.json", fingerprints(config, records))
        print(f"Frozen {len(records)} validation maps and unchanged model weights.")
    else:
        if args.config.read_bytes() != (args.run / "protocol.toml").read_bytes():
            raise ValueError("Protocol changed")
        run(args.run, config)


if __name__ == "__main__":
    main()
