"""Prepare separate selector development maps and collect checked runtime targets."""

import argparse
import gzip
import hashlib
import json
import platform
import random
import statistics
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from run_demand_heuristic import grid_for, measure, queries_for

from learned_astar.data.dataset import layout_id, occupancy, wall_grid
from learned_astar.environments import Grid, random_grid, validate_path
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json

METHODS = ("manhattan_demand", "landmark_demand", "bfs_policy")


def features(grid, pairs):
    """Use only the map and known batch endpoints, never costs or family names."""
    blocked = occupancy(grid)
    rows, columns = blocked.mean(axis=1), blocked.mean(axis=0)
    return {
        "height": grid.height,
        "width": grid.width,
        "obstacle_fraction": float(blocked.mean()),
        "max_line_obstacles": float(max(rows.max(), columns.max())),
        "line_obstacle_std": float(max(rows.std(), columns.std())),
        "queries": len(pairs),
        "unique_goals": len({goal for _, goal in pairs}),
        "mean_manhattan_fraction": statistics.mean(
            abs(s[0] - g[0]) + abs(s[1] - g[1]) for s, g in pairs
        )
        / max(grid.height + grid.width - 2, 1),
    }


def validate_config(config):
    for key in ("train_per_family", "validation_per_family", "queries_per_session"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["train_per_family"] % 2 or config["validation_per_family"] % 2:
        raise ValueError("Use even split counts to balance random-map densities")
    if config["repetitions"] < 3 or config["repetitions"] % 2 != 1:
        raise ValueError("Use at least three odd repetitions")
    if config["cpu_threads"] != 1:
        raise ValueError("This timing pilot uses one CPU thread")
    goals = config["goal_counts"]
    if (
        not goals
        or len(set(goals)) != len(goals)
        or any(type(g) is not int or g < 1 or config["queries_per_session"] % g for g in goals)
    ):
        raise ValueError("Distinct positive goal counts must divide the query count")
    sizes = config["sizes"]
    if (
        not sizes
        or len(set(sizes)) != len(sizes)
        or any(type(s) is not int or s < 8 for s in sizes)
    ):
        raise ValueError("Use distinct grid sizes of at least eight")


def prepare(root, config_path):
    config = tomllib.loads(config_path.read_text())
    validate_config(config)
    archive = Path(config["exclusion_archive"])
    excluded = set(json.loads((archive / "separation.json").read_text())["excluded_groups"])
    excluded.update(
        r["group_id"] for r in json.loads((archive / "policy_manifest.json").read_text())
    )
    initial = sorted(excluded)
    root.mkdir(parents=True, exist_ok=False)
    (root / "maps").mkdir()
    (root / "measurements").mkdir()
    (root / "protocol.toml").write_bytes(config_path.read_bytes())
    rng, records, workloads = random.Random(config["seed"]), [], []
    for split in ("train", "validation"):
        for size in config["sizes"]:
            for kind in ("random", "walls"):
                for n in range(config[f"{split}_per_family"]):
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
                        enough = size * size - len(grid.blocked) >= (
                            config["queries_per_session"] + max(config["goal_counts"])
                        )
                        if group not in excluded and enough:
                            break
                    else:
                        raise ValueError("Cannot generate enough distinct usable maps")
                    excluded.add(group)
                    identifier = layout_id(blocked)
                    filename = f"maps/{identifier}.npz"
                    np.savez_compressed(root / filename, blocked=blocked)
                    record = {
                        "layout_id": identifier,
                        "group_id": group,
                        "split": split,
                        "map_seed": seed,
                        "kind": kind,
                        "height": size,
                        "width": size,
                        "requested_density": density,
                        "status": "ready",
                        "file": filename,
                        "directory": str(root),
                        "sha256": digest(root / filename),
                    }
                    records.append(record)
                    workloads.extend(queries_for(grid, record, config))
                    print(
                        f"Prepared map {len(records)} ({split}, {kind}, {size}x{size})", flush=True
                    )
    save_json(root / "manifest.json", records)
    save_json(root / "workloads.json", workloads)
    save_json(root / "separation.json", {"excluded_groups": initial, "new_maps": len(records)})
    paths = list(Path("src").rglob("*.py")) + list(Path("scripts").glob("*.py"))
    paths += [archive / "separation.json", archive / "policy_manifest.json"]
    paths += [
        root / name
        for name in ("protocol.toml", "manifest.json", "workloads.json", "separation.json")
    ]
    paths += [root / r["file"] for r in records]
    save_json(root / "fingerprints.json", {str(p): digest(p) for p in sorted(set(paths))})
    print(
        json.dumps(
            {
                "prepared_maps": len(records),
                "excluded_groups": len(initial),
                "workloads": len(workloads),
                "run": str(root),
            },
            indent=2,
        )
    )


def verify_inputs(root):
    frozen = json.loads((root / "fingerprints.json").read_text())
    if any(not Path(p).is_file() or digest(p) != sha for p, sha in frozen.items()):
        raise ValueError("Frozen inputs changed; use a new run directory")


def check_results(grid, workload, results):
    for query, result in zip(workload["queries"], results, strict=True):
        if query["optimal_cost"] >= 0:
            validate_path(
                grid,
                list(map(tuple, result["path"])),
                tuple(query["start"]),
                tuple(query["goal"]),
                result["cost"],
            )
            if result["status"] != "found" or result["cost"] != query["optimal_cost"]:
                raise ValueError("Wrong optimal path")
        elif result["status"] != "no_path" or result["cost"] is not None or result["path"]:
            raise ValueError("Wrong reachability")


def outcome_digest(outcomes):
    return hashlib.sha256(
        json.dumps(outcomes, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def collect(root):
    verify_inputs(root)
    config = tomllib.loads((root / "protocol.toml").read_text())
    configure_torch(7, config["cpu_threads"])
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
    }
    env_path = root / "environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise ValueError("Environment changed; use a new run directory")
    save_json(env_path, environment)
    records = json.loads((root / "manifest.json").read_text())
    workloads = json.loads((root / "workloads.json").read_text())
    indexed = {(w["layout_id"], w["goal_count"]): w for w in workloads}
    for size in config["sizes"]:
        for method in METHODS:
            for _ in range(3):
                measure(method, Grid(size, size), [((0, 0), (size - 1, size - 1))])
    random.Random(config["order_seed"]).shuffle(records)
    begun = time.perf_counter()
    for i, record in enumerate(records, 1):
        destination = root / "measurements" / f"{record['layout_id']}.json.gz"
        if destination.exists():
            print(f"Kept completed map {i}/{len(records)}", flush=True)
            continue
        grid, sessions, outcomes, signatures = grid_for(record), [], {}, {}
        rng = random.Random(config["order_seed"] + record["map_seed"])
        for rep in range(config["repetitions"]):
            jobs = [(g, m) for g in config["goal_counts"] for m in METHODS]
            rng.shuffle(jobs)
            for goals, method in jobs:
                workload = indexed[record["layout_id"], goals]
                pairs = [(tuple(q["start"]), tuple(q["goal"])) for q in workload["queries"]]
                tick = time.perf_counter()
                values = features(grid, pairs)
                feature_seconds = time.perf_counter() - tick
                timing, results = measure(method, grid, pairs)
                check_results(grid, workload, results)
                saved = [
                    {
                        **{k: r[k] for k in ("status", "cost", "path")},
                        "stats": asdict(r["stats"]) if r["stats"] is not None else None,
                    }
                    for r in results
                ]
                sha = outcome_digest(saved)
                if signatures.setdefault((goals, method), sha) != sha:
                    raise ValueError("Repeated results disagree")
                if timing["goals_prepared"] != goals:
                    raise ValueError("Goal cache mismatch")
                sessions.append(
                    {
                        "goal_count": goals,
                        "method": method,
                        "repetition": rep,
                        "features": values,
                        "feature_seconds": feature_seconds,
                        "result_sha256": sha,
                        **timing,
                    }
                )
                if rep == 0:
                    outcomes[f"{goals}:{method}"] = saved
        partial = destination.with_suffix(".partial")
        with gzip.open(partial, "wt") as handle:
            json.dump(
                {"layout_id": record["layout_id"], "sessions": sessions, "outcomes": outcomes},
                handle,
                allow_nan=False,
            )
        partial.replace(destination)
        print(
            f"Measured {i}/{len(records)} maps; this invocation {time.perf_counter() - begun:.1f}s",
            flush=True,
        )
    verify_inputs(root)
    summarize(root)


def summarize(root):
    verify_inputs(root)
    summary_path = root / "summary.json"
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text())["measurement_hashes"]
        if any(not Path(p).is_file() or digest(p) != sha for p, sha in recorded.items()):
            raise ValueError("Changed measurement evidence")
    config = tomllib.loads((root / "protocol.toml").read_text())
    records = json.loads((root / "manifest.json").read_text())
    workloads = json.loads((root / "workloads.json").read_text())
    indexed = {(w["layout_id"], w["goal_count"]): w for w in workloads}
    rows, hashes = [], {}
    expected = {
        (g, m, r)
        for g in config["goal_counts"]
        for m in METHODS
        for r in range(config["repetitions"])
    }
    for record in records:
        path = root / "measurements" / f"{record['layout_id']}.json.gz"
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        sessions = data["sessions"]
        if (
            data["layout_id"] != record["layout_id"]
            or len(sessions) != len(expected)
            or {(s["goal_count"], s["method"], s["repetition"]) for s in sessions} != expected
        ):
            raise ValueError("Incomplete session coverage")
        grid = grid_for(record)
        for goals in config["goal_counts"]:
            workload = indexed[record["layout_id"], goals]
            pairs = [(tuple(q["start"]), tuple(q["goal"])) for q in workload["queries"]]
            values = features(grid, pairs)
            times, feature_times, samples = {}, [], {}
            for method in METHODS:
                subset = [s for s in sessions if s["goal_count"] == goals and s["method"] == method]
                check_results(grid, workload, data["outcomes"][f"{goals}:{method}"])
                sha = outcome_digest(data["outcomes"][f"{goals}:{method}"])
                if {s["result_sha256"] for s in subset} != {sha} or any(
                    s["features"] != values
                    or s["goals_prepared"] != goals
                    or not np.isfinite(s["total_seconds"])
                    or s["total_seconds"] <= 0
                    or not np.isfinite(s["feature_seconds"])
                    or s["feature_seconds"] < 0
                    for s in subset
                ):
                    raise ValueError("Invalid timings, features, or repeated results")
                samples[method] = [s["total_seconds"] for s in subset]
                times[method] = statistics.median(samples[method])
                feature_times.extend(s["feature_seconds"] for s in subset)
            ordered = sorted(METHODS, key=times.get)
            rows.append(
                {
                    "layout_id": record["layout_id"],
                    "group_id": record["group_id"],
                    "split": record["split"],
                    "family": workload["family"],
                    "goal_count": goals,
                    "features": values,
                    "median_seconds": times,
                    "samples_seconds": samples,
                    "feature_seconds": statistics.median(feature_times),
                    "winner": ordered[0],
                    "relative_runner_up_gap": times[ordered[1]] / times[ordered[0]] - 1,
                }
            )
        hashes[str(path)] = digest(path)
    save_json(root / "dataset.json", rows)
    summary = {
        "maps": len(records),
        "rows": len(rows),
        "split_maps": {s: sum(r["split"] == s for r in records) for s in ("train", "validation")},
        "sessions": len(records) * len(expected),
        "timed_queries": len(records) * len(expected) * config["queries_per_session"],
        "all_saved_paths_checked": True,
        "dataset_sha256": digest(root / "dataset.json"),
        "measurement_hashes": hashes,
    }
    save_json(root / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "measurement_hashes"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "measure", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("outputs/selector_pilot"))
    parser.add_argument("--config", type=Path, default=Path("configs/selector_data.toml"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.config)
    elif args.stage == "measure":
        collect(args.run)
    else:
        summarize(args.run)


if __name__ == "__main__":
    main()
