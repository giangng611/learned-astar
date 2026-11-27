"""Evaluate one frozen distance CNN on new, excluded-layout test maps."""

import argparse
import gzip
import json
import platform
import random
import shutil
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from learned_astar.data import load_record, reverse_bfs
from learned_astar.data.dataset import layout_id, occupancy, wall_grid
from learned_astar.environments import Grid, random_grid, validate_path
from learned_astar.models import load_model
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, astar, bfs, weighted_astar
from learned_astar.search.learned import learned_astar

METHODS = ("manhattan", "weighted_1.5", "bfs", "learned")


def check_files(root):
    files = json.loads((root / "fingerprints.json").read_text())
    for path, sha in files.items():
        if digest(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    return files


def config_at(root):
    check_files(root)
    return tomllib.loads((root / "protocol.toml").read_text())


def make_grid(size, kind, density, seed):
    rng = random.Random(seed)
    return wall_grid(size, rng) if kind == "walls" else random_grid(size, size, density, rng)


def endpoints(grid, seed, count):
    cells = random.Random(seed).sample(list(grid.free_cells()), count + 1)
    return cells[0], cells[1:]


def prepare(root, config_path):
    config = tomllib.loads(config_path.read_text())
    sizes = config["sizes"]
    if (
        not sizes
        or len(set(sizes)) != len(sizes)
        or any(type(s) is not int or s < 8 for s in sizes)
    ):
        raise ValueError("Use distinct integer sizes of at least eight")
    for key in ("maps_per_family", "starts_per_map", "repetitions", "max_expansions"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Positive integer budgets required")
    if config["maps_per_family"] % 2 or config["repetitions"] < 3 or config["repetitions"] % 2 != 1:
        raise ValueError("Use even map counts and at least three odd repetitions")
    if not np.isfinite(config["time_limit_seconds"]) or config["time_limit_seconds"] <= 0:
        raise ValueError("Positive finite time limit required")
    source, exclusion = Path(config["source_run"]), Path(config["exclusion_run"])
    provenance = {**check_files(source), **check_files(exclusion)}
    checkpoint = source / "frontier/model.pt"
    if digest(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the predeclared model")
    training = json.loads((source / "training.json").read_text())
    if training["frontier"]["sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from training record")
    excluded = set(json.loads((exclusion / "separation.json").read_text())["excluded_groups"])
    excluded.update(r["group_id"] for r in json.loads((exclusion / "manifest.json").read_text()))
    _, metadata = load_model(checkpoint)
    if not set(metadata.get("development_layout_groups", ())).issubset(excluded):
        raise ValueError("Exclusion inventory misses checkpoint development layouts")
    root.mkdir(parents=True, exist_ok=False)
    (root / "maps").mkdir()
    shutil.copyfile(checkpoint, root / "model.pt")
    shutil.copyfile(config_path, root / "protocol.toml")
    save_json(root / "separation.json", {"excluded_groups": sorted(excluded)})
    occupied, records, rng = set(excluded), [], random.Random(config["map_seed"])
    for size in sizes:
        for kind in ("random", "walls"):
            for n in range(config["maps_per_family"]):
                density = (0.1 if n % 2 == 0 else 0.25) if kind == "random" else None
                for _ in range(10000):
                    seed = rng.randrange(2**32)
                    grid = make_grid(size, kind, density, seed)
                    blocked = occupancy(grid)
                    group = layout_id(blocked, rotations=True)
                    if (
                        group not in occupied
                        and size * size - len(grid.blocked) > config["starts_per_map"]
                    ):
                        break
                else:
                    raise ValueError("Unable to generate distinct usable maps")
                occupied.add(group)
                goal, starts = endpoints(
                    grid, seed + config["query_seed"], config["starts_per_map"]
                )
                truth = reverse_bfs(grid, goal)
                filename = f"maps/{group}.npz"
                np.savez_compressed(root / filename, blocked=blocked)
                records.append(
                    {
                        "group_id": group,
                        "kind": kind,
                        "family": f"{kind}_{size}",
                        "height": size,
                        "width": size,
                        "map_seed": seed,
                        "requested_density": density,
                        "goal": goal,
                        "starts": starts,
                        "optimal_costs": [int(truth[s]) for s in starts],
                        "status": "ready",
                        "file": filename,
                        "sha256": digest(root / filename),
                    }
                )
    save_json(root / "manifest.json", records)
    files = list(Path("src").rglob("*.py")) + [
        Path(__file__),
        config_path,
        checkpoint,
        source / "training.json",
        source / "fingerprints.json",
        exclusion / "fingerprints.json",
        exclusion / "manifest.json",
        exclusion / "separation.json",
    ]
    files += [p for p in root.rglob("*") if p.is_file()]
    provenance.update({str(p): digest(p) for p in files})
    save_json(root / "fingerprints.json", provenance)
    audit_maps(root)
    report = {
        "stage": "fresh_test_prepared",
        "trained": False,
        "maps": len(records),
        "excluded_groups": len(excluded),
        "queries": len(records) * config["starts_per_map"],
        "timed_executions": len(records)
        * config["starts_per_map"]
        * len(METHODS)
        * config["repetitions"],
        "checkpoint_sha256": digest(root / "model.pt"),
    }
    save_json(root / "preparation.json", report)
    print(json.dumps(report, indent=2))


def audit_maps(root):
    config = config_at(root)
    if digest(root / "model.pt") != config["checkpoint_sha256"]:
        raise ValueError("Frozen model differs")
    records = json.loads((root / "manifest.json").read_text())
    excluded = set(json.loads((root / "separation.json").read_text())["excluded_groups"])
    seen, maps, counts = set(), {}, {}
    for r in records:
        blocked = load_record(root, r)["blocked"]
        group = layout_id(blocked, rotations=True)
        size, kind = r["height"], r["kind"]
        if size not in config["sizes"] or kind not in ("random", "walls") or r["width"] != size:
            raise ValueError("Unexpected map family")
        grid = make_grid(size, kind, r["requested_density"], r["map_seed"])
        if not np.array_equal(blocked, occupancy(grid)):
            raise ValueError("Map differs from saved generation seed")
        if group != r["group_id"] or group in excluded or group in seen:
            raise ValueError("Duplicate, excluded, or incorrect layout group")
        if r["family"] != f"{kind}_{size}":
            raise ValueError("Incorrect family label")
        seen.add(group)
        key = size, kind, r["requested_density"]
        counts[key] = counts.get(key, 0) + 1
        goal, starts = endpoints(
            grid, r["map_seed"] + config["query_seed"], config["starts_per_map"]
        )
        if list(goal) != r["goal"] or [list(s) for s in starts] != r["starts"]:
            raise ValueError("Endpoints differ from saved query seed")
        truth = reverse_bfs(grid, goal)
        if [int(truth[s]) for s in starts] != r["optimal_costs"]:
            raise ValueError("Saved BFS labels differ")
        maps[group] = grid
    expected = {
        (s, k, d): config["maps_per_family"] // (2 if k == "random" else 1)
        for s in config["sizes"]
        for k, d in (("random", 0.1), ("random", 0.25), ("walls", None))
    }
    if counts != expected:
        raise ValueError("Family counts or density balance differs")
    return config, records, maps


def check_outcome(row, record, grid):
    q = row["query"]
    optimal = record["optimal_costs"][q]
    if row["family"] != record["family"] or row["optimal_cost"] != optimal:
        raise ValueError("Outcome labels differ")
    if row["status"] != ("found" if optimal >= 0 else "no_path"):
        raise ValueError("Incomplete or incorrect reachability")
    if optimal >= 0:
        validate_path(
            grid,
            [tuple(c) for c in row["path"]],
            tuple(record["starts"][q]),
            tuple(record["goal"]),
            row["cost"],
        )
        if row["cost"] < optimal or (
            row["method"] in ("manhattan", "bfs") and row["cost"] != optimal
        ):
            raise ValueError("Incorrect path cost")
        if row["method"] == "weighted_1.5" and row["cost"] > 1.5 * optimal:
            raise ValueError("Weighted path bound exceeded")
    elif row["path"] or row["cost"] is not None:
        raise ValueError("No-path query returned a path")
    if not np.isfinite(row["total_ms"]) or row["total_ms"] <= 0:
        raise ValueError("Invalid timing")


def representatives(rows, records, repetitions):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["group_id"], row["query"], row["method"]), []).append(row)
    expected = {
        (r["group_id"], q, m) for r in records for q in range(len(r["starts"])) for m in METHODS
    }
    if set(grouped) != expected:
        raise ValueError("Query or method coverage differs")
    result = []
    for values in grouped.values():
        if len(values) != repetitions or {v["repetition"] for v in values} != set(
            range(repetitions)
        ):
            raise ValueError("Repetition coverage differs")
        signatures = {
            json.dumps({k: r[k] for k in ("status", "cost", "path", "stats")}, sort_keys=True)
            for r in values
        }
        if len(signatures) != 1:
            raise ValueError("Repeated outcomes differ")
        result.append(sorted(values, key=lambda r: r["total_ms"])[repetitions // 2])
    return result


def summarize(rows, seed):
    report = {}
    for family in sorted({r["family"] for r in rows}):
        selected = [r for r in rows if r["family"] == family]
        groups = sorted({r["group_id"] for r in selected})
        values, methods = {}, {}
        for method in METHODS:
            subset = [r for r in selected if r["method"] == method]
            values[method] = {
                key: np.array(
                    [
                        np.mean(
                            [
                                r["total_ms"] if key == "total_ms" else r["stats"][key]
                                for r in subset
                                if r["group_id"] == group
                            ]
                        )
                        for group in groups
                    ]
                )
                for key in ("total_ms", "expanded", "reexpanded")
            }
            reachable = [r for r in subset if r["optimal_cost"] >= 0]
            ratios = [r["cost"] / r["optimal_cost"] for r in reachable if r["optimal_cost"] > 0]
            methods[method] = {"mean_" + k: float(v.mean()) for k, v in values[method].items()}
            methods[method].update(
                {
                    "queries": len(subset),
                    "reachable_queries": len(reachable),
                    "optimal_paths": sum(r["cost"] == r["optimal_cost"] for r in reachable),
                    "mean_path_ratio": float(np.mean(ratios)) if ratios else None,
                    "max_path_ratio": max(ratios, default=None),
                    "statuses": {
                        s: sum(r["status"] == s for r in subset) for s in ("found", "no_path")
                    },
                }
            )
        paired = {}
        indices = np.random.default_rng(seed).integers(0, len(groups), (2000, len(groups)))
        for key in ("total_ms", "expanded", "reexpanded"):
            delta = values["learned"][key] - values["manhattan"][key]
            paired[key] = {
                "mean_learned_minus_manhattan": float(delta.mean()),
                "whole_map_95ci": np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
            }
        report[family] = {"maps": len(groups), "methods": methods, "paired_differences": paired}
    return report


def evaluate(root):
    config, records, maps = audit_maps(root)
    destination = root / "evaluation.jsonl.gz"
    if destination.exists():
        raise FileExistsError("Evaluation already exists; verify or read the saved result")
    configure_torch(7, 1)
    model, _ = load_model(root / "model.pt")
    limits = SearchLimits(config["max_expansions"], config["time_limit_seconds"])

    def search(method, grid, start, goal):
        if method == "learned":
            return learned_astar(grid, start, goal, model, limits=limits)
        if method == "weighted_1.5":
            return weighted_astar(grid, start, goal, 1.5, limits=limits)
        return (astar if method == "manhattan" else bfs)(grid, start, goal, limits=limits)

    save_json(
        root / "environment.json",
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
        },
    )
    for size in config["sizes"]:
        for method in METHODS:
            for _ in range(3):
                search(method, Grid(size, size), (0, 0), (size - 1, size - 1))
    rng, rows = random.Random(config["order_seed"]), []
    rng.shuffle(records)
    with gzip.open(destination, "wt") as stream:
        for index, record in enumerate(records, 1):
            grid, goal = maps[record["group_id"]], tuple(record["goal"])
            for rep in range(config["repetitions"]):
                jobs = [(q, m) for q in range(len(record["starts"])) for m in METHODS]
                rng.shuffle(jobs)
                for q, method in jobs:
                    start = tuple(record["starts"][q])
                    tick = time.perf_counter()
                    result = search(method, grid, start, goal)
                    total_ms = (time.perf_counter() - tick) * 1000
                    row = {
                        "group_id": record["group_id"],
                        "family": record["family"],
                        "query": q,
                        "method": method,
                        "repetition": rep,
                        "optimal_cost": record["optimal_costs"][q],
                        **asdict(result),
                        "total_ms": total_ms,
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    check_outcome(row, record, grid)
                    rows.append(row)
            print(f"Evaluated {index}/{len(records)} maps", flush=True)
    config_at(root)
    medians = representatives(rows, records, config["repetitions"])
    report = {
        "stage": "frozen_frontier_fresh_map_test",
        "trained": False,
        "training_seeds": 1,
        "fresh_inference_per_query": True,
        "timer": "external_call",
        "maps": len(records),
        "timed_queries": len(rows),
        "checkpoint_sha256": digest(root / "model.pt"),
        "raw_sha256": digest(destination),
        "families": summarize(medians, config["bootstrap_seed"]),
    }
    save_json(root / "summary.json", report)
    print(json.dumps(report, indent=2))


def verify(root):
    config, records, maps = audit_maps(root)
    report = json.loads((root / "summary.json").read_text())
    if (
        digest(root / "evaluation.jsonl.gz") != report["raw_sha256"]
        or digest(root / "model.pt") != report["checkpoint_sha256"]
    ):
        raise ValueError("Evidence checksum differs")
    with gzip.open(root / "evaluation.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    medians = representatives(rows, records, config["repetitions"])
    indexed = {r["group_id"]: r for r in records}
    for row in rows:
        check_outcome(row, indexed[row["group_id"]], maps[row["group_id"]])
    if (
        report["maps"] != len(records)
        or report["timed_queries"] != len(rows)
        or report["families"] != summarize(medians, config["bootstrap_seed"])
    ):
        raise ValueError("Saved summary differs from raw evidence")
    print(
        json.dumps(
            {
                "verified": True,
                "maps": len(records),
                "timed_queries": len(rows),
                "all_saved_paths_checked": True,
                "summary_recomputed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "verify"))
    parser.add_argument("--run", type=Path, default=Path("outputs/frontier_fresh_test"))
    parser.add_argument("--config", type=Path, default=Path("configs/frontier_fresh_test.toml"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.config)
    else:
        {"evaluate": evaluate, "verify": verify}[args.stage](args.run)
