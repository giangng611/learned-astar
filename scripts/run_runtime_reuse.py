"""Freeze and run the repeated-query experiment on unseen layout groups."""

import argparse
import gzip
import hashlib
import json
import random
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from learned_astar.data import (
    DatasetConfig,
    generate_dataset,
    load_record,
    read_manifest,
    verify_dataset,
)
from learned_astar.environments import Grid, validate_path
from learned_astar.models import load_model
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.runtime import measure_session, methods, result_signature
from learned_astar.search import SearchLimits

CODE = (
    "src/learned_astar/search/core.py",
    "src/learned_astar/search/prepared.py",
    "src/learned_astar/search/learned.py",
    "src/learned_astar/models/network.py",
    "src/learned_astar/research/runtime.py",
    "scripts/run_runtime_reuse.py",
)


def fingerprints(config):
    paths = list(CODE) + [str(Path(p) / "manifest.jsonl") for p in config["sources"]]
    paths += [str(Path(config["models"]) / f"seed_{seed}.pt") for seed in config["seeds"]]
    return {p: digest(p) for p in paths}


def run(root, config):
    torch.set_num_threads(config["cpu_threads"])
    models = {
        seed: load_model(Path(config["models"]) / f"seed_{seed}.pt")[0] for seed in config["seeds"]
    }
    records = read_manifest(root / "fresh_test")
    limits = SearchLimits(config["max_expansions"], config["time_limit_seconds"])
    for size in (16, 32, 64):
        grid = Grid(size, size)
        for name in methods(config["seeds"], 1):
            for _ in range(3):
                measure_session(name, grid, (size - 1, size - 1), [(0, 0)], models, limits)
    rng = random.Random(config["order_seed"])
    raw, queries = [], []
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        data = load_record(root / "fresh_test", record)
        grid = Grid(
            *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
        )
        goal = tuple(map(int, data["goals"][0]))
        starts = [tuple(map(int, s)) for s in data["starts"]]
        if len(starts) < max(config["batch_sizes"]):
            raise ValueError("Insufficient distinct starts")
        signatures = {}
        for size in config["batch_sizes"]:
            for repetition in range(config["repetitions"]):
                order = methods(config["seeds"], size)
                rng.shuffle(order)
                for name in order:
                    timing, results = measure_session(
                        name, grid, goal, starts[:size], models, limits
                    )
                    for start, result in zip(starts[:size], results, strict=True):
                        optimal = int(data["distances"][0][start])
                        if optimal < 0:
                            if result["status"] != "no_path" or result["path"]:
                                raise ValueError(
                                    "Incorrect unreachable result or unfinished search"
                                )
                        else:
                            validate_path(grid, result["path"], start, goal, result["cost"])
                            if result["status"] != "found" or result["cost"] != optimal:
                                raise ValueError("Incorrect or unfinished reachable query")
                    encoded = json.dumps(result_signature(results), separators=(",", ":")).encode()
                    signature = hashlib.sha256(encoded).hexdigest()
                    key = (size, name)
                    if key in signatures and signatures[key] != signature:
                        raise ValueError("Repeated session changed paths or counts")
                    signatures[key] = signature
                    raw.append(
                        {
                            "layout_id": record["layout_id"],
                            "split": record["split"],
                            "batch_size": size,
                            "repetition": repetition,
                            "method": name,
                            "result_sha256": signature,
                            **timing,
                        }
                    )
                    if repetition == 0:
                        for q, (start, result) in enumerate(
                            zip(starts[:size], results, strict=True)
                        ):
                            queries.append(
                                {
                                    "layout_id": record["layout_id"],
                                    "split": record["split"],
                                    "batch_size": size,
                                    "method": name,
                                    "query": q,
                                    "start": start,
                                    "goal": goal,
                                    "optimal_cost": int(data["distances"][0][start]),
                                    "status": result["status"],
                                    "cost": result["cost"],
                                    "expanded": result["stats"].expanded
                                    if result["stats"]
                                    else None,
                                    "path_steps": result["path_steps"],
                                }
                            )
            if signatures[size, "manhattan"] != signatures[size, "prepared_manhattan"]:
                raise ValueError("Prepared Manhattan changed search behavior")
            for seed in config["seeds"]:
                if signatures[size, f"cached_{seed}"] != signatures[size, f"prepared_{seed}"]:
                    raise ValueError("Prepared learned search changed paths or counts")
                if (
                    size == 1
                    and signatures[size, f"fresh_{seed}"] != signatures[size, f"cached_{seed}"]
                ):
                    raise ValueError("Prediction reuse changed search behavior")
        if index % 25 == 0 or index == len(records):
            print(
                f"maps={index}/{len(records)} seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    measured_seconds = time.perf_counter() - started
    for name, values in (("sessions", raw), ("queries", queries)):
        with gzip.open(root / f"{name}.jsonl.gz", "wt") as stream:
            for row in values:
                stream.write(json.dumps(row) + "\n")
    save_json(
        root / "measurement.json",
        {
            "maps": len(records),
            "sessions": len(raw),
            "timed_queries": sum(r["batch_size"] for r in raw),
            "query_records": len(queries),
            "seconds": measured_seconds,
            "sessions_sha256": digest(root / "sessions.jsonl.gz"),
            "queries_sha256": digest(root / "queries.jsonl.gz"),
            "dataset_manifest_sha256": digest(root / "fresh_test/manifest.jsonl"),
            "all_optimal_and_correct_reachability": True,
            "all_prepared_paths_and_counts_match_reference": True,
            "environment": {
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": "cpu",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/runtime_reuse.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v4"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if args.stage == "prepare":
        args.run.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.config, args.run / "protocol.toml")
        save_json(args.run / "fingerprints.json", fingerprints(config))
        excluded = {r["group_id"] for p in config["sources"] for r in read_manifest(p)}
        generate_dataset(
            args.run / "fresh_test", DatasetConfig(**config["dataset"]), exclude_groups=excluded
        )
        verified = verify_dataset(args.run / "fresh_test")
        fresh = {r["group_id"] for r in read_manifest(args.run / "fresh_test")}
        if excluded & fresh:
            raise ValueError("Fresh layouts overlap earlier maps")
        verified.update(excluded_groups=len(excluded), disjoint=True)
        save_json(args.run / "dataset_verified.json", verified)
        return
    if (args.run / "protocol.toml").read_bytes() != args.config.read_bytes():
        raise ValueError("Protocol changed")
    if fingerprints(config) != json.loads((args.run / "fingerprints.json").read_text()):
        raise ValueError("Source data, checkpoint, or implementation changed")
    if (args.run / "sessions.jsonl.gz").exists():
        raise ValueError("Measurements already exist")
    run(args.run, config)


if __name__ == "__main__":
    main()
