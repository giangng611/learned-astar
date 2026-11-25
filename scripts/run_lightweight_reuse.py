"""Compare frozen model widths with shared predictions and equally prepared baselines."""

import argparse
import gzip
import hashlib
import json
import platform
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
from learned_astar.research.runtime import measure_session, result_signature
from learned_astar.search import SearchLimits

CONTROLS = ("manhattan", "prepared_manhattan", "prepared_exact", "bfs_policy")


def model_specs(config):
    """Composite IDs prevent the same training seed overwriting a different width."""
    result = []
    for width in config["widths"]:
        for seed in config["seeds"]:
            if not 0 <= seed < 1000:
                raise ValueError("Seed must fit the composite model ID")
            result.append(
                {
                    "name": f"width_{width}_seed_{seed}",
                    "id": width * 1000 + seed,
                    "width": width,
                    "seed": seed,
                    "path": str(Path(config["models"][str(width)]) / f"seed_{seed}.pt"),
                }
            )
    if len({s["id"] for s in result}) != len(result):
        raise ValueError("Duplicate width or seed")
    return result


def method_map(config):
    return {name: name for name in CONTROLS} | {
        spec["name"]: f"prepared_{spec['id']}" for spec in model_specs(config)
    }


def fingerprints(config):
    paths = sorted(str(p) for p in Path("src").rglob("*.py"))
    paths += [__file__, *[str(Path(p) / "manifest.jsonl") for p in config["sources"]]]
    # Use a repository-relative script path in the retained provenance.
    paths = [str(Path(p).relative_to(Path.cwd())) if Path(p).is_absolute() else p for p in paths]
    paths += [spec["path"] for spec in model_specs(config)]
    return {p: digest(p) for p in paths}


def load_models(config):
    models = {}
    for spec in model_specs(config):
        model, metadata = load_model(spec["path"])
        if (
            model.model_config
            != {
                "channels": spec["width"],
                "residual": False,
                "obstacles": True,
            }
            or metadata["seed"] != spec["seed"]
        ):
            raise ValueError("Checkpoint width, seed, or model settings changed")
        if metadata["training_config"]["route_loss_weight"] != 0.02:
            raise ValueError("Checkpoint is not the route-trained control")
        models[spec["id"]] = model
    return models


def write_row(stream, row):
    stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def run(root, config):
    torch.set_num_threads(config["cpu_threads"])
    models = load_models(config)
    methods = method_map(config)
    records = read_manifest(root / "fresh_test")
    limits = SearchLimits(config["max_expansions"], config["time_limit_seconds"])
    for size in (16, 32, 64):
        grid = Grid(size, size)
        for internal in methods.values():
            for _ in range(3):
                measure_session(internal, grid, (size - 1, size - 1), [(0, 0)], models, limits)
    rng = random.Random(config["order_seed"])
    session_count = query_count = timed_queries = 0
    started = time.perf_counter()
    with (
        gzip.open(root / "sessions.jsonl.gz", "wt", compresslevel=1) as sessions,
        gzip.open(root / "queries.jsonl.gz", "wt", compresslevel=1) as queries,
    ):
        for index, record in enumerate(records, 1):
            data = load_record(root / "fresh_test", record)
            grid = Grid(
                *data["blocked"].shape,
                frozenset(map(tuple, np.argwhere(data["blocked"]).tolist())),
            )
            goal = tuple(map(int, data["goals"][0]))
            starts = [tuple(map(int, s)) for s in data["starts"]]
            if len(set(starts)) < max(config["batch_sizes"]) or goal in starts:
                raise ValueError("Insufficient distinct starts or a start equals the goal")
            signatures = {}
            for size in config["batch_sizes"]:
                for repetition in range(config["repetitions"]):
                    order = list(methods)
                    rng.shuffle(order)
                    for name in order:
                        timing, results = measure_session(
                            methods[name], grid, goal, starts[:size], models, limits
                        )
                        if len(results) != size:
                            raise ValueError("Query count changed")
                        for start, result in zip(starts[:size], results, strict=True):
                            optimal = int(data["distances"][0][start])
                            if optimal < 0:
                                if (
                                    result["status"] != "no_path"
                                    or result["path"]
                                    or result["cost"] is not None
                                ):
                                    raise ValueError("Incorrect unreachable result")
                            else:
                                validate_path(grid, result["path"], start, goal, result["cost"])
                                if result["status"] != "found" or result["cost"] != optimal:
                                    raise ValueError("Incorrect or unfinished reachable query")
                        encoded = json.dumps(
                            result_signature(results), separators=(",", ":")
                        ).encode()
                        signature = hashlib.sha256(encoded).hexdigest()
                        key = (size, name)
                        if key in signatures and signatures[key] != signature:
                            raise ValueError("Repeated session changed paths or counts")
                        signatures[key] = signature
                        write_row(
                            sessions,
                            {
                                "layout_id": record["layout_id"],
                                "split": record["split"],
                                "batch_size": size,
                                "repetition": repetition,
                                "method": name,
                                "result_sha256": signature,
                                **timing,
                            },
                        )
                        session_count += 1
                        timed_queries += size
                        if repetition == 0:
                            for q, (start, result) in enumerate(
                                zip(starts[:size], results, strict=True)
                            ):
                                write_row(
                                    queries,
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
                                    },
                                )
                                query_count += 1
                if signatures[size, "manhattan"] != signatures[size, "prepared_manhattan"]:
                    raise ValueError("Prepared Manhattan changed search behavior")
            if index % 10 == 0 or index == len(records):
                print(
                    f"maps={index}/{len(records)} seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    save_json(
        root / "measurement.json",
        {
            "maps": len(records),
            "sessions": session_count,
            "timed_queries": timed_queries,
            "query_records": query_count,
            "seconds": time.perf_counter() - started,
            "sessions_sha256": digest(root / "sessions.jsonl.gz"),
            "queries_sha256": digest(root / "queries.jsonl.gz"),
            "dataset_manifest_sha256": digest(root / "fresh_test/manifest.jsonl"),
            "all_optimal_and_correct_reachability": True,
            "all_repeated_paths_and_counts_match": True,
            "prepared_manhattan_matches_original": True,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cpu_threads": torch.get_num_threads(),
                "device": "cpu",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/lightweight_reuse.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v6"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if args.stage == "prepare":
        load_models(config)
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
        print(json.dumps(verified), flush=True)
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
