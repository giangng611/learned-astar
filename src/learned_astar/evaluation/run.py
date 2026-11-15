import argparse
import hashlib
import json
import platform
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

from learned_astar.config import require_int
from learned_astar.data import load_record, read_manifest, reverse_bfs
from learned_astar.environments import Grid, validate_path
from learned_astar.models import load_model, predict_distances
from learned_astar.reproducibility import configure_torch
from learned_astar.search import SearchLimits, SearchStatus, astar, bfs, dijkstra, weighted_astar
from learned_astar.search.learned import learned_astar


@dataclass(frozen=True)
class EvaluationConfig:
    repetitions: int = 5
    order_seed: int = 8102
    bootstrap_seed: int = 8103
    bootstrap_samples: int = 2000
    cpu_threads: int = 1
    max_expansions: int = 1000000
    time_limit_seconds: float = 5.0

    def __post_init__(self):
        for key in ("repetitions", "bootstrap_samples", "cpu_threads", "max_expansions"):
            require_int(getattr(self, key), key, minimum=1)
        require_int(self.order_seed, "order_seed")
        require_int(self.bootstrap_seed, "bootstrap_seed")
        SearchLimits(self.max_expansions, self.time_limit_seconds)
        if self.time_limit_seconds == 0:
            raise ValueError("Evaluation time limit must be positive")


def methods_for_models(models):
    methods = {"bfs": bfs, "dijkstra": dijkstra, "astar_manhattan": astar}
    for weight in (1.25, 1.5, 2.0):
        methods[f"weighted_astar_{weight}"] = partial(weighted_astar, weight=weight)
    for name, model in models.items():
        methods[name] = partial(learned_astar, model=model)
    return methods


def check_result(grid, start, goal, result, optimal, method):
    if result.status == SearchStatus.FOUND:
        validate_path(grid, result.path, start, goal, result.cost)
        if optimal < 0 or result.cost < optimal:
            raise ValueError(f"Invalid reference comparison for {method}")
        if method in ("bfs", "dijkstra", "astar_manhattan") and result.cost != optimal:
            raise ValueError(f"Nonoptimal baseline result for {method}")
        if method.startswith("weighted_astar_"):
            if result.cost > float(method.removeprefix("weighted_astar_")) * optimal:
                raise ValueError(f"Weighted bound failed for {method}")
    elif result.status == SearchStatus.NO_PATH and optimal >= 0:
        raise ValueError(f"Search incorrectly reported no route: {method}")


def read_checkpoints(paths, records, manifest_hash):
    models, metadata = {}, {}
    held_out = {record["group_id"] for record in records}
    for path in paths:
        started = time.perf_counter()
        model, meta = load_model(path)
        loading_seconds = time.perf_counter() - started
        if meta.get("dataset_manifest_sha256") != manifest_hash:
            raise ValueError("Checkpoint and evaluation dataset manifests differ")
        if "development_layout_groups" not in meta:
            raise ValueError("Checkpoint must identify its training and validation layouts")
        if held_out & set(meta["development_layout_groups"]):
            raise ValueError("Evaluation layouts overlap model development layouts")
        name = f"learned_seed_{meta['seed']}"
        if name in models:
            raise ValueError("Checkpoint seeds must be distinct")
        models[name] = model
        metadata[name] = {
            "file": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "seed": meta["seed"],
            "epoch": meta["epoch"],
            "validation_loss": meta["validation_loss"],
            "loading_seconds": loading_seconds,
        }
    if not models:
        raise ValueError("At least one learned checkpoint is required")
    return models, metadata


def evaluate(directory, checkpoints, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configure_torch(config.order_seed, config.cpu_threads)
    records = [r for r in read_manifest(directory) if r["split"] not in ("train", "validation")]
    if not records:
        raise ValueError("Dataset has no held-out maps")
    manifest_hash = hashlib.sha256((Path(directory) / "manifest.jsonl").read_bytes()).hexdigest()
    models, checkpoint_metadata = read_checkpoints(checkpoints, records, manifest_hash)
    methods = methods_for_models(models)
    limits = SearchLimits(config.max_expansions, config.time_limit_seconds)
    rng = random.Random(config.order_seed)
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "dataset_manifest_sha256": manifest_hash,
        "checkpoints": checkpoint_metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
        },
        "timing": "Feature preparation, fresh prediction, search, and reconstruction; no file I/O.",
        "maps": len(records),
        "excluded_maps": [r for r in records if r["status"] != "ready"],
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Warm up every method and shape on independent open maps.
    for size in sorted({r["height"] for r in records}):
        grid = Grid(size, size)
        for method in methods.values():
            for _ in range(3):
                method(grid, (0, 0), (size - 1, size - 1), limits=limits)
    started = time.perf_counter()
    queries = 0
    with (output / "runs.jsonl").open("w") as runs, (output / "maps.jsonl").open("w") as maps:
        for index, record in enumerate(records, 1):
            if record["status"] != "ready":
                continue
            data = load_record(directory, record)
            grid = Grid(
                *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
            )
            goal = tuple(map(int, data["goals"][0]))
            labels = data["distances"][0]
            oracle_times = []
            for _ in range(config.repetitions):
                tick = time.perf_counter()
                reference = reverse_bfs(grid, goal)
                oracle_times.append(time.perf_counter() - tick)
                np.testing.assert_array_equal(reference, labels)
            prediction_errors = {}
            for name, model in models.items():
                prediction = predict_distances(model, grid, goal)
                error = prediction[labels >= 0] - labels[labels >= 0]
                positive = error[error > 1e-6]
                prediction_errors[name] = {
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                    "overestimate_fraction": float((error > 1e-6).mean()),
                    "mean_overestimate": float(positive.mean()) if len(positive) else 0.0,
                    "max_overestimate": float(max(0.0, error.max())),
                }
            maps.write(
                json.dumps(
                    {
                        "layout_id": record["layout_id"],
                        "split": record["split"],
                        "oracle_seconds": oracle_times,
                        "prediction_errors": prediction_errors,
                        "obstacle_fraction": record["actual_density"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            for query_index, start_array in enumerate(data["starts"]):
                start = tuple(map(int, start_array))
                optimal = int(labels[start])
                for repetition in range(config.repetitions):
                    order = list(methods)
                    rng.shuffle(order)
                    for name in order:
                        result = methods[name](grid, start, goal, limits=limits)
                        check_result(grid, start, goal, result, optimal, name)
                        row = {
                            "layout_id": record["layout_id"],
                            "split": record["split"],
                            "query": query_index,
                            "start": start,
                            "goal": goal,
                            "optimal_cost": optimal,
                            "method": name,
                            "repetition": repetition,
                            "status": result.status,
                            "cost": result.cost,
                            "elapsed_seconds": result.elapsed_seconds,
                            **asdict(result.stats),
                        }
                        runs.write(json.dumps(row, allow_nan=False) + "\n")
                queries += 1
            if index % 25 == 0 or index == len(records):
                runs.flush()
                maps.flush()
                print(
                    f"maps={index}/{len(records)} queries={queries} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    metadata["queries"] = queries
    metadata["evaluation_seconds"] = time.perf_counter() - started
    metadata["completed"] = True
    metadata["runs_sha256"] = hashlib.sha256((output / "runs.jsonl").read_bytes()).hexdigest()
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on held-out maps")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = (
        EvaluationConfig(**tomllib.loads(args.config.read_text()))
        if args.config
        else EvaluationConfig()
    )
    evaluate(args.data, args.checkpoints, args.output, config)


if __name__ == "__main__":
    main()
