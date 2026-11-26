"""Split frozen learned A* query time without changing the search implementation."""

import argparse
import gzip
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from run_frontier_pilot import checked, grid_from

from learned_astar.data import load_record, reverse_bfs
from learned_astar.data.dataset import occupancy
from learned_astar.environments import Grid, validate_path
from learned_astar.models import distance_scale, encode_inputs, load_model
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, SearchResult, SearchStats, SearchStatus, astar
from learned_astar.search.core import best_first_search
from learned_astar.search.learned import learned_astar

PARTS = ("input_seconds", "cnn_seconds", "output_seconds", "search_seconds", "other_seconds")
METHODS = ("manhattan", "learned_reference", "learned_profiled")


@torch.inference_mode()
def timed_prediction(model, grid, goal):
    """Same operations and inference-mode scope as predict_distances, with timers."""
    grid.require_cell(goal, free=True)
    tick = time.perf_counter()
    inputs = torch.from_numpy(encode_inputs(occupancy(grid), goal)).unsqueeze(0)
    input_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    output = model(inputs)
    cnn_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    prediction = output.squeeze(0).numpy() * distance_scale(grid.height, grid.width)
    if prediction.shape != (grid.height, grid.width) or not np.isfinite(prediction).all():
        raise ValueError("Model must produce one finite distance per cell")
    if np.any(prediction < 0):
        raise ValueError("Model distances must be nonnegative")
    prediction[goal] = 0.0
    output_seconds = time.perf_counter() - tick
    return prediction, {
        "input_seconds": input_seconds,
        "cnn_seconds": cnn_seconds,
        "output_seconds": output_seconds,
    }


def profiled_search(grid, start, goal, model, limits=None):
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    limits = limits or SearchLimits()
    begun = time.perf_counter()
    distances, parts = timed_prediction(model, grid, goal)
    if limits.time_limit_seconds is not None:
        remaining = limits.time_limit_seconds - (time.perf_counter() - begun)
        if remaining <= 0:
            result = SearchResult(
                SearchStatus.TIME_LIMIT,
                (),
                None,
                time.perf_counter() - begun,
                SearchStats(0, 0, 0, 0, 0, 0, 0),
            )
            return result, {**parts, "search_seconds": 0.0}
        limits = replace(limits, time_limit_seconds=remaining)

    def score(cell, _):
        return float(distances[cell])

    tick = time.perf_counter()
    result = best_first_search(grid, start, goal, score, limits=limits)
    parts["search_seconds"] = time.perf_counter() - tick
    return replace(result, elapsed_seconds=time.perf_counter() - begun), parts


def measure(method, grid, start, goal, model, limits):
    # External call boundaries are used for all three methods, including returned objects.
    tick = time.perf_counter()
    if method == "learned_profiled":
        result, timing = profiled_search(grid, start, goal, model, limits)
    elif method == "learned_reference":
        result, timing = learned_astar(grid, start, goal, model, limits=limits), {}
    elif method == "manhattan":
        result, timing = astar(grid, start, goal, limits=limits), {}
    else:
        raise ValueError("Unknown profile method")
    total = time.perf_counter() - tick
    if method == "learned_profiled":
        timing["other_seconds"] = total - sum(timing.values())
        if timing["other_seconds"] < 0:
            raise ValueError("Component timing exceeds total")
    return result, {**timing, "total_seconds": total}


def signature(result):
    return result.status, result.cost, result.path, tuple(asdict(result.stats).values())


def summarize(rows, repetitions, bootstrap_seed=179502):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["group_id"], row["query"], row["method"]), []).append(row)
    # Keep the full repetition with median total time, so its components still sum exactly.
    representatives = []
    for values in grouped.values():
        if len(values) != repetitions or {r["repetition"] for r in values} != set(
            range(repetitions)
        ):
            raise ValueError("Incomplete repetition coverage")
        representatives.append(
            sorted(values, key=lambda r: r["timing"]["total_seconds"])[repetitions // 2]
        )
    result = {}
    for kind in sorted({r["kind"] for r in representatives}):
        subset = [r for r in representatives if r["kind"] == kind]
        groups = sorted({r["group_id"] for r in subset})
        maps = {}
        for group in groups:
            maps[group] = {}
            query_sets = []
            for method in METHODS:
                selected = [r for r in subset if r["group_id"] == group and r["method"] == method]
                query_sets.append({r["query"] for r in selected})
                if not selected:
                    raise ValueError("Missing method")
                keys = (
                    (*PARTS, "total_seconds")
                    if method == "learned_profiled"
                    else ("total_seconds",)
                )
                maps[group][method] = {
                    k: float(np.mean([r["timing"][k] for r in selected])) for k in keys
                }
            if not query_sets[0] or any(q != query_sets[0] for q in query_sets):
                raise ValueError("Methods used different queries")
        means = {
            method: {
                k: float(np.mean([m[method][k] for m in maps.values()]))
                for k in next(iter(maps.values()))[method]
            }
            for method in METHODS
        }
        totals = {m: means[m]["total_seconds"] for m in METHODS}
        components = {k: means["learned_profiled"][k] for k in PARTS}
        if not np.isclose(
            sum(components.values()), totals["learned_profiled"], atol=1e-12, rtol=1e-9
        ):
            raise ValueError("Components do not add up")
        ratios = np.array(
            [
                m["learned_profiled"]["total_seconds"] / m["learned_reference"]["total_seconds"]
                for m in maps.values()
            ]
        )
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(0, len(groups), (2000, len(groups)))
        ci = np.quantile(ratios[indices].mean(axis=1), [0.025, 0.975]).tolist()
        result[kind] = {
            "maps": len(groups),
            "queries": sum(r["method"] == "learned_profiled" for r in subset),
            "mean_total_ms": {m: t * 1000 for m, t in totals.items()},
            "profile_components_ms": {
                k.removesuffix("_seconds"): v * 1000 for k, v in components.items()
            },
            "profile_component_percent": {
                k.removesuffix("_seconds"): v / totals["learned_profiled"] * 100
                for k, v in components.items()
            },
            "profile_over_reference_map_ratio": float(ratios.mean()),
            "profile_over_reference_95ci": ci,
            "unprofiled_learned_over_manhattan_ratio_of_means": totals["learned_reference"]
            / totals["manhattan"],
            "search_only_over_manhattan_ratio_of_means": components["search_seconds"]
            / totals["manhattan"],
            "remaining_time_budget_ms": 1000 * (totals["manhattan"] - components["search_seconds"]),
        }
    return result


def run(source, output, repetitions):
    if repetitions < 3 or repetitions % 2 != 1:
        raise ValueError("Use an odd repetition count of at least three")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    checked(source)
    checkpoint = source / "frontier/model.pt"
    training = json.loads((source / "training.json").read_text())
    if digest(checkpoint) != training["frontier"]["sha256"]:
        raise ValueError("Checkpoint does not match its training record")
    model_sha = digest(checkpoint)
    policy = json.loads((source / "policy.json").read_text())
    configure_torch(7, 1)
    model, _ = load_model(checkpoint)
    limits = SearchLimits(1000000, 5.0)
    for method in METHODS:
        for _ in range(5):
            measure(method, Grid(32, 32), (0, 0), (31, 31), model, limits)
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        source / "fingerprints.json",
        source / "policy.json",
        source / "training.json",
        checkpoint,
    ]
    fingerprints = {str(p): digest(p) for p in paths}
    save_json(output / "fingerprints.json", fingerprints)
    save_json(
        output / "protocol.json",
        {
            "source": str(source),
            "repetitions": repetitions,
            "order_seed": 179501,
            "cpu_threads": 1,
            "model_sha256": model_sha,
            "timer": "external_call",
            "summary": "representative_median_total_repetition",
        },
    )
    rows, rng = [], random.Random(179501)
    rng.shuffle(policy)
    begun = time.perf_counter()
    with gzip.open(output / "queries.jsonl.gz", "wt") as stream:
        for index, record in enumerate(policy, 1):
            grid = grid_from(load_record(record["directory"], record)["blocked"])
            goal = tuple(record["goal"])
            truth = reverse_bfs(grid, goal)
            signatures = {}
            for rep in range(repetitions):
                jobs = [(q, m) for q in range(len(record["starts"])) for m in METHODS]
                rng.shuffle(jobs)
                for q, method in jobs:
                    start = tuple(record["starts"][q])
                    optimal = int(truth[start])
                    if optimal != record["optimal_costs"][q]:
                        raise ValueError("Stored reference label changed")
                    result, timing = measure(method, grid, start, goal, model, limits)
                    if result.status == "found":
                        validate_path(grid, result.path, start, goal, result.cost)
                        if (
                            optimal < 0
                            or result.cost < optimal
                            or (method == "manhattan" and result.cost != optimal)
                        ):
                            raise ValueError("Incorrect path cost")
                    elif result.status != "no_path" or optimal >= 0:
                        raise ValueError("Invalid reachability or incomplete profiling query")
                    sig = signature(result)
                    if signatures.setdefault((q, method), sig) != sig:
                        raise ValueError("Repeated search outcomes changed")
                    row = {
                        "group_id": record["group_id"],
                        "kind": record["kind"],
                        "query": q,
                        "method": method,
                        "repetition": rep,
                        "timing": timing,
                        "optimal_cost": optimal,
                        "status": result.status,
                        "cost": result.cost,
                        "path": result.path,
                        "stats": asdict(result.stats),
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
            for q in range(len(record["starts"])):
                if signatures[q, "learned_profiled"] != signatures[q, "learned_reference"]:
                    raise ValueError("Instrumentation changed search behavior")
            print(
                f"Profiled {index}/{len(policy)} maps in {time.perf_counter() - begun:.1f}s",
                flush=True,
            )
    checked(source)
    if any(digest(p) != sha for p, sha in fingerprints.items()):
        raise ValueError("Inputs changed while profiling")
    summary = {
        "stage": "development_runtime_profile",
        "trained": False,
        "maps": len(policy),
        "timed_queries": len(rows),
        "repetitions": repetitions,
        "model_sha256": model_sha,
        "profile_matches_reference_paths_and_counters": True,
        "families": summarize(rows, repetitions),
        "raw_sha256": digest(output / "queries.jsonl.gz"),
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/frontier_diversity"))
    parser.add_argument("--output", type=Path, default=Path("outputs/frontier_profile"))
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    run(args.source, args.output, args.repetitions)


if __name__ == "__main__":
    main()
