import gzip
import hashlib
import json
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from learned_astar.data import load_record, read_manifest, reverse_bfs
from learned_astar.environments import Grid
from learned_astar.evaluation.run import check_result
from learned_astar.evaluation.summarize import (
    bootstrap_mean,
    collapse_repetitions,
    describe_method,
    distribution,
)
from learned_astar.models import load_model, predict_distances
from learned_astar.reproducibility import configure_torch
from learned_astar.search import SearchLimits, SearchResult, SearchStats, SearchStatus
from learned_astar.search.core import astar, best_first_search, weighted_astar


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def specification(root, *, include_replicates):
    root = Path(root)
    models = {
        f"original_seed_{seed}": str(Path("artifacts/checkpoints") / f"seed_{seed}.pt")
        for seed in (7, 17, 27)
    }
    for variant in ("direct", "residual", "no_obstacles"):
        models[f"{variant}_seed_7"] = str(root / variant / "seed_7.pt")
    selection = None
    if include_replicates:
        selection = json.loads((root / "selection.json").read_text())
        if selection["protocol_sha256"] != digest(root / "protocol.toml"):
            raise ValueError("Selection belongs to a different protocol")
        validation_summary = root / "validation" / "summary.json"
        if digest(validation_summary) != selection["validation_summary_sha256"]:
            raise ValueError("Validation evidence changed after selection")
        frozen = json.loads(validation_summary.read_text())["settings"]["checkpoints"]
        for name, path in models.items():
            if digest(path) != frozen[name]["sha256"]:
                raise ValueError("Candidate weights changed after selection")
        for seed in (17, 27):
            models[f"{selection['variant']}_seed_{seed}"] = str(
                root / "replicates" / f"seed_{seed}.pt"
            )
    for path in models.values():
        if not Path(path).is_file():
            raise ValueError(f"Missing frozen model: {path}")
    return {"models": models, "selection": selection}


def manhattan_field(shape, goal):
    rows, columns = np.indices(shape)
    return np.abs(rows - goal[0]) + np.abs(columns - goal[1])


def parity_distances(prediction, goal):
    manhattan = manhattan_field(prediction.shape, goal)
    return manhattan + 2 * np.rint(np.maximum(prediction - manhattan, 0) / 2)


def error_metrics(prediction, labels, manhattan):
    mask = labels >= 0
    detour = mask & (labels > manhattan)
    errors = prediction[mask] - labels[mask]
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "overestimate_fraction": float((errors > 1e-6).mean()),
        "detour_mae": float(np.abs(prediction[detour] - labels[detour]).mean())
        if detour.any()
        else None,
        "detour_fraction": float(detour.sum() / mask.sum()),
    }


def map_predictions(models, grid, goal, labels):
    manhattan = manhattan_field(labels.shape, goal)
    errors = {"manhattan": error_metrics(manhattan, labels, manhattan)}
    for name, model in models.items():
        prediction = predict_distances(model, grid, goal)
        errors[name] = error_metrics(prediction, labels, manhattan)
        obstacles = model.model_config["obstacles"]
        try:
            model.model_config["obstacles"] = False
            hidden = predict_distances(model, grid, goal)
        finally:
            model.model_config["obstacles"] = obstacles
        errors[name + "_hidden"] = error_metrics(hidden, labels, manhattan)
        errors[name]["obstacle_response_mae"] = float(
            np.abs(hidden[labels >= 0] - prediction[labels >= 0]).mean()
        )
    return errors


def field_search(grid, start, goal, make_field, limits):
    """Include prediction/preprocessing in the soft time limit and total timer."""
    tick = time.perf_counter()
    field = make_field()
    prediction_seconds = time.perf_counter() - tick
    remaining = limits.time_limit_seconds - prediction_seconds
    if remaining <= 0:
        result = SearchResult(
            SearchStatus.TIME_LIMIT, (), None, prediction_seconds, SearchStats(0, 0, 0, 0, 0, 0, 0)
        )
    else:
        result = best_first_search(
            grid,
            start,
            goal,
            lambda cell, _: float(max(field[cell], 0)),
            limits=replace(limits, time_limit_seconds=remaining),
        )
    search_seconds = 0 if remaining <= 0 else result.elapsed_seconds
    return replace(result, elapsed_seconds=time.perf_counter() - tick), {
        "prediction_seconds": prediction_seconds,
        "search_seconds": search_seconds,
    }


def query_methods(models, selection):
    methods = {"astar_manhattan": (None, "manhattan")}
    for weight in (1.25, 1.5, 2.0):
        methods[f"weighted_astar_{weight}"] = (None, weight)
    methods["oracle_lookup"] = (None, "oracle")
    methods["exact_bfs_fresh"] = (None, "exact")
    for name, model in models.items():
        modes = ("raw",) if name.startswith("original_") else ("raw", "parity")
        if selection and not name.endswith("_7") and not name.startswith("original_"):
            modes = (selection["mode"],)
        for mode in modes:
            methods[f"{name}_{mode}"] = (model, mode)
    return methods


def execute_query(spec, grid, start, goal, labels, limits):
    model, mode = spec
    if mode == "manhattan":
        result = astar(grid, start, goal, limits=limits)
    elif isinstance(mode, float):
        result = weighted_astar(grid, start, goal, mode, limits=limits)
    else:

        def make_field():
            if mode == "oracle":
                return labels
            if mode == "exact":
                return reverse_bfs(grid, goal)
            field = predict_distances(model, grid, goal)
            return parity_distances(field, goal) if mode == "parity" else field

        return field_search(grid, start, goal, make_field, limits)
    return result, {"prediction_seconds": 0.0, "search_seconds": result.elapsed_seconds}


def query_starts(record, data, grid, goal, validation):
    if not validation:
        return [tuple(map(int, start)) for start in data["starts"]]
    # Validation has training-style goal arrays, so create fixed query starts separately.
    free = [cell for cell in grid.free_cells() if cell != goal]
    rng = random.Random(record["map_seed"] ^ 0x5A17)
    return rng.sample(free, min(4, len(free)))


def evaluate_research(directory, specification, output, config, *, validation, exclude_groups=()):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = config["evaluation"]
    configure_torch(settings["order_seed"], settings["cpu_threads"])
    records = read_manifest(directory, "validation") if validation else read_manifest(directory)
    if not records or (
        not validation and any(r["split"] in ("train", "validation") for r in records)
    ):
        raise ValueError("Invalid evaluation split")
    held_out = {r["group_id"] for r in records}
    if held_out & set(exclude_groups):
        raise ValueError("Fresh test overlaps a previous dataset")
    models, checkpoints = {}, {}
    for name, path in specification["models"].items():
        model, meta = load_model(path)
        if "development_layout_groups" not in meta:
            raise ValueError("Checkpoint has no development provenance")
        if not validation and held_out & set(meta["development_layout_groups"]):
            raise ValueError("Test overlaps model development layouts")
        models[name] = model
        checkpoints[name] = {
            k: meta[k] for k in ("seed", "epoch", "validation_loss", "dataset_manifest_sha256")
        }
        checkpoints[name].update(
            file=path, sha256=digest(path), model_config=model.model_config.copy()
        )
    methods = query_methods(models, specification["selection"])
    limits = SearchLimits(settings["max_expansions"], settings["time_limit_seconds"])
    for height, width in sorted({(r["height"], r["width"]) for r in records}):
        grid = Grid(height, width)
        goal = (height - 1, width - 1)
        labels = reverse_bfs(grid, goal)
        for method in methods.values():
            for _ in range(3):
                execute_query(method, grid, (0, 0), goal, labels, limits)
    raw_rows, map_rows = [], []
    rng = random.Random(settings["order_seed"])
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        if record["status"] != "ready":
            raise ValueError("Research experiment requires ready maps")
        data = load_record(directory, record)
        grid = Grid(
            *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
        )
        goal = tuple(map(int, data["goals"][0]))
        labels = data["distances"][0]
        split = record["kind"] if validation else record["split"]
        map_rows.append(
            {
                "layout_id": record["layout_id"],
                "split": split,
                "errors": map_predictions(models, grid, goal, labels),
            }
        )
        starts = query_starts(record, data, grid, goal, validation)
        for query, start in enumerate(starts):
            optimal = int(labels[start])
            for repetition in range(settings["repetitions"]):
                order = list(methods)
                rng.shuffle(order)
                for name in order:
                    result, parts = execute_query(methods[name], grid, start, goal, labels, limits)
                    check_result(grid, start, goal, result, optimal, name)
                    if name in ("oracle_lookup", "exact_bfs_fresh") and optimal >= 0:
                        if result.cost != optimal or result.stats.expanded != optimal:
                            raise ValueError("Exact heuristic did not follow an optimal path")
                    raw_rows.append(
                        {
                            "layout_id": record["layout_id"],
                            "split": split,
                            "query": query,
                            "method": name,
                            "repetition": repetition,
                            "start": start,
                            "goal": goal,
                            "optimal_cost": optimal,
                            "manhattan_cost": sum(
                                abs(a - b) for a, b in zip(start, goal, strict=True)
                            ),
                            "status": result.status,
                            "cost": result.cost,
                            "elapsed_seconds": result.elapsed_seconds,
                            **asdict(result.stats),
                            **parts,
                        }
                    )
        if index % 25 == 0 or index == len(records):
            print(
                f"maps={index}/{len(records)} seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    rows = collapse_repetitions(raw_rows, settings["repetitions"])
    timing_groups = defaultdict(list)
    for row in raw_rows:
        timing_groups[(row["layout_id"], row["query"], row["method"])].append(row)
    for row in rows:
        group = timing_groups[(row["layout_id"], row["query"], row["method"])]
        for key in ("prediction_seconds", "search_seconds"):
            row[key] = float(np.median([r[key] for r in group]))
    metadata = {
        "completed": True,
        "validation": validation,
        "config": config,
        "dataset_manifest_sha256": digest(directory / "manifest.jsonl"),
        "checkpoints": checkpoints,
        "selection": specification["selection"],
        "maps": len(records),
        "raw_runs": len(raw_rows),
        "seconds": time.perf_counter() - started,
        "environment": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "timing": (
            "Fresh prediction and search per query. oracle_lookup excludes reference construction."
        ),
    }
    for name, values in (("runs", raw_rows), ("queries", rows), ("maps", map_rows)):
        with gzip.open(output / f"{name}.jsonl.gz", "wt") as stream:
            for row in values:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        metadata[f"{name}_sha256"] = digest(output / f"{name}.jsonl.gz")
    summary = summarize_research(rows, map_rows, metadata)
    save_json(output / "summary.json", summary)
    return summary


def summarize_research(rows, map_rows, metadata):
    rng = np.random.default_rng(metadata["config"]["evaluation"]["bootstrap_seed"])
    samples = metadata["config"]["evaluation"]["bootstrap_samples"]
    summary = {"settings": metadata, "splits": {}}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        grouped = {
            name: [r for r in subset if r["method"] == name]
            for name in sorted({r["method"] for r in subset})
        }
        grouped["original_mean"] = [r for r in subset if r["method"].startswith("original_")]
        selection = metadata["selection"]
        if selection:
            names = {
                f"{selection['variant']}_seed_{seed}_{selection['mode']}" for seed in (7, 17, 27)
            }
            grouped["selected_mean"] = [r for r in subset if r["method"] in names]
        baseline = {(r["layout_id"], r["query"]): r for r in grouped["astar_manhattan"]}
        methods = {}
        for name, group in grouped.items():
            description = describe_method(group)
            description["prediction_ms"] = distribution(
                [1000 * r["prediction_seconds"] for r in group]
            )
            description["search_ms"] = distribution([1000 * r["search_seconds"] for r in group])
            pairs = defaultdict(lambda: defaultdict(list))
            for row in group:
                ref = baseline[(row["layout_id"], row["query"])]
                if not row["completed"] or not ref["completed"]:
                    continue
                values = {
                    "runtime_ratio": row["elapsed_seconds"] / ref["elapsed_seconds"],
                    "expansion_ratio": row["expanded"] / ref["expanded"],
                }
                if row["optimal_cost"] >= 0:
                    values["reachable_expansion_ratio"] = values["expansion_ratio"]
                    values["reachable_runtime_ratio"] = values["runtime_ratio"]
                if row["path_ratio"] is not None:
                    values["path_ratio"] = row["path_ratio"]
                for key, value in values.items():
                    pairs[key][row["layout_id"]].append(value)
            description["paired_95ci"] = {
                key: bootstrap_mean([np.mean(v) for v in maps.values()], rng, samples)
                for key, maps in sorted(pairs.items())
            }
            methods[name] = description
        selected_maps = [r for r in map_rows if r["split"] == split]
        errors = {}
        for name in selected_maps[0]["errors"]:
            errors[name] = {
                key: distribution(
                    [
                        r["errors"][name][key]
                        for r in selected_maps
                        if r["errors"][name][key] is not None
                    ]
                )
                for key in selected_maps[0]["errors"][name]
            }
        summary["splits"][split] = {"methods": methods, "prediction_errors": errors}
    return summary


def select_candidate(summary_path, selection):
    summary = json.loads(Path(summary_path).read_text())
    if not summary["settings"]["validation"]:
        raise ValueError("Cannot select a model using test results")
    candidates = []
    for variant in ("direct", "residual"):
        for mode in ("raw", "parity"):
            name = f"{variant}_seed_7_{mode}"
            groups = [summary["splits"][kind]["methods"][name] for kind in ("random", "walls")]
            eligible = all(
                g["reachable_success_rate"] == 1
                and g["statuses"].get("found", 0) + g["statuses"].get("no_path", 0) == g["queries"]
                and g["path_ratio"]["mean"] <= selection["maximum_mean_path_ratio"]
                and g["optimal_fraction_of_found"] >= selection["minimum_optimal_fraction"]
                for g in groups
            )
            score = float(
                np.mean([g["paired_95ci"]["reachable_expansion_ratio"]["mean"] for g in groups])
            )
            candidates.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "eligible": eligible,
                    "expansion_score": score,
                    "worst_mean_path_ratio": max(g["path_ratio"]["mean"] for g in groups),
                }
            )
    eligible = [c for c in candidates if c["eligible"]]
    chosen = (
        min(eligible, key=lambda c: (c["expansion_score"], c["variant"], c["mode"]))
        if eligible
        else min(
            candidates,
            key=lambda c: (
                c["worst_mean_path_ratio"],
                c["expansion_score"],
                c["variant"],
                c["mode"],
            ),
        )
    )
    return {
        **chosen,
        "quality_gate_passed": bool(eligible),
        "candidates": candidates,
        "validation_summary_sha256": digest(summary_path),
    }
