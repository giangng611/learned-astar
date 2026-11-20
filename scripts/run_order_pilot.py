"""Audit the ordering ceiling, then run a bounded matched training pilot."""

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
from order_limit import expansion_floor

from learned_astar.data import (
    DatasetConfig,
    generate_dataset,
    load_record,
    read_manifest,
    verify_dataset,
)
from learned_astar.environments import Grid, validate_path
from learned_astar.models import load_model, predict_distances
from learned_astar.models.transforms import manhattan_field
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.runtime import measure_session, result_signature
from learned_astar.search import SearchLimits, best_first_search
from learned_astar.search.core import manhattan
from learned_astar.search.prepared import PreparedAStar, PreparedGrid
from learned_astar.training.train import (
    load_training_tensors,
    loss_components,
    validation_components,
)

FAMILIES = ("random_32", "walls_32", "random_64", "walls_64")


def fingerprints(config):
    paths = [str(p) for p in Path("src").rglob("*.py")]
    paths += ["scripts/order_limit.py", "scripts/run_order_pilot.py"]
    paths += [str(Path(p) / "manifest.jsonl") for p in config["sources"]]
    paths += [config["model_16"], str(Path(config["models_8"]) / "seed_7.pt")]
    return {p: digest(p) for p in sorted(paths)}


def policy_records(root, config):
    original = read_manifest(config["policy_32"], "validation")
    selected = []
    for density in (0.1, 0.25):
        selected += [
            r for r in original if r["kind"] == "random" and r["requested_density"] == density
        ][:20]
    selected += [r for r in original if r["kind"] == "walls"][:40]
    if len(selected) != 80:
        raise ValueError("Missing original policy-validation maps")
    return [(Path(config["policy_32"]), r) for r in selected] + [
        (root / "policy_64", r) for r in read_manifest(root / "policy_64", "validation")
    ]


def query_map(directory, record, count):
    data = load_record(directory, record)
    grid = Grid(
        *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
    )
    goal = tuple(map(int, data["goals"][0]))
    starts = random.Random(record["map_seed"] + 68101).sample(
        [cell for cell in grid.free_cells() if cell != goal], count
    )
    return grid, goal, starts, data["distances"][0]


def shifted_detour(field, goal):
    value = field - manhattan_field(field.shape, goal)
    return value - value.min()


def headroom(root, config):
    destination = root / "headroom.jsonl.gz"
    if destination.exists():
        raise ValueError("Headroom evidence already exists")
    torch.set_num_threads(1)
    models = {
        "learned_8": load_model(Path(config["models_8"]) / "seed_7.pt")[0],
        "learned_16": load_model(config["model_16"])[0],
    }
    count = 0
    with gzip.open(destination, "wt") as stream:
        for index, (directory, record) in enumerate(policy_records(root, config), 1):
            grid, goal, starts, dg = query_map(directory, record, config["session_starts"])
            starts = starts[: config["headroom_starts"]]
            prepared = PreparedGrid.build(grid)
            fields = {name: predict_distances(model, grid, goal) for name, model in models.items()}
            fields.update(
                {
                    "learned_8_detour": shifted_detour(fields["learned_8"], goal),
                    "exact_distance": np.maximum(dg, 0),
                    "exact_detour": np.maximum(dg - manhattan_field(dg.shape, goal), 0),
                }
            )
            planners = {
                name: PreparedAStar(prepared, goal, ties=field.ravel())
                for name, field in fields.items()
            }
            planners["manhattan"] = PreparedAStar(prepared, goal)
            planners["exact_primary"] = PreparedAStar(
                prepared, goal, primary=np.maximum(dg, 0).ravel()
            )
            for q, start in enumerate(starts):
                floor, mandatory, field = expansion_floor(grid, start, goal, dg)
                results = {name: planner.search(start) for name, planner in planners.items()}
                results["fifo"] = best_first_search(
                    grid, start, goal, manhattan, tie_breaker="fifo"
                )
                results["attained_floor"] = PreparedAStar(
                    prepared, goal, ties=field.ravel()
                ).search(start)
                if results["attained_floor"].stats.expanded != floor:
                    raise ValueError("Constructed ordering missed the expansion floor")
                for name, result in results.items():
                    optimal = int(dg[start])
                    if optimal >= 0:
                        validate_path(grid, result.path, start, goal, result.cost)
                        if result.cost != optimal or result.status != "found":
                            raise ValueError("Wrong diagnostic path")
                    elif result.status != "no_path":
                        raise ValueError("Wrong diagnostic reachability")
                    if name != "exact_primary" and result.stats.expanded < floor:
                        raise ValueError("A tie rule beat the proved expansion floor")
                stream.write(
                    json.dumps(
                        {
                            "layout_id": record["layout_id"],
                            "family": f"{record['kind']}_{grid.width}",
                            "query": q,
                            "start": start,
                            "goal": goal,
                            "optimal_cost": int(dg[start]),
                            "floor": floor,
                            "mandatory": mandatory,
                            "expanded": {
                                name: result.stats.expanded for name, result in results.items()
                            },
                        }
                    )
                    + "\n"
                )
                count += 1
            if index % 20 == 0:
                print(f"headroom maps={index}/120", flush=True)
    save_json(
        root / "headroom_verified.json",
        {
            "queries": count,
            "all_floors_attained": True,
            "all_correct": True,
            "sha256": digest(destination),
        },
    )


def training_gate(root, config):
    verified = json.loads((root / "headroom_verified.json").read_text())
    if verified["sha256"] != digest(root / "headroom.jsonl.gz"):
        raise ValueError("Headroom evidence changed")
    values = {family: {} for family in FAMILIES}
    with gzip.open(root / "headroom.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if r["optimal_cost"] > 0:
                values[r["family"]].setdefault(r["layout_id"], []).append(
                    r["floor"] / r["expanded"]["manhattan"]
                )
    ratios = {f: float(np.mean([np.mean(v) for v in maps.values()])) for f, maps in values.items()}
    return {
        "ratios": ratios,
        "run_pilot": any(v <= 1 - config["gate"]["minimum_headroom"] for v in ratios.values()),
    }


def train_branch(root, config, branch, seed):
    output = root / branch
    output.mkdir(exist_ok=False)
    settings = config["training"]
    configure_torch(seed, settings["cpu_threads"])
    source = Path(config["models_8"]) / f"seed_{seed}.pt"
    model, meta = load_model(source)
    pools = {"original": load_training_tensors(config["development"], "train")}
    if branch == "candidate":
        pools.update(
            {
                "hard_32": load_training_tensors(root / "hard_32", "train"),
                "mixed_64": load_training_tensors(root / "mixed_64", "train"),
            }
        )
    validation = [
        load_training_tensors(config["development"], "validation"),
        load_training_tensors(root / "mixed_64", "validation"),
    ]
    train_groups = {g for _, groups in pools.values() for g in groups}
    validation_groups = {g for _, groups in validation for g in groups}
    policy_groups = {r["group_id"] for _, r in policy_records(root, config)}
    if train_groups & validation_groups or (train_groups | validation_groups) & policy_groups:
        raise ValueError("Training, checkpoint validation, or policy validation overlap")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    schedule = (
        ["original"] * 4 if branch == "control" else ["original", "original", "hard_32", "mixed_64"]
    )
    history, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        losses = []
        for step in range(settings["steps_per_epoch"]):
            tensors = pools[schedule[step % len(schedule)]][0]
            indices = torch.randint(len(tensors[0]), (settings["batch_size"],), generator=generator)
            x, y, mask = [t[indices] for t in tensors]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_components(model(x), y, mask, settings["route_loss_weight"])[0]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        vals = [
            float(
                validation_components(
                    model, tensors, settings["batch_size"], settings["route_loss_weight"]
                )[0]
            )
            for tensors, _ in validation
        ]
        value = float(np.mean(vals))
        if not np.isfinite(value):
            raise ValueError("Nonfinite checkpoint-validation loss")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_32": vals[0],
                "validation_64": vals[1],
                "validation_loss": value,
            }
        )
        if value < best:
            best = value
            torch.save(
                {
                    "architecture": meta["architecture"],
                    "model_config": model.model_config,
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": best,
                    "training_config": settings,
                    "initial_checkpoint_sha256": digest(source),
                    "branch": branch,
                    "development_layout_groups": sorted(train_groups | validation_groups),
                },
                output / f"seed_{seed}.pt",
            )
        save_json(output / f"history_{seed}.json", history)
        print(
            f"{branch} epoch={epoch} validation={value:.6f} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
    save_json(
        output / "summary.json",
        {
            "branch": branch,
            "seed": seed,
            "epochs": settings["epochs"],
            "updates": settings["epochs"] * settings["steps_per_epoch"],
            "best_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"],
            "training_seconds": time.perf_counter() - started,
            "initial_checkpoint_sha256": digest(source),
            "checkpoint_sha256": digest(output / f"seed_{seed}.pt"),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )


def measure_learned(model, rule, grid, goal, starts, limits):
    started = time.perf_counter()
    graph_started = time.perf_counter()
    prepared = PreparedGrid.build(grid)
    graph_seconds = time.perf_counter() - graph_started
    tick = time.perf_counter()
    field = predict_distances(model, grid, goal)
    if rule == "detour":
        field = shifted_detour(field, goal)
    prediction_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    planner = PreparedAStar(prepared, goal, ties=field.ravel().tolist())
    graph_seconds += time.perf_counter() - tick
    setup_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    results = []
    for start in starts:
        result = planner.search(start, limits=limits)
        results.append(
            {
                "status": result.status,
                "cost": result.cost,
                "path": result.path,
                "stats": result.stats,
                "path_steps": None,
            }
        )
    search_seconds = time.perf_counter() - tick
    return {
        "total_seconds": time.perf_counter() - started,
        "setup_seconds": setup_seconds,
        "search_seconds": search_seconds,
        "graph_seconds": graph_seconds,
        "prediction_seconds": prediction_seconds,
        "reference_seconds": 0,
        "preprocess_expanded": 0,
    }, results


def evaluate(root, config):
    if (root / "sessions.jsonl.gz").exists():
        raise ValueError("Evaluation already exists")
    torch.set_num_threads(1)
    paths = {
        "frozen": Path(config["models_8"]) / "seed_7.pt",
        "control": root / "control/seed_7.pt",
        "candidate": root / "candidate/seed_7.pt",
    }
    models = {name: load_model(path)[0] for name, path in paths.items()}
    methods = ["prepared_manhattan", "prepared_exact", "bfs_policy"] + [
        f"{name}_{rule}" for name in models for rule in ("distance", "detour")
    ]
    limits = SearchLimits(1000000, 5.0)

    def measure(name, grid, goal, starts):
        if name in ("prepared_manhattan", "prepared_exact", "bfs_policy"):
            return measure_session(name, grid, goal, starts, {}, limits)
        model, rule = name.split("_")
        return measure_learned(models[model], rule, grid, goal, starts, limits)

    for size in (32, 64):
        for name in methods:
            for _ in range(3):
                measure(name, Grid(size, size), (size - 1, size - 1), [(0, 0)])
    rng = random.Random(config["order_seed"])
    count = query_count = timed_queries = 0
    started = time.perf_counter()
    with (
        gzip.open(root / "sessions.jsonl.gz", "wt") as sessions,
        gzip.open(root / "queries.jsonl.gz", "wt") as queries,
    ):
        for index, (directory, record) in enumerate(policy_records(root, config), 1):
            grid, goal, starts, labels = query_map(directory, record, config["session_starts"])
            signatures = {}
            for size in config["batch_sizes"]:
                for rep in range(config["repetitions"]):
                    order = methods.copy()
                    rng.shuffle(order)
                    for name in order:
                        timing, results = measure(name, grid, goal, starts[:size])
                        for start, result in zip(starts[:size], results, strict=True):
                            optimal = int(labels[start])
                            if optimal >= 0:
                                validate_path(grid, result["path"], start, goal, result["cost"])
                                if result["cost"] != optimal or result["status"] != "found":
                                    raise ValueError("Wrong or incomplete path")
                            elif (
                                result["status"] != "no_path"
                                or result["path"]
                                or result["cost"] is not None
                            ):
                                raise ValueError("Wrong unreachable result")
                        signature = hashlib.sha256(
                            json.dumps(result_signature(results)).encode()
                        ).hexdigest()
                        if (size, name) in signatures and signatures[size, name] != signature:
                            raise ValueError("Repeated paths or counts changed")
                        signatures[size, name] = signature
                        base = {
                            "layout_id": record["layout_id"],
                            "family": f"{record['kind']}_{grid.width}",
                            "batch_size": size,
                            "method": name,
                        }
                        sessions.write(
                            json.dumps(
                                {**base, "repetition": rep, "result_sha256": signature, **timing}
                            )
                            + "\n"
                        )
                        count += 1
                        timed_queries += size
                        if rep == 0:
                            for q, (start, result) in enumerate(
                                zip(starts[:size], results, strict=True)
                            ):
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
                                            "stats": asdict(result["stats"])
                                            if result["stats"]
                                            else None,
                                            "path_steps": result["path_steps"],
                                        }
                                    )
                                    + "\n"
                                )
                                query_count += 1
            if index % 10 == 0:
                print(
                    f"evaluated maps={index}/120 seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    save_json(
        root / "measurement.json",
        {
            "maps": 120,
            "sessions": count,
            "query_records": query_count,
            "timed_queries": timed_queries,
            "seconds": time.perf_counter() - started,
            "all_correct": True,
            "repeated_paths_and_counts_match": True,
            "methods": methods,
            "checkpoints": {
                name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()
            },
            "sessions_sha256": digest(root / "sessions.jsonl.gz"),
            "queries_sha256": digest(root / "queries.jsonl.gz"),
            "torch": str(torch.__version__),
            "cpu_threads": 1,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "headroom", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=Path("configs/order_pilot.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v7"))
    args = parser.parse_args()
    root, config = args.run, tomllib.loads(args.config.read_text())
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.config, root / "protocol.toml")
        save_json(root / "fingerprints.json", fingerprints(config))
        excluded = {r["group_id"] for p in config["sources"] for r in read_manifest(p)}
        verified = {"earlier_groups": len(excluded)}
        for name in ("hard_32", "mixed_64", "policy_64"):
            generate_dataset(root / name, DatasetConfig(**config[name]), exclude_groups=excluded)
            verified[name] = verify_dataset(root / name)
            verified[name]["manifest_sha256"] = digest(root / name / "manifest.jsonl")
            groups = {r["group_id"] for r in read_manifest(root / name)}
            if groups & excluded:
                raise ValueError("Dataset leakage")
            excluded |= groups
        save_json(root / "datasets_verified.json", verified)
        selected = [{"directory": str(p), **r} for p, r in policy_records(root, config)]
        save_json(root / "policy_manifest.json", selected)
        return
    if args.config.read_bytes() != (root / "protocol.toml").read_bytes() or fingerprints(
        config
    ) != json.loads((root / "fingerprints.json").read_text()):
        raise ValueError("Frozen protocol, source, or checkpoint changed")
    verified = json.loads((root / "datasets_verified.json").read_text())
    for name in ("hard_32", "mixed_64", "policy_64"):
        if verified[name]["manifest_sha256"] != digest(root / name / "manifest.jsonl"):
            raise ValueError("Generated dataset manifest changed")
    if args.stage == "headroom":
        headroom(root, config)
        save_json(root / "training_gate.json", training_gate(root, config))
    elif args.stage == "train":
        gate = training_gate(root, config)
        if not gate["run_pilot"]:
            raise ValueError("Insufficient headroom: do not spend resources on this pilot")
        for branch in ("control", "candidate"):
            train_branch(root, config, branch, config["pilot_seed"])
    else:
        evaluate(root, config)


if __name__ == "__main__":
    main()
