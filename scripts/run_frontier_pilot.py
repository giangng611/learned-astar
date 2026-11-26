"""Matched distance-CNN continuation with an additional frontier ordering loss."""

import argparse
import copy
import gzip
import json
import random
import shutil
import statistics
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from frontier_learning import frontier_pairs, padded_pairs, pair_accuracy, pair_loss

from learned_astar.data import load_record, read_manifest, reverse_bfs
from learned_astar.environments import Grid, validate_path
from learned_astar.models import distance_scale, encode_inputs, load_model
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, astar, bfs, weighted_astar
from learned_astar.search.learned import learned_astar
from learned_astar.training.train import loss_components


def checked(root):
    for path, sha in json.loads((root / "fingerprints.json").read_text()).items():
        if digest(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    return tomllib.loads((root / "protocol.toml").read_text())


def grid_from(blocked):
    return Grid(*blocked.shape, frozenset(map(tuple, np.argwhere(blocked).tolist())))


def select_records(directory, split, count):
    records = read_manifest(directory, split)
    chosen = []
    for kind, density, amount in (
        ("random", 0.1, count // 2),
        ("random", 0.25, count // 2),
        ("walls", None, count),
    ):
        subset = [
            r
            for r in records
            if r["status"] == "ready"
            and r["kind"] == kind
            and (density is None or r["requested_density"] == density)
        ]
        if len(subset) < amount:
            raise ValueError("Insufficient source maps")
        chosen.extend(subset[:amount])
    return chosen


def prepare(root, config_path):
    config = tomllib.loads(config_path.read_text())
    if any(
        config[k] < 1 for k in ("trace_starts", "snapshots", "negatives", "epochs", "batch_size")
    ):
        raise ValueError("Positive budgets required")
    if any(config[k] < 2 or config[k] % 2 for k in ("train_per_kind", "validation_per_kind")):
        raise ValueError("Use even positive counts")
    if config["repetitions"] < 3 or config["repetitions"] % 2 != 1:
        raise ValueError("Use at least three odd repetitions")
    _, initial = load_model(config["checkpoint"])
    selected = {
        s: select_records(config["source"], s, config[f"{s}_per_kind"])
        for s in ("train", "validation")
    }
    groups = {s: {r["group_id"] for r in rs} for s, rs in selected.items()}
    if groups["train"] & groups["validation"]:
        raise ValueError("Training and checkpoint validation overlap")
    policy, policy_groups = [], set()
    prohibited = groups["train"] | groups["validation"] | set(initial["development_layout_groups"])
    for source in config["policy_sources"]:
        for record in json.loads((Path(source) / "manifest.json").read_text()):
            if record["width"] != 32:
                continue
            if record["group_id"] in prohibited or record["group_id"] in policy_groups:
                raise ValueError("Policy maps overlap model development or one another")
            policy_groups.add(record["group_id"])
            data = load_record(source, record)
            grid = grid_from(data["blocked"])
            rng = random.Random(record["map_seed"] + config["query_seed"])
            cells = rng.sample(list(grid.free_cells()), config["evaluation_starts"] + 1)
            goal, starts = cells[0], cells[1:]
            truth = reverse_bfs(grid, goal)
            policy.append(
                {
                    **record,
                    "directory": str(source),
                    "goal": goal,
                    "starts": starts,
                    "optimal_costs": [int(truth[s]) for s in starts],
                }
            )
    if not policy:
        raise ValueError("No separate policy maps")
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.toml").write_bytes(config_path.read_bytes())
    shutil.copyfile(config["checkpoint"], root / "initial.pt")
    save_json(root / "selected_records.json", selected)
    save_json(root / "policy.json", policy)
    counts = {}
    for split, records in selected.items():
        arrays = {k: [] for k in ("inputs", "targets", "mask", "pairs", "pair_mask")}
        coverage = []
        for i, record in enumerate(records):
            data = load_record(config["source"], record)
            grid = grid_from(data["blocked"])
            # One goal per map keeps map weighting and the continuation budget explicit.
            goal = tuple(map(int, data["goals"][0]))
            labels = data["distances"][0]
            rng = random.Random(record["map_seed"] + config["query_seed"])
            starts = rng.sample([c for c in grid.free_cells() if c != goal], config["trace_starts"])
            pairs = []
            for j, start in enumerate(starts):
                found, expanded, cost = frontier_pairs(
                    grid,
                    start,
                    goal,
                    labels,
                    seed=record["map_seed"] + j,
                    snapshots=config["snapshots"],
                    negatives=config["negatives"],
                )
                reference = astar(grid, start, goal)
                if (expanded, cost) != (reference.stats.expanded, reference.cost):
                    raise ValueError("Trace disagrees with ordinary Manhattan A*")
                pairs.extend(found)
            maximum = config["trace_starts"] * config["snapshots"] * config["negatives"]
            padded, valid = padded_pairs(pairs, maximum)
            arrays["inputs"].append(encode_inputs(data["blocked"], goal))
            arrays["targets"].append(
                np.maximum(labels, 0).astype(np.float32) / distance_scale(*labels.shape)
            )
            arrays["mask"].append(labels >= 0)
            arrays["pairs"].append(padded)
            arrays["pair_mask"].append(valid)
            coverage.append(
                {"group_id": record["group_id"], "kind": record["kind"], "pairs": len(pairs)}
            )
            if (i + 1) % 16 == 0:
                print(f"Prepared {split} {i + 1}/{len(records)}", flush=True)
        np.savez_compressed(root / f"{split}.npz", **{k: np.stack(v) for k, v in arrays.items()})
        save_json(root / f"{split}_coverage.json", coverage)
        counts[split] = {
            "maps": len(records),
            "pairs": sum(r["pairs"] for r in coverage),
            "maps_without_pairs": sum(r["pairs"] == 0 for r in coverage),
        }
    paths = list(Path("src").rglob("*.py"))
    paths += [Path("scripts/frontier_learning.py"), Path(__file__), config_path]
    paths += [
        root / name
        for name in (
            "initial.pt",
            "protocol.toml",
            "selected_records.json",
            "policy.json",
            "train.npz",
            "validation.npz",
        )
    ]
    paths += [Path(config["source"]) / r["file"] for rs in selected.values() for r in rs]
    paths += [Path(r["directory"]) / r["file"] for r in policy]
    save_json(root / "fingerprints.json", {str(p): digest(p) for p in sorted(set(paths))})
    save_json(
        root / "preparation.json",
        {
            **counts,
            "policy_maps": len(policy),
            "initial_checkpoint_sha256": digest(root / "initial.pt"),
        },
    )
    print((root / "preparation.json").read_text())


def tensors(root, split):
    with np.load(root / f"{split}.npz", allow_pickle=False) as data:
        return {k: torch.from_numpy(data[k]) for k in data.files}


def diagnostics(model, data, batch_size):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["inputs"]), batch_size):
            predictions.append(model(data["inputs"][start : start + batch_size]))
        prediction = torch.cat(predictions)
        total, distance, route = loss_components(prediction, data["targets"], data["mask"], 0.02)
        return {
            "distance_loss": float(distance),
            "route_loss": float(route),
            "frontier_loss": float(
                pair_loss(prediction, data["inputs"], data["pairs"], data["pair_mask"])
            ),
            "frontier_accuracy": pair_accuracy(
                prediction, data["inputs"], data["pairs"], data["pair_mask"]
            ),
        }


def train(root):
    config = checked(root)
    configure_torch(config["seed"], config["cpu_threads"])
    training, validation = tensors(root, "train"), tensors(root, "validation")
    initial, _ = load_model(root / "initial.pt")
    reports = {"frozen": diagnostics(initial, validation, config["batch_size"])}
    for branch in ("control", "frontier"):
        destination = root / branch
        if destination.exists():
            raise FileExistsError(f"Training output exists: {destination}")
        destination.mkdir()
        model, metadata = load_model(root / "initial.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        generator = torch.Generator().manual_seed(config["seed"])
        history, begun = [], time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            order = torch.randperm(len(training["inputs"]), generator=generator)
            for indices in order.split(config["batch_size"]):
                batch = {k: v[indices] for k, v in training.items()}
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["inputs"])
                loss = loss_components(
                    prediction, batch["targets"], batch["mask"], config["route_weight"]
                )[0]
                if branch == "frontier":
                    loss = loss + config["frontier_weight"] * pair_loss(
                        prediction, batch["inputs"], batch["pairs"], batch["pair_mask"]
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Training diverged")
                loss.backward()
                optimizer.step()
            metrics = diagnostics(model, validation, config["batch_size"])
            history.append({"epoch": epoch, **metrics})
            if epoch == 1 or epoch % 5 == 0:
                print(f"{branch} epoch={epoch}: {json.dumps(metrics)}", flush=True)
        checkpoint = {
            **copy.deepcopy(metadata),
            "state_dict": model.state_dict(),
            "epoch": config["epochs"],
            "training_config": config,
            "validation_loss": metrics["distance_loss"]
            + config["route_weight"] * metrics["route_loss"]
            + (config["frontier_weight"] * metrics["frontier_loss"] if branch == "frontier" else 0),
            "checkpoint_rule": "fixed_final_epoch",
            "training_selection_sha256": digest(root / "selected_records.json"),
            "frontier_pilot": branch,
            "continuation_epochs": config["epochs"],
            "initial_checkpoint_sha256": digest(root / "initial.pt"),
        }
        torch.save(checkpoint, destination / "model.pt")
        save_json(destination / "history.json", history)
        reports[branch] = {
            "seconds": time.perf_counter() - begun,
            **history[-1],
            "sha256": digest(destination / "model.pt"),
        }
    checked(root)
    save_json(root / "training.json", reports)
    print(json.dumps(reports, indent=2))


def evaluate(root):
    config = checked(root)
    configure_torch(config["seed"], 1)
    destination = root / "evaluation.jsonl.gz"
    if destination.exists():
        raise FileExistsError("Evaluation already exists")
    training = json.loads((root / "training.json").read_text())
    paths = {
        "frozen": root / "initial.pt",
        "control": root / "control/model.pt",
        "frontier": root / "frontier/model.pt",
    }
    for branch in ("control", "frontier"):
        if digest(paths[branch]) != training[branch]["sha256"]:
            raise ValueError("Trained checkpoint changed")
    models = {k: load_model(p)[0] for k, p in paths.items()}
    methods = ("manhattan", "bfs", "weighted_1.25", "weighted_1.5", "weighted_2.0", *models)
    limits = SearchLimits(1000000, 5.0)

    def run(method, grid, start, goal):
        if method in models:
            return learned_astar(grid, start, goal, models[method], limits=limits)
        if method.startswith("weighted_"):
            return weighted_astar(grid, start, goal, float(method.split("_")[1]), limits=limits)
        return (astar if method == "manhattan" else bfs)(grid, start, goal, limits=limits)

    for method in methods:
        for _ in range(3):
            run(method, Grid(32, 32), (0, 0), (31, 31))
    policy = json.loads((root / "policy.json").read_text())
    rng = random.Random(config["query_seed"])
    rng.shuffle(policy)
    medians = []
    with gzip.open(destination, "wt") as stream:
        for index, record in enumerate(policy, 1):
            grid = grid_from(load_record(record["directory"], record)["blocked"])
            goal = tuple(record["goal"])
            sessions = {}
            for rep in range(config["repetitions"]):
                jobs = [(q, m) for q in range(len(record["starts"])) for m in methods]
                rng.shuffle(jobs)
                for q, method in jobs:
                    start, truth = tuple(record["starts"][q]), record["optimal_costs"][q]
                    result = run(method, grid, start, goal)
                    if result.status == "found":
                        validate_path(grid, result.path, start, goal, result.cost)
                        if truth < 0 or result.cost < truth:
                            raise ValueError("Path disagrees with BFS label")
                        if method in ("manhattan", "bfs") and result.cost != truth:
                            raise ValueError("Exact baseline returned a nonoptimal path")
                        if (
                            method.startswith("weighted_")
                            and result.cost > float(method.split("_")[1]) * truth
                        ):
                            raise ValueError("Weighted A* exceeded its bound")
                    elif result.status == "no_path" and truth >= 0:
                        raise ValueError("Incorrect no-path result")
                    row = {
                        "group_id": record["group_id"],
                        "kind": record["kind"],
                        "query": q,
                        "method": method,
                        "repetition": rep,
                        "optimal_cost": truth,
                        **asdict(result),
                    }
                    stream.write(json.dumps(row) + "\n")
                    sessions.setdefault((q, method), []).append(row)
            for rows in sessions.values():
                signatures = [
                    json.dumps(
                        {k: r[k] for k in ("status", "path", "cost", "stats")}, sort_keys=True
                    )
                    for r in rows
                ]
                if len(set(signatures)) != 1:
                    raise ValueError("Repeated search outcomes differ")
                row = rows[0]
                medians.append(
                    {
                        **row,
                        "elapsed_seconds": statistics.median(r["elapsed_seconds"] for r in rows),
                    }
                )
            print(f"Evaluated {index}/{len(policy)} maps", flush=True)
    summary = {
        "stage": "policy_validation_not_final_test",
        "fresh_inference_per_query": True,
        "maps": len(policy),
        "queries": sum(len(r["starts"]) for r in policy),
        "families": {},
    }
    for kind in sorted({r["kind"] for r in medians}):
        values = {}
        for method in methods:
            rows = [r for r in medians if r["kind"] == kind and r["method"] == method]
            reachable = [r for r in rows if r["optimal_cost"] > 0]
            found = [r for r in reachable if r["status"] == "found"]
            values[method] = {
                "mean_ms": 1000 * statistics.mean(r["elapsed_seconds"] for r in rows),
                "mean_expanded": statistics.mean(r["stats"]["expanded"] for r in rows),
                "reachable_queries": len(reachable),
                "found": len(found),
                "optimal_fraction": sum(r["cost"] == r["optimal_cost"] for r in found)
                / len(reachable)
                if reachable
                else None,
                "mean_path_ratio": statistics.mean(r["cost"] / r["optimal_cost"] for r in found)
                if found
                else None,
                "max_path_ratio": max((r["cost"] / r["optimal_cost"] for r in found), default=None),
                "statuses": {
                    s: sum(r["status"] == s for r in rows)
                    for s in sorted({r["status"] for r in rows})
                },
            }
        summary["families"][kind] = values
    checked(root)
    summary["checkpoints"] = {k: digest(p) for k, p in paths.items()}
    if any(summary["checkpoints"][k] != training[k]["sha256"] for k in ("control", "frontier")):
        raise ValueError("Checkpoint changed during evaluation")
    summary["raw_sha256"] = digest(destination)
    save_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--run", type=Path, default=Path("outputs/frontier_pilot"))
    parser.add_argument("--config", type=Path, default=Path("configs/frontier_pilot.toml"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.config)
    elif args.stage == "train":
        train(args.run)
    else:
        evaluate(args.run)


if __name__ == "__main__":
    main()
