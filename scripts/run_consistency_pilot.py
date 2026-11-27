"""Matched continuation with an adjacent-cell consistency penalty."""

import argparse
import copy
import gzip
import json
import random
import shutil
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from frontier_learning import pair_loss
from run_frontier_pilot import checked, diagnostics, grid_from, tensors

from learned_astar.data import load_record, reverse_bfs
from learned_astar.environments import Grid, validate_path
from learned_astar.models import distance_scale, load_model
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchLimits, astar, weighted_astar
from learned_astar.search.learned import learned_astar
from learned_astar.training.train import loss_components

BRANCHES = ("control", "consistency")


def edge_penalty(prediction, inputs, mask):
    """Squared excess over one step, averaged over undirected edges per map."""
    h = prediction.masked_fill(inputs[:, 1].bool(), 0) * distance_scale(*prediction.shape[-2:])
    valid = mask.bool() & ~inputs[:, 0].bool()
    deltas, masks = [], []
    for a, b, ma, mb in (
        (h[:, 1:, :], h[:, :-1, :], valid[:, 1:, :], valid[:, :-1, :]),
        (h[:, :, 1:], h[:, :, :-1], valid[:, :, 1:], valid[:, :, :-1]),
    ):
        deltas.append((a - b).abs().flatten(1))
        masks.append((ma & mb).flatten(1))
    delta, edges = torch.cat(deltas, 1), torch.cat(masks, 1)
    excess = (delta - 1).clamp_min(0)
    counts = edges.sum(1).clamp_min(1)
    loss = ((excess.square() * edges).sum(1) / counts).mean()
    # Each undirected edge represents two directed inequalities; at most one fails.
    fraction = (((excess > 1e-5) & edges).sum(1) / (2 * counts)).mean()
    return loss, fraction


def metrics(model, data, batch_size):
    result = diagnostics(model, data, batch_size)
    losses, fractions = [], []
    with torch.inference_mode():
        for start in range(0, len(data["inputs"]), batch_size):
            batch = {k: v[start : start + batch_size] for k, v in data.items()}
            loss, fraction = edge_penalty(model(batch["inputs"]), batch["inputs"], batch["mask"])
            losses.append(float(loss) * len(batch["inputs"]))
            fractions.append(float(fraction) * len(batch["inputs"]))
    return {
        **result,
        "consistency_loss": sum(losses) / len(data["inputs"]),
        "inconsistent_directed_edge_fraction": sum(fractions) / len(data["inputs"]),
    }


def prepare(root, config_path):
    config = tomllib.loads(config_path.read_text())
    source = Path(config["source_run"])
    checked(source)
    for key in ("epochs", "batch_size", "cpu_threads"):
        if config[key] < 1:
            raise ValueError("Positive training budgets required")
    if config["repetitions"] < 3 or config["repetitions"] % 2 != 1:
        raise ValueError("Use at least three odd repetitions")
    for key in ("learning_rate", "route_weight", "frontier_weight", "consistency_weight"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Positive finite loss weights and learning rate required")
    source_model = source / "frontier/model.pt"
    training = json.loads((source / "training.json").read_text())
    if digest(source_model) != training["frontier"]["sha256"]:
        raise ValueError("Parent checkpoint changed")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, root / "protocol.toml")
    shutil.copyfile(source_model, root / "initial.pt")
    for name in ("train.npz", "validation.npz", "policy.json", "selected_records.json"):
        shutil.copyfile(source / name, root / name)
    files = list(Path("src").rglob("*.py")) + [
        Path(__file__),
        Path("scripts/run_frontier_pilot.py"),
        Path("scripts/frontier_learning.py"),
        config_path,
        source / "fingerprints.json",
        source / "training.json",
        source_model,
    ]
    files += list(root.iterdir())
    fingerprints = json.loads((source / "fingerprints.json").read_text())
    fingerprints.update({str(p): digest(p) for p in files})
    save_json(root / "fingerprints.json", fingerprints)
    count = len(tensors(root, "train")["inputs"])
    save_json(
        root / "preparation.json",
        {
            "training_maps": count,
            "epochs": config["epochs"],
            "updates_per_branch": ((count + config["batch_size"] - 1) // config["batch_size"])
            * config["epochs"],
            "initial_checkpoint_sha256": digest(root / "initial.pt"),
            "policy_maps": len(json.loads((root / "policy.json").read_text())),
        },
    )
    checked(root)
    print((root / "preparation.json").read_text())


def train(root):
    config = checked(root)
    if any((root / branch).exists() for branch in BRANCHES):
        raise FileExistsError("Training outputs already exist")
    configure_torch(config["seed"], config["cpu_threads"])
    training, validation = tensors(root, "train"), tensors(root, "validation")
    initial, _ = load_model(root / "initial.pt")
    reports = {"frozen": metrics(initial, validation, config["batch_size"])}
    for branch in BRANCHES:
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
                loss = loss + config["frontier_weight"] * pair_loss(
                    prediction, batch["inputs"], batch["pairs"], batch["pair_mask"]
                )
                if branch == "consistency":
                    loss = (
                        loss
                        + config["consistency_weight"]
                        * edge_penalty(prediction, batch["inputs"], batch["mask"])[0]
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Training diverged")
                loss.backward()
                optimizer.step()
            result = metrics(model, validation, config["batch_size"])
            history.append({"epoch": epoch, **result})
            print(f"{branch} epoch={epoch}: {json.dumps(result)}", flush=True)
        destination = root / branch
        destination.mkdir()
        metadata = copy.deepcopy(metadata)
        # Replace inherited branch labels and objective value from the parent run.
        metadata.pop("frontier_pilot", None)
        metadata.pop("validation_loss", None)
        torch.save(
            {
                **metadata,
                "state_dict": model.state_dict(),
                "epoch": config["epochs"],
                "training_config": config,
                "checkpoint_rule": "fixed_final_epoch",
                "consistency_pilot": branch,
                "continuation_epochs": config["epochs"],
                "initial_checkpoint_sha256": digest(root / "initial.pt"),
                "training_selection_sha256": digest(root / "selected_records.json"),
            },
            destination / "model.pt",
        )
        save_json(destination / "history.json", history)
        reports[branch] = {
            "seconds": time.perf_counter() - begun,
            **history[-1],
            "sha256": digest(destination / "model.pt"),
        }
    checked(root)
    save_json(root / "training.json", reports)
    print(json.dumps(reports, indent=2))


def summarize(rows):
    families = {}
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind]
        methods = sorted({r["method"] for r in selected})
        groups = sorted({r["group_id"] for r in selected})
        means, family = {}, {}
        for method in methods:
            subset = [r for r in selected if r["method"] == method]
            maps = [[r for r in subset if r["group_id"] == group] for group in groups]
            values = {
                key: [float(np.mean([r[key] for r in records])) for records in maps]
                for key in ("total_ms", "expanded", "reexpanded")
            }
            means[method] = values
            reachable = [r for r in subset if r["optimal_cost"] >= 0]
            ratios = [r["cost"] / r["optimal_cost"] for r in reachable if r["optimal_cost"] > 0]
            family[method] = {"mean_" + k: float(np.mean(v)) for k, v in values.items()}
            family[method].update(
                {
                    "reachable_queries": len(reachable),
                    "optimal_fraction": sum(r["cost"] == r["optimal_cost"] for r in reachable)
                    / len(reachable)
                    if reachable
                    else None,
                    "mean_path_ratio": float(np.mean(ratios)) if ratios else None,
                    "max_path_ratio": max(ratios, default=None),
                }
            )
        paired = {}
        for key in ("total_ms", "expanded", "reexpanded"):
            delta = np.array(means["consistency"][key]) - np.array(means["control"][key])
            indices = np.random.default_rng(189502).integers(0, len(groups), (2000, len(groups)))
            paired[key] = {
                "mean_candidate_minus_control": float(delta.mean()),
                "whole_map_95ci": np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
            }
        families[kind] = {"maps": len(groups), "methods": family, "paired_differences": paired}
    return families


def evaluate(root):
    config = checked(root)
    destination = root / "evaluation.jsonl.gz"
    if destination.exists():
        raise FileExistsError("Evaluation already exists")
    configure_torch(config["seed"], 1)
    training = json.loads((root / "training.json").read_text())
    paths = {"frozen": root / "initial.pt", **{b: root / b / "model.pt" for b in BRANCHES}}
    hashes = {k: digest(p) for k, p in paths.items()}
    if any(hashes[b] != training[b]["sha256"] for b in BRANCHES):
        raise ValueError("Trained checkpoint changed")
    models = {k: load_model(p)[0] for k, p in paths.items()}
    methods = ("manhattan", "weighted_1.5", *models)
    limits = SearchLimits(1000000, 5.0)

    def search(method, grid, start, goal):
        if method in models:
            return learned_astar(grid, start, goal, models[method], limits=limits)
        if method == "weighted_1.5":
            return weighted_astar(grid, start, goal, 1.5, limits=limits)
        return astar(grid, start, goal, limits=limits)

    for method in methods:
        for _ in range(3):
            search(method, Grid(32, 32), (0, 0), (31, 31))
    policy = json.loads((root / "policy.json").read_text())
    rng, representatives = random.Random(189501), []
    rng.shuffle(policy)
    with gzip.open(destination, "wt") as stream:
        for index, record in enumerate(policy, 1):
            grid = grid_from(load_record(record["directory"], record)["blocked"])
            goal, sessions = tuple(record["goal"]), {}
            truth = reverse_bfs(grid, goal)
            for rep in range(config["repetitions"]):
                jobs = [(q, m) for q in range(len(record["starts"])) for m in methods]
                rng.shuffle(jobs)
                for q, method in jobs:
                    start = tuple(record["starts"][q])
                    optimal = int(truth[start])
                    if optimal != record["optimal_costs"][q]:
                        raise ValueError("Saved BFS label changed")
                    tick = time.perf_counter()
                    result = search(method, grid, start, goal)
                    total_ms = (time.perf_counter() - tick) * 1000
                    if result.status != ("found" if optimal >= 0 else "no_path"):
                        raise ValueError("Incomplete or incorrect query; raw partial run retained")
                    if result.status == "found":
                        validate_path(grid, result.path, start, goal, result.cost)
                        if result.cost < optimal or (
                            method == "manhattan" and result.cost != optimal
                        ):
                            raise ValueError("Invalid path cost")
                        if method == "weighted_1.5" and result.cost > 1.5 * optimal:
                            raise ValueError("Weighted bound exceeded")
                    row = {
                        "group_id": record["group_id"],
                        "kind": record["kind"],
                        "query": q,
                        "method": method,
                        "repetition": rep,
                        "optimal_cost": optimal,
                        **asdict(result),
                        "total_ms": total_ms,
                        "expanded": result.stats.expanded,
                        "reexpanded": result.stats.reexpanded,
                    }
                    stream.write(json.dumps(row) + "\n")
                    sessions.setdefault((q, method), []).append(row)
            for values in sessions.values():
                signatures = {
                    json.dumps(
                        {k: r[k] for k in ("status", "path", "cost", "stats")}, sort_keys=True
                    )
                    for r in values
                }
                if len(signatures) != 1:
                    raise ValueError("Repeated outcomes changed")
                representatives.append(
                    sorted(values, key=lambda r: r["total_ms"])[config["repetitions"] // 2]
                )
            print(f"Evaluated {index}/{len(policy)} maps", flush=True)
    checked(root)
    if any(digest(paths[k]) != sha for k, sha in hashes.items()):
        raise ValueError("Checkpoint changed during evaluation")
    report = {
        "stage": "consistency_development_not_final_test",
        "fresh_inference_per_query": True,
        "timer": "external_call",
        "maps": len(policy),
        "timed_queries": len(representatives) * config["repetitions"],
        "families": summarize(representatives),
        "checkpoints": hashes,
        "raw_sha256": digest(destination),
    }
    save_json(root / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--run", type=Path, default=Path("outputs/consistency_pilot"))
    parser.add_argument("--config", type=Path, default=Path("configs/consistency_pilot.toml"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.config)
    else:
        {"train": train, "evaluate": evaluate}[args.stage](args.run)
