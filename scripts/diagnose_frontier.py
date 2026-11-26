"""Inspect saved wall-map expansion regressions without training or timing again."""

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from run_frontier_pilot import checked, grid_from

from learned_astar.data import load_record, reverse_bfs
from learned_astar.environments import validate_path
from learned_astar.models import load_model, predict_distances
from learned_astar.models.transforms import manhattan_field
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json
from learned_astar.search.core import best_first_search


def outcome(row):
    return {k: row[k] for k in ("status", "cost", "path", "stats", "optimal_cost")}


def rank_queries(rows, policy, repetitions):
    expected = {
        (r["group_id"], q, m)
        for r in policy
        for q in range(len(r["starts"]))
        for m in ("manhattan", "learned_reference", "learned_profiled")
    }
    grouped = {}
    for row in rows:
        key = row["group_id"], row["query"], row["method"]
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected:
        raise ValueError("Unexpected or missing query coverage")
    saved = {}
    for key, values in grouped.items():
        if len(values) != repetitions or {v["repetition"] for v in values} != set(
            range(repetitions)
        ):
            raise ValueError("Invalid repetition coverage")
        if any(outcome(v) != outcome(values[0]) for v in values):
            raise ValueError("Repeated outcomes changed")
        saved[key] = values[0]
    ranked = []
    for record in policy:
        for q, optimal in enumerate(record["optimal_costs"]):
            key = record["group_id"], q
            learned = saved[*key, "learned_reference"]
            baseline = saved[*key, "manhattan"]
            if outcome(learned) != outcome(saved[*key, "learned_profiled"]):
                raise ValueError("Instrumented outcome differs")
            for row in (learned, baseline):
                if row["optimal_cost"] != optimal or row["kind"] != record["kind"]:
                    raise ValueError("Policy labels differ")
                if row["status"] != ("found" if optimal >= 0 else "no_path"):
                    raise ValueError("Incomplete or inconsistent query")
            if record["kind"] != "walls":
                continue
            a, b = learned["stats"]["expanded"], baseline["stats"]["expanded"]
            ranked.append(
                {
                    "group_id": key[0],
                    "query": q,
                    "optimal_cost": optimal,
                    "manhattan_expanded": b,
                    "learned_expanded": a,
                    "extra_expansions": a - b,
                    "learned_cost": learned["cost"],
                    "learned_reexpanded": learned["stats"]["reexpanded"],
                }
            )
    return sorted(ranked, key=lambda r: (-r["extra_expansions"], r["group_id"], r["query"])), saved


def field_errors(grid, goal, prediction, truth):
    manhattan = manhattan_field(truth.shape, goal)
    reachable = truth >= 0
    groups = {}
    for name, mask in (
        ("all_reachable", reachable),
        ("detour_required", reachable & (truth > manhattan)),
        ("no_detour_required", reachable & (truth == manhattan)),
    ):
        error = prediction[mask] - truth[mask]
        groups[name] = {
            "cells": int(mask.sum()),
            "learned_mae": float(np.abs(error).mean()) if error.size else None,
            "manhattan_mae": float(np.abs(manhattan[mask] - truth[mask]).mean())
            if error.size
            else None,
            "mean_signed_error": float(error.mean()) if error.size else None,
            "overestimate_fraction": float((error > 1e-5).mean()) if error.size else None,
        }
    violations = [
        max(0.0, float(prediction[cell] - prediction[neighbor] - 1))
        for cell in grid.free_cells()
        if reachable[cell]
        for neighbor in grid.neighbors(cell)
        if reachable[neighbor]
    ]
    return {
        "regions": groups,
        "directed_reachable_edges": len(violations),
        "inconsistent_edge_fraction": float(np.mean(np.array(violations) > 1e-5))
        if violations
        else 0.0,
        "max_consistency_violation": max(violations, default=0.0),
    }


def ascii_path(grid, start, goal, path):
    cells = {tuple(c) for c in path}
    return "\n".join(
        "".join(
            "S"
            if (r, c) == start
            else "G"
            if (r, c) == goal
            else "#"
            if (r, c) in grid.blocked
            else "*"
            if (r, c) in cells
            else "."
            for c in range(grid.width)
        )
        for r in range(grid.height)
    )


def run(profile, output, top):
    if top < 1:
        raise ValueError("top must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    protocol = json.loads((profile / "protocol.json").read_text())
    source = Path(protocol["source"])
    checked(source)
    fingerprints = json.loads((profile / "fingerprints.json").read_text())
    fingerprints.update(
        {
            str(profile / name): digest(profile / name)
            for name in ("summary.json", "protocol.json", "queries.jsonl.gz")
        }
    )
    if any(digest(p) != sha for p, sha in fingerprints.items()):
        raise ValueError("Profile inputs changed")
    summary = json.loads((profile / "summary.json").read_text())
    if digest(profile / "queries.jsonl.gz") != summary["raw_sha256"]:
        raise ValueError("Raw evidence checksum differs")
    checkpoint = source / "frontier/model.pt"
    if digest(checkpoint) != summary["model_sha256"]:
        raise ValueError("Checkpoint differs")
    policy = json.loads((source / "policy.json").read_text())
    with gzip.open(profile / "queries.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    ranked, saved = rank_queries(rows, policy, protocol["repetitions"])
    if len(rows) != summary["timed_queries"]:
        raise ValueError("Row count differs")
    selected, seen = [], set()
    for row in ranked:
        if row["extra_expansions"] > 0 and row["group_id"] not in seen:
            selected.append(row)
            seen.add(row["group_id"])
        if len(selected) == top:
            break
    configure_torch(7, 1)
    model, _ = load_model(checkpoint)
    records = {r["group_id"]: r for r in policy}
    cases = []
    output.mkdir(parents=True)
    for index, row in enumerate(selected, 1):
        record = records[row["group_id"]]
        grid = grid_from(load_record(record["directory"], record)["blocked"])
        start, goal = tuple(record["starts"][row["query"]]), tuple(record["goal"])
        truth = reverse_bfs(grid, goal)
        prediction = predict_distances(model, grid, goal)
        replay = best_first_search(
            grid, start, goal, lambda cell, _, field=prediction: float(field[cell])
        )
        original = saved[row["group_id"], row["query"], "learned_reference"]
        if (
            replay.status != original["status"]
            or replay.cost != original["cost"]
            or [list(c) for c in replay.path] != original["path"]
            or asdict(replay.stats) != original["stats"]
        ):
            raise ValueError("Replay differs from saved search")
        if int(truth[start]) != row["optimal_cost"]:
            raise ValueError("BFS label differs")
        if replay.path:
            validate_path(grid, replay.path, start, goal, replay.cost)
        baseline = saved[row["group_id"], row["query"], "manhattan"]
        case = {
            **row,
            "start": start,
            "goal": goal,
            "field_errors": field_errors(grid, goal, prediction, truth),
            "returned_path_cells": [
                {
                    "cell": cell,
                    "predicted_distance": float(prediction[cell]),
                    "true_distance": int(truth[cell]),
                    "signed_error": float(prediction[cell] - truth[cell]),
                }
                for cell in replay.path
            ],
            "replay_matches_saved": True,
        }
        cases.append(case)
        save_json(output / f"case_{index}.json", case)
        np.savez_compressed(output / f"case_{index}_fields.npz", prediction=prediction, truth=truth)
        (output / f"case_{index}_map.txt").write_text(
            "S=start, G=goal, #=wall, *=returned path, .=free\n\nManhattan\n"
            + ascii_path(grid, start, goal, baseline["path"])
            + "\n\nLearned\n"
            + ascii_path(grid, start, goal, replay.path)
            + "\n"
        )
    checked(source)
    if any(digest(p) != sha for p, sha in fingerprints.items()):
        raise ValueError("Inputs changed during diagnosis")
    report = {
        "stage": "development_wall_diagnosis",
        "trained": False,
        "new_timing": False,
        "wall_queries": len(ranked),
        "fewer_expansions": sum(r["extra_expansions"] < 0 for r in ranked),
        "equal_expansions": sum(r["extra_expansions"] == 0 for r in ranked),
        "more_expansions": sum(r["extra_expansions"] > 0 for r in ranked),
        "selection": "largest positive extra expansions, one query per map",
        "limitations": (
            "Selected failures are not representative averages or causal explanations. "
            "Returned-path errors do not trace frontier decisions. No retraining or final test."
        ),
        "cases": [{k: v for k, v in c.items() if k != "returned_path_cells"} for c in cases],
        "model_sha256": summary["model_sha256"],
        "raw_sha256": summary["raw_sha256"],
    }
    save_json(output / "ranking.json", ranked)
    save_json(output / "summary.json", report)
    save_json(output / "fingerprints.json", {**fingerprints, str(Path(__file__)): digest(__file__)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("outputs/frontier_profile"))
    parser.add_argument("--output", type=Path, default=Path("outputs/frontier_diagnosis"))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    run(args.profile, args.output, args.top)
