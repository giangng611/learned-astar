"""Verify and retain the third experiment and paired objective comparisons."""

import argparse
import gzip
import json
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from archive_research import verify_evidence

from learned_astar.data import read_manifest
from learned_astar.evaluation.summarize import bootstrap_mean


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def paired_effects(directory, config):
    rows = read_gzip(directory / "queries.jsonl.gz")
    map_rows = read_gzip(directory / "maps.jsonl.gz")
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"] + 1)
    samples = config["evaluation"]["bootstrap_samples"]
    result = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        indexed = {(r["layout_id"], r["query"], r["method"]): r for r in subset}
        metrics = defaultdict(lambda: defaultdict(list))
        for row in subset:
            if not row["method"].startswith("route_seed_") or row["optimal_cost"] < 0:
                continue
            reference = indexed[
                (row["layout_id"], row["query"], row["method"].replace("route_", "distance_", 1))
            ]
            if not row["completed"] or not reference["completed"]:
                continue
            mode = row["method"].rsplit("_", 1)[1]
            metrics[f"{mode}_expansion_ratio"][row["layout_id"]].append(
                row["expanded"] / reference["expanded"]
            )
            metrics[f"{mode}_runtime_ratio"][row["layout_id"]].append(
                row["elapsed_seconds"] / reference["elapsed_seconds"]
            )
        for row in map_rows:
            if row["split"] != split:
                continue
            for key in ("optimal_move_fraction", "branch_move_fraction", "mae"):
                differences = []
                for seed in (7, 17, 27):
                    route = row["errors"][f"route_seed_{seed}"][key]
                    distance = row["errors"][f"distance_seed_{seed}"][key]
                    if route is not None and distance is not None:
                        differences.append(route - distance)
                if differences:
                    metrics[f"{key}_difference"][row["layout_id"]] = differences
        result[split] = {
            name: bootstrap_mean([np.mean(v) for v in maps.values()], rng, samples)
            for name, maps in metrics.items()
        }
    return {
        "comparison": "Route training versus distance-only training, matched query and seed. "
        "Ratios below one favor route training; positive move-accuracy differences favor it.",
        "splits": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v3"))
    parser.add_argument("--results", type=Path, default=Path("results/research_v3"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/checkpoints/research_v3"))
    args = parser.parse_args()
    config = tomllib.loads((args.run / "protocol.toml").read_text())
    verified = {name: verify_evidence(args.run / name) for name in ("validation", "test")}
    earlier = {
        r["group_id"]
        for name in ("original", "development", "previous_test")
        for r in read_manifest(config["sources"][name])
    }
    validation = {r["group_id"] for r in read_manifest(args.run / "policy_validation")}
    fresh = {r["group_id"] for r in read_manifest(args.run / "fresh_test")}
    if earlier & validation or earlier & fresh or validation & fresh:
        raise ValueError("Layout groups overlap across experiment boundaries")
    verified["disjoint_layout_groups"] = {
        "earlier": len(earlier),
        "policy_validation": len(validation),
        "fresh_test": len(fresh),
    }
    args.results.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.run / "route", args.models)
    for stage in ("validation", "test"):
        destination = args.results / stage
        destination.mkdir()
        for filename in ("summary.json", "queries.jsonl.gz", "maps.jsonl.gz"):
            shutil.copy2(args.run / stage / filename, destination / filename)
        (destination / "paired_objective_effects.json").write_text(
            json.dumps(paired_effects(args.run / stage, config), indent=2) + "\n"
        )
    for filename in (
        "protocol.toml",
        "sources.json",
        "oracle_headroom.json",
        "selection.json",
        "policy_validation_verified.json",
        "fresh_test_verified.json",
    ):
        shutil.copy2(args.run / filename, args.results / filename)
    for name in ("policy_validation", "fresh_test"):
        shutil.copy2(args.run / name / "metadata.json", args.results / f"{name}_metadata.json")
        with gzip.open(args.results / f"{name}_manifest.jsonl.gz", "wb") as stream:
            stream.write((args.run / name / "manifest.jsonl").read_bytes())
    (args.results / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
