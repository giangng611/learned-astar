"""Verify and retain compact evidence and all trained follow-up models."""

import argparse
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from learned_astar.data import load_record, read_manifest
from learned_astar.models.transforms import manhattan_field
from learned_astar.research.experiment import digest, query_methods


def verify_evidence(directory):
    summary = json.loads((directory / "summary.json").read_text())
    settings = summary["settings"]
    rows = {}
    for name in ("runs", "queries", "maps"):
        path = directory / f"{name}.jsonl.gz"
        if digest(path) != settings[f"{name}_sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {path}")
        with gzip.open(path, "rt") as stream:
            rows[name] = [json.loads(line) for line in stream]
    raw = defaultdict(list)
    for row in rows["runs"]:
        raw[(row["layout_id"], row["query"], row["method"])].append(row)
    if len(rows["runs"]) != settings["raw_runs"] or len(raw) != len(rows["queries"]):
        raise ValueError("Incomplete query evidence")
    by_query = defaultdict(set)
    for row in rows["queries"]:
        key = row["layout_id"], row["query"], row["method"]
        group = raw[key]
        repetitions = settings["config"]["evaluation"]["repetitions"]
        if len(group) != repetitions or {r["repetition"] for r in group} != set(range(repetitions)):
            raise ValueError("Missing or duplicate repetitions")
        for field in ("elapsed_seconds", "expanded", "prediction_seconds", "search_seconds"):
            if row[field] != float(np.median([r[field] for r in group])):
                raise ValueError("Collapsed evidence differs from raw measurements")
        by_query[key[:2]].add(row["method"])
    expected = set(query_methods(dict.fromkeys(settings["checkpoints"]), settings["selection"]))
    if any(names != expected for names in by_query.values()):
        raise ValueError("Methods are not paired on identical queries")
    for model in settings["checkpoints"].values():
        if digest(model["file"]) != model["sha256"]:
            raise ValueError("Measured checkpoint changed")
    return {
        "maps": len(rows["maps"]),
        "queries": len(by_query),
        "query_method_records": len(rows["queries"]),
        "raw_runs": len(rows["runs"]),
    }


def training_profile(directory):
    groups = defaultdict(list)
    for record in read_manifest(directory, "train"):
        data = load_record(directory, record)
        for goal, labels in zip(data["goals"], data["distances"], strict=True):
            lower = manhattan_field(labels.shape, goal)
            mask = labels >= 0
            delta = labels[mask] - lower[mask]
            groups[record["kind"]].append(
                [
                    float(delta.mean()),
                    float((delta > 0).mean()),
                    int(delta.max()),
                    int(labels[mask].max()),
                ]
            )
    return {
        name: {
            "goal_maps": len(values),
            "mean_detour": float(np.mean(values, axis=0)[0]),
            "detour_fraction": float(np.mean(values, axis=0)[1]),
            "max_detour": int(np.max(values, axis=0)[2]),
            "max_distance": int(np.max(values, axis=0)[3]),
        }
        for name, values in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v2"))
    parser.add_argument("--original", type=Path, default=Path("datasets/main"))
    parser.add_argument("--results", type=Path, default=Path("results/research_v2"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/checkpoints/research_v2"))
    args = parser.parse_args()
    verified = {name: verify_evidence(args.run / name) for name in ("validation", "test")}
    old = {r["group_id"] for r in read_manifest(args.original)}
    development = {r["group_id"] for r in read_manifest(args.run / "development")}
    fresh = {r["group_id"] for r in read_manifest(args.run / "fresh_test")}
    if old & development or old & fresh or development & fresh:
        raise ValueError("Dataset layout groups overlap")
    verified["disjoint_layout_groups"] = {
        "original": len(old),
        "development": len(development),
        "fresh_test": len(fresh),
    }
    args.results.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    for name in ("validation", "test"):
        destination = args.results / name
        destination.mkdir()
        for filename in ("summary.json", "queries.jsonl.gz", "maps.jsonl.gz"):
            shutil.copy2(args.run / name / filename, destination / filename)
    for name in ("direct", "residual", "no_obstacles", "replicates"):
        shutil.copytree(args.run / name, args.models / name)
    for filename in (
        "protocol.toml",
        "selection.json",
        "development_metadata.json",
        "development_verified.json",
        "fresh_test_metadata.json",
        "fresh_test_verified.json",
        "validation_diagnostics.json",
    ):
        path = args.run / filename
        if path.is_file():
            shutil.copy2(path, args.results / filename)
    for name in ("development", "fresh_test"):
        with gzip.open(args.results / f"{name}_manifest.jsonl.gz", "wb") as stream:
            stream.write((args.run / name / "manifest.jsonl").read_bytes())
    (args.results / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    profile = {
        "original": training_profile(args.original),
        "mixed": training_profile(args.run / "development"),
    }
    (args.results / "data_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
