"""Run the third experiment with frozen objectives and separate policy-validation maps."""

import argparse
import json
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from learned_astar.data import (
    DatasetConfig,
    generate_dataset,
    load_record,
    read_manifest,
    verify_dataset,
)
from learned_astar.environments import Grid
from learned_astar.research.experiment import digest, query_starts, save_json
from learned_astar.search import astar, best_first_search
from learned_astar.search.core import manhattan
from learned_astar.training.train import TrainingConfig, train_models


def source_groups(config):
    sources = config["sources"]
    return {
        record["group_id"]
        for name in ("original", "development", "previous_test")
        for record in read_manifest(sources[name])
    }


def headroom(directory):
    ratios = defaultdict(lambda: defaultdict(list))
    for record in read_manifest(directory, "validation"):
        data = load_record(directory, record)
        grid = Grid(
            *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
        )
        goal = tuple(map(int, data["goals"][0]))
        labels = data["distances"][0]

        def field(cell, _, values=labels):
            return float(max(values[cell], 0))

        per_map = defaultdict(list)
        for start in query_starts(record, data, grid, goal, True):
            if labels[start] < 0:
                continue
            baseline = astar(grid, start, goal)
            for name, heuristic, tie in (
                ("oracle_primary", field, None),
                ("oracle_tie", manhattan, field),
            ):
                result = best_first_search(grid, start, goal, heuristic, tie_breaker=tie)
                if result.cost != int(labels[start]):
                    raise ValueError("Oracle reference returned a wrong path cost")
                per_map[name].append(result.stats.expanded / baseline.stats.expanded)
        for name, values in per_map.items():
            ratios[record["kind"]][name].append(float(np.mean(values)))
    return {
        kind: {
            name: {
                "mean_paired_expansion_ratio": float(np.mean(values)),
                "eligible_maps": len(values),
            }
            for name, values in methods.items()
        }
        for kind, methods in ratios.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "select", "test"))
    parser.add_argument("--config", type=Path, default=Path("configs/search_learning.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v3"))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    root = args.run
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        (root / "protocol.toml").write_bytes(config_bytes)
        sources = {
            name: digest(Path(config["sources"][name]) / "manifest.jsonl")
            for name in ("original", "development", "previous_test")
        }
        save_json(root / "sources.json", sources)
        generate_dataset(
            root / "policy_validation",
            DatasetConfig(**config["policy_validation"]),
            exclude_groups=source_groups(config),
        )
        save_json(
            root / "policy_validation_verified.json", verify_dataset(root / "policy_validation")
        )
        save_json(root / "oracle_headroom.json", headroom(root / "policy_validation"))
        return
    if (root / "protocol.toml").read_bytes() != config_bytes:
        raise ValueError("Protocol changed after preparation")
    sources = json.loads((root / "sources.json").read_text())
    if any(
        digest(Path(config["sources"][name]) / "manifest.jsonl") != expected
        for name, expected in sources.items()
    ):
        raise ValueError("A source dataset changed")
    if args.stage == "train":
        train_models(
            config["sources"]["development"], root / "route", TrainingConfig(**config["training"])
        )
        return
    from learned_astar.research.search_learning import choose_tie_family, evaluate_stage, make_spec

    if args.stage == "select":
        spec = make_spec(root, config)
        evaluate_stage(
            root / "policy_validation", spec, root / "validation", config, validation=True
        )
        selection = choose_tie_family(root / "validation" / "summary.json")
        selection["protocol_sha256"] = digest(root / "protocol.toml")
        save_json(root / "selection.json", selection)
    else:
        selection = json.loads((root / "selection.json").read_text())
        if selection["protocol_sha256"] != digest(root / "protocol.toml"):
            raise ValueError("Selection belongs to another protocol")
        validation_path = root / "validation" / "summary.json"
        if digest(validation_path) != selection["validation_summary_sha256"]:
            raise ValueError("Selection evidence changed")
        spec = make_spec(root, config, selection=selection)
        frozen = json.loads(validation_path.read_text())["settings"]["checkpoints"]
        for name, path in spec["models"].items():
            if digest(path) != frozen[name]["sha256"]:
                raise ValueError("Model changed after selection")
        excluded = source_groups(config) | {
            r["group_id"] for r in read_manifest(root / "policy_validation")
        }
        generate_dataset(
            root / "fresh_test", DatasetConfig(**config["fresh_test"]), exclude_groups=excluded
        )
        save_json(root / "fresh_test_verified.json", verify_dataset(root / "fresh_test"))
        evaluate_stage(
            root / "fresh_test",
            spec,
            root / "test",
            config,
            validation=False,
            exclude_groups=excluded,
        )


if __name__ == "__main__":
    main()
