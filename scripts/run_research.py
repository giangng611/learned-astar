"""Run explicit development, selection, and held-out stages without overwriting evidence."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from learned_astar.data import DatasetConfig, generate_dataset, read_manifest, verify_dataset
from learned_astar.training.train import TrainingConfig, train_models

VARIANTS = {
    "direct": {"residual": False, "obstacles": True},
    "residual": {"residual": True, "obstacles": True},
    "no_obstacles": {"residual": True, "obstacles": False},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "select", "replicate", "test"])
    parser.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    parser.add_argument("--original", type=Path, default=Path("datasets/main"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v2"))
    parser.add_argument("--resume", action="store_true", help="Skip completed development runs")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    root = args.run
    data = root / "development"
    digest = hashlib.sha256(config_bytes).hexdigest()
    if args.stage == "develop":
        if args.resume:
            if (root / "protocol.toml").read_bytes() != config_bytes:
                raise ValueError("Cannot resume a different development protocol")
            verify_dataset(data)
        else:
            root.mkdir(parents=True, exist_ok=False)
            (root / "protocol.toml").write_bytes(config_bytes)
            old_groups = {r["group_id"] for r in read_manifest(args.original)}
            metadata = generate_dataset(
                data, DatasetConfig(**config["development"]), exclude_groups=old_groups
            )
            write_json(root / "development_verified.json", verify_dataset(data))
            write_json(root / "development_metadata.json", metadata)
        for name, options in VARIANTS.items():
            if args.resume and (root / name / "summary.json").is_file():
                completed = json.loads((root / name / "summary.json").read_text())
                if {s["seed"] for s in completed} == set(config["training"]["seeds"]):
                    print(f"Already completed: {name}", flush=True)
                    continue
            train_models(data, root / name, TrainingConfig(**config["training"], **options))
        return
    if hashlib.sha256((root / "protocol.toml").read_bytes()).hexdigest() != digest:
        raise ValueError("Protocol changed after development started")
    from learned_astar.research.experiment import (
        evaluate_research,
        select_candidate,
        specification,
    )

    selection_path = root / "selection.json"
    if args.stage == "select":
        spec = specification(root, include_replicates=False)
        evaluate_research(data, spec, root / "validation", config, validation=True)
        selection = select_candidate(root / "validation" / "summary.json", config["selection"])
        selection["protocol_sha256"] = digest
        write_json(selection_path, selection)
    elif args.stage == "replicate":
        selection = json.loads(selection_path.read_text())
        variant = selection["variant"]
        settings = {**config["training"], "seeds": config["selection"]["additional_seeds"]}
        train_models(data, root / "replicates", TrainingConfig(**settings, **VARIANTS[variant]))
    else:
        # Generation happens only after the candidate and replicate weights are frozen.
        spec = specification(root, include_replicates=True)
        old = read_manifest(args.original)
        development = read_manifest(data)
        excluded = {r["group_id"] for r in old + development}
        fresh = root / "fresh_test"
        metadata = generate_dataset(
            fresh, DatasetConfig(**config["fresh_test"]), exclude_groups=excluded
        )
        write_json(root / "fresh_test_verified.json", verify_dataset(fresh))
        write_json(root / "fresh_test_metadata.json", metadata)
        evaluate_research(
            fresh, spec, root / "test", config, validation=False, exclude_groups=excluded
        )


if __name__ == "__main__":
    main()
