"""Train smaller widths and select them before generating a fresh test set."""

import argparse
import json
import shutil
import tomllib
from pathlib import Path

from learned_astar.data import DatasetConfig, generate_dataset, read_manifest, verify_dataset
from learned_astar.research.experiment import digest, evaluate_research, save_json
from learned_astar.research.lightweight import choose_width, make_spec, paired_widths
from learned_astar.training.train import TrainingConfig, train_models


def fingerprints(config):
    paths = list(Path("src/learned_astar").rglob("*.py")) + [Path(__file__)]
    paths += [Path(p) / "manifest.jsonl" for p in config["sources"]]
    paths += [
        Path(config["reference_models"]) / f"seed_{s}.pt" for s in config["training"]["seeds"]
    ]
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def previous_groups(config):
    return {r["group_id"] for p in config["sources"] for r in read_manifest(p)}


def generate_checked(root, name, config, excluded):
    generate_dataset(root / name, DatasetConfig(**config[name]), exclude_groups=excluded)
    checked = verify_dataset(root / name)
    groups = {r["group_id"] for r in read_manifest(root / name)}
    if groups & excluded:
        raise ValueError("New layouts overlap earlier maps")
    checked.update(excluded_groups=len(excluded), disjoint=True)
    save_json(root / f"{name}_verified.json", checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "select", "test"))
    parser.add_argument("--config", type=Path, default=Path("configs/lightweight.toml"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v5"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    root = args.run
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.config, root / "protocol.toml")
        save_json(root / "fingerprints.json", fingerprints(config))
        generate_checked(root, "policy_validation", config, previous_groups(config))
        return
    if (root / "protocol.toml").read_bytes() != args.config.read_bytes():
        raise ValueError("Protocol changed after preparation")
    if fingerprints(config) != json.loads((root / "fingerprints.json").read_text()):
        raise ValueError("Implementation, reference models, or source data changed")
    if args.stage == "train":
        for width in config["widths"]:
            if width != config["reference_width"]:
                print(f"Training width={width}", flush=True)
                train_models(
                    config["development"],
                    root / f"width_{width}",
                    TrainingConfig(**config["training"], channels=width),
                )
        return
    if args.stage == "select":
        spec = make_spec(root, config)
        summary = evaluate_research(
            root / "policy_validation", spec, root / "validation", config, validation=True
        )
        paired = paired_widths(root / "validation", config)
        save_json(root / "validation/paired_widths.json", paired)
        selection = choose_width(summary, paired, config)
        selection.update(
            protocol_sha256=digest(root / "protocol.toml"),
            validation_summary_sha256=digest(root / "validation/summary.json"),
            validation_pairs_sha256=digest(root / "validation/paired_widths.json"),
        )
        save_json(root / "selection.json", selection)
        print(json.dumps(selection, indent=2))
        return
    selection = json.loads((root / "selection.json").read_text())
    for key, name in (
        ("protocol_sha256", "protocol.toml"),
        ("validation_summary_sha256", "validation/summary.json"),
        ("validation_pairs_sha256", "validation/paired_widths.json"),
    ):
        if selection[key] != digest(root / name):
            raise ValueError("Selection evidence changed")
    validation = json.loads((root / "validation/summary.json").read_text())
    paired = json.loads((root / "validation/paired_widths.json").read_text())
    if any(selection[k] != v for k, v in choose_width(validation, paired, config).items()):
        raise ValueError("Selection no longer follows the fixed rule")
    spec = make_spec(root, config, selection)
    for name, path in spec["models"].items():
        if digest(path) != validation["settings"]["checkpoints"][name]["sha256"]:
            raise ValueError("Checkpoint changed after selection")
    excluded = previous_groups(config) | {
        r["group_id"] for r in read_manifest(root / "policy_validation")
    }
    generate_checked(root, "fresh_test", config, excluded)
    evaluate_research(
        root / "fresh_test", spec, root / "test", config, validation=False, exclude_groups=excluded
    )
    save_json(root / "test/paired_widths.json", paired_widths(root / "test", config))


if __name__ == "__main__":
    main()
