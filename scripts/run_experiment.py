"""Reproduce all experiments in a new directory without replacing earlier runs."""

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    settings = run / "configs"
    settings.mkdir()
    for name in ("dataset", "training", "evaluation"):
        shutil.copyfile(root / "configs" / f"{name}.toml", settings / f"{name}.toml")
    data, models, raw, summary, figures = [
        run / name for name in ("data", "models", "raw", "summary", "figures")
    ]
    seeds = tomllib.loads((settings / "training.toml").read_text())["seeds"]
    commands = [
        ["learned_astar.data", "generate", "--config", settings / "dataset.toml", "--output", data],
        ["learned_astar.data", "verify", "--output", data],
        [
            "learned_astar.training.train",
            "--data",
            data,
            "--config",
            settings / "training.toml",
            "--output",
            models,
        ],
        [
            "learned_astar.evaluation.run",
            "--data",
            data,
            "--checkpoints",
            *[models / f"seed_{seed}.pt" for seed in seeds],
            "--config",
            settings / "evaluation.toml",
            "--output",
            raw,
        ],
        ["learned_astar.evaluation.summarize", "--input", raw, "--output", summary],
        [
            "learned_astar.evaluation.plot",
            "--results",
            summary,
            "--training",
            models,
            "--data",
            data,
            "--checkpoints",
            models,
            "--output",
            figures,
        ],
    ]
    for command in commands:
        print(f"\nRunning {command[0]}", flush=True)
        subprocess.run([sys.executable, "-m", *map(str, command)], cwd=root, check=True)
    print(f"\nFinished. Results: {summary}; figures: {figures}")


if __name__ == "__main__":
    main()
