"""Inspect local consistency after validation without changing model selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from learned_astar.data import load_record, read_manifest
from learned_astar.environments import Grid
from learned_astar.evaluation.summarize import distribution
from learned_astar.models import load_model
from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import map_predictions, specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v2"))
    args = parser.parse_args()
    configure_torch(7, 1)
    models = {
        name: load_model(path)[0]
        for name, path in specification(args.run, include_replicates=False)["models"].items()
    }
    rows = []
    for record in read_manifest(args.run / "development", "validation"):
        data = load_record(args.run / "development", record)
        grid = Grid(
            *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
        )
        goal = tuple(map(int, data["goals"][0]))
        rows.append(
            {
                "kind": record["kind"],
                "errors": map_predictions(models, grid, goal, data["distances"][0]),
            }
        )
    summary = {}
    for kind in ("random", "walls"):
        selected = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            name: {
                metric: distribution(
                    [
                        r["errors"][name][metric]
                        for r in selected
                        if r["errors"][name][metric] is not None
                    ]
                )
                for metric in selected[0]["errors"][name]
            }
            for name in selected[0]["errors"]
        }
    path = args.run / "validation_diagnostics.json"
    if path.exists():
        raise ValueError("Diagnostics output already exists")
    path.write_text(
        json.dumps(
            {
                "purpose": "Exploratory validation analysis; not used for selection",
                "groups": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
