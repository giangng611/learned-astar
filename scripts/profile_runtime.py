"""Profile the fixed validation workload; these instrumented times are diagnostic only."""

import cProfile
import io
import pstats
from pathlib import Path

import numpy as np
import torch

from learned_astar.data import load_record, read_manifest
from learned_astar.environments import Grid
from learned_astar.models import load_model
from learned_astar.research.experiment import execute_query, query_starts, save_json
from learned_astar.search import SearchLimits


def main():
    root = Path("outputs/research_v3/policy_validation")
    output = Path("outputs/runtime_profile")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, _ = load_model("artifacts/checkpoints/research_v3/seed_7.pt")
    records = read_manifest(root, "validation")
    jobs, layouts = [], []
    for kind in ("random", "walls"):
        for record in [r for r in records if r["kind"] == kind][:16]:
            data = load_record(root, record)
            grid = Grid(
                *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
            )
            goal = tuple(map(int, data["goals"][0]))
            layouts.append(record["layout_id"])
            for start in query_starts(record, data, grid, goal, True):
                jobs.append((grid, start, goal, data["distances"][0]))
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(5):
        for grid, start, goal, labels in jobs:
            execute_query(
                (model, "tie"), grid, start, goal, labels, SearchLimits(time_limit_seconds=5)
            )
    profile.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    (output / "profile.txt").write_text(stream.getvalue())
    save_json(
        output / "profile.json",
        {
            "purpose": "Instrumented validation diagnosis; not a runtime benchmark.",
            "layouts": layouts,
            "seed": 7,
            "queries": len(jobs),
            "repetitions": 5,
            "functions": [
                {
                    "file": key[0],
                    "line": key[1],
                    "function": key[2],
                    "primitive_calls": value[0],
                    "total_calls": value[1],
                    "self_seconds": value[2],
                    "cumulative_seconds": value[3],
                }
                for key, value in sorted(stats.stats.items(), key=lambda p: -p[1][3])[:30]
            ],
        },
    )
    print(stream.getvalue())


if __name__ == "__main__":
    main()
