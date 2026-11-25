"""Export the learning-objective and search-rule comparisons."""

import argparse
import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_astar.data import load_record, read_manifest
from learned_astar.environments import Grid
from learned_astar.models import load_model
from learned_astar.research.experiment import execute_query
from learned_astar.research.plot import save
from learned_astar.search import SearchLimits
from learned_astar.terminal import render_map

COLORS = ["#176b91", "#48a0bd", "#bb5526", "#e8a350"]
FAMILIES = ("distance_raw_mean", "distance_tie_mean", "route_raw_mean", "route_tie_mean")
LABELS = ("Distance / direct", "Distance / ties", "Route / direct", "Route / ties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/research_v3"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v3"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/checkpoints/research_v3"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v3_figures"))
    args = parser.parse_args()
    root, run, output = args.results, args.run, args.output
    config = tomllib.loads((root / "protocol.toml").read_text())
    old_models = Path(config["sources"]["distance_models"])
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((root / "test/summary.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    splits = ("test", "walls", "density_40", "size_16", "size_64")
    ticks = ("Random 32", "Walls 32", "Dense 32", "Random 16", "Random 64")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("reachable_expansion_ratio", "reachable_runtime_ratio"),
        ("Expanded cells", "Total time, including prediction"),
        strict=True,
    ):
        for i, (name, label, color) in enumerate(zip(FAMILIES, LABELS, COLORS, strict=True)):
            estimates = [
                summary["splits"][s]["methods"][name]["paired_95ci"][metric] for s in splits
            ]
            means = np.array([v["mean"] for v in estimates])
            errors = np.array(
                [
                    [v["mean"] - v["low"] for v in estimates],
                    [v["high"] - v["mean"] for v in estimates],
                ]
            )
            ax.errorbar(
                np.arange(5) + (i - 1.5) * 0.15,
                means,
                yerr=errors,
                fmt="o",
                capsize=3,
                label=label,
                color=color,
            )
        ax.axhline(1, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(range(5), ticks, rotation=25)
        ax.set(title=title, ylabel="Ratio to Manhattan A* (lower is better)")
        ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fresh test · reachable queries · 3 seeds · map-bootstrap 95% intervals")
    save(fig, output, "search_rules")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, (split, title) in enumerate((("test", "Random 32×32"), ("walls", "Walls 32×32"))):
        errors = summary["splits"][split]["prediction_errors"]
        for row, (metric, ylabel) in enumerate(
            (
                ("branch_move_fraction", "Correct branching moves (%)"),
                ("mae", "Distance error (steps)"),
            )
        ):
            values = [errors["manhattan"][metric]["mean"]]
            for family in ("distance", "route"):
                values.append(
                    np.mean(
                        [errors[f"{family}_seed_{seed}"][metric]["mean"] for seed in (7, 17, 27)]
                    )
                )
            if row == 0:
                values = [100 * v for v in values]
            ax = axes[row, column]
            bars = ax.bar(range(3), values, color=["#777777", COLORS[0], COLORS[2]])
            ax.bar_label(bars, fmt="%.1f", padding=3)
            ax.set_xticks(range(3), ["Manhattan", "Distance", "Route"])
            ax.set(ylabel=ylabel, ylim=(0, 112 if row == 0 else max(values) * 1.22))
            if row == 0:
                ax.set_title(title)
    fig.suptitle("Learning diagnostics · map averages, then 3-seed means")
    save(fig, output, "local_moves")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for i, seed in enumerate((7, 17, 27)):
        old_dir = "direct" if seed == 7 else "replicates"
        old = json.loads((old_models / old_dir / f"history_{seed}.json").read_text())
        new = json.loads((args.models / f"history_{seed}.json").read_text())
        axes[0].plot(
            [r["epoch"] for r in old],
            [r["validation_loss"] for r in old],
            color=COLORS[i],
            linestyle="--",
            label=f"Distance, seed {seed}",
        )
        axes[0].plot(
            [r["epoch"] for r in new],
            [r["validation_distance_loss"] for r in new],
            color=COLORS[i],
            label=f"Route, seed {seed}",
        )
        axes[1].plot(
            [r["epoch"] for r in new],
            [r["validation_route_loss"] for r in new],
            color=COLORS[i],
            label=f"Seed {seed}",
        )
    axes[0].set(title="Distance loss, both objectives", ylabel="Smooth L1 on normalized distance")
    axes[1].set(title="Move loss, route training", ylabel="Negative log probability of good moves")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Checkpoint validation · same maps and training budget")
    save(fig, output, "training_objectives")

    torch.set_num_threads(1)
    record = read_manifest(run / "fresh_test", "walls")[0]
    data = load_record(run / "fresh_test", record)
    grid = Grid(
        *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
    )
    goal, start = (tuple(map(int, data[key][0])) for key in ("goals", "starts"))
    distance, _ = load_model(old_models / "direct/seed_7.pt")
    route, _ = load_model(args.models / "seed_7.pt")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), layout="constrained")
    examples = []
    for ax, label, method in zip(
        axes,
        ("Manhattan", "Distance / ties", "Route / ties"),
        ((None, "manhattan"), (distance, "tie"), (route, "tie")),
        strict=True,
    ):
        result, _ = execute_query(
            method, grid, start, goal, data["distances"][0], SearchLimits(time_limit_seconds=5)
        )
        ax.imshow(data["blocked"], cmap="Greys", vmin=0, vmax=1)
        if result.path:
            path = np.asarray(result.path)
            ax.plot(path[:, 1], path[:, 0], color="#2185a0", linewidth=2)
        ax.scatter(start[1], start[0], color="#e76639", s=40, edgecolors="black", zorder=3)
        ax.scatter(
            goal[1], goal[0], color="#ffd447", marker="*", s=110, edgecolors="black", zorder=3
        )
        ax.set(
            title=f"{label}\nCost {result.cost} · expanded {result.stats.expanded}",
            xticks=[],
            yticks=[],
        )
        examples.append({"method": label, "cost": result.cost, "expanded": result.stats.expanded})
    fig.suptitle("First fresh wall query · seed 7 · orange start, yellow goal")
    save(fig, output, "example_paths")
    (output / "example_map.txt").write_text(render_map(grid, start, goal) + "\n")
    (output / "example.json").write_text(
        json.dumps(
            {
                "layout_id": record["layout_id"],
                "query": 0,
                "start": start,
                "goal": goal,
                "optimal_cost": int(data["distances"][0][start]),
                "results": examples,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
