"""Export scientific figures for the controlled follow-up experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_astar.data import load_record, read_manifest
from learned_astar.environments import Grid
from learned_astar.models import load_model, predict_distances
from learned_astar.models.transforms import manhattan_field, parity_distances
from learned_astar.research.experiment import execute_query
from learned_astar.search import SearchLimits

COLORS = ["#176b91", "#c76e23", "#389272", "#8764a8", "#888888"]


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_research(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run / "test" / "summary.json").read_text())
    selection = summary["settings"]["selection"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for name, label, color in zip(
        ("direct", "residual", "no_obstacles"),
        ("Direct distance", "Learned detour", "Detour without obstacles"),
        COLORS,
        strict=False,
    ):
        history = json.loads((run / name / "history_7.json").read_text())
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_loss"] for r in history],
            label=label,
            color=color,
            linewidth=2,
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation loss (normalized distances)",
        title="Controlled training runs · same data, width, and seed",
    )
    ax.legend(frameon=False)
    save(fig, output, "training_controls")

    splits = ["test", "walls", "density_40", "size_16", "size_64"]
    labels = ["Random 32", "Walls 32", "Dense 32", "Random 16", "Random 64"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("reachable_expansion_ratio", "reachable_runtime_ratio"),
        ("Expanded cells · reachable queries", "Total time · reachable queries"),
        strict=True,
    ):
        for offset, (method, label, color) in enumerate(
            zip(
                ("original_mean", "selected_mean", "weighted_astar_1.5"),
                ("Original CNN (3 seeds)", "Selected CNN (3 seeds)", "Weighted A* (1.5)"),
                COLORS,
                strict=False,
            )
        ):
            estimates = [
                summary["splits"][s]["methods"][method]["paired_95ci"][metric] for s in splits
            ]
            means = np.array([e["mean"] for e in estimates])
            errors = np.array(
                [
                    [e["mean"] - e["low"] for e in estimates],
                    [e["high"] - e["mean"] for e in estimates],
                ]
            )
            ax.errorbar(
                np.arange(len(splits)) + (offset - 1) * 0.18,
                means,
                yerr=errors,
                fmt="o",
                capsize=3,
                label=label,
                color=color,
            )
        ax.axhline(1, color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(splits)), labels, rotation=25)
        ax.set(title=title, ylabel="Ratio to Manhattan A* (lower is better)")
        if metric.endswith("runtime_ratio"):
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fresh test · map bootstrap 95% intervals")
    save(fig, output, "search_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    names = [
        "manhattan",
        "direct_seed_7",
        "residual_seed_7",
        "residual_seed_7_hidden",
        "no_obstacles_seed_7",
    ]
    ticks = ["Manhattan", "Direct", "Detour", "Detour\ninput hidden", "No-obstacle\ntraining"]
    for ax, split, title in zip(
        axes, ("test", "walls"), ("Random 32×32", "Walls 32×32"), strict=True
    ):
        errors = summary["splits"][split]["prediction_errors"]
        values = [errors[name]["mae"]["mean"] for name in names]
        bars = ax.bar(range(len(names)), values, color=COLORS)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks(range(len(names)), ticks, fontsize=10)
        ax.set(
            ylabel="Mean absolute distance error (steps)", title=title, ylim=(0, max(values) * 1.23)
        )
    fig.suptitle("Obstacle controls · seed 7, raw predictions, reachable cells")
    save(fig, output, "obstacle_controls")

    # First wall layout and first query, fixed by file order rather than by model outcome.
    torch.set_num_threads(1)
    record = read_manifest(run / "fresh_test", "walls")[0]
    data = load_record(run / "fresh_test", record)
    grid = Grid(
        *data["blocked"].shape, frozenset(map(tuple, np.argwhere(data["blocked"]).tolist()))
    )
    goal, start = (tuple(map(int, data[key][0])) for key in ("goals", "starts"))
    model, _ = load_model(run / selection["variant"] / "seed_7.pt")
    prediction = predict_distances(model, grid, goal)
    if selection["mode"] == "parity":
        prediction = parity_distances(prediction, goal)
    labels = data["distances"][0]
    lower = manhattan_field(labels.shape, goal)
    fields = [lower, labels, prediction]
    names = ["Manhattan distance", "Exact BFS distance", "Selected CNN heuristic"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#222222")
    maximum = max(float(np.max(f[labels >= 0])) for f in fields)
    for ax, field, title in zip(axes, fields, names, strict=True):
        masked = np.ma.masked_where(labels < 0, field)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=maximum)
        ax.scatter([start[1]], [start[0]], color="#ff5348", marker="o", s=40, edgecolors="white")
        ax.scatter([goal[1]], [goal[0]], color="#ffffff", marker="*", s=95, edgecolors="#111111")
        ax.set(title=title, xticks=[], yticks=[])
    fig.colorbar(im, ax=axes, shrink=0.75, label="Steps to goal")
    fig.suptitle("First fresh wall map · red start, white goal, dark obstacles")
    save(fig, output, "distance_fields")
    result, _ = execute_query(
        (model, selection["mode"]), grid, start, goal, labels, SearchLimits(time_limit_seconds=5)
    )
    from learned_astar.terminal import render_map

    (output / "example_map.txt").write_text(render_map(grid, start, goal) + "\n")
    (output / "example.json").write_text(
        json.dumps(
            {
                "layout_id": record["layout_id"],
                "query": 0,
                "start": start,
                "goal": goal,
                "variant": selection["variant"],
                "mode": selection["mode"],
                "seed": 7,
                "optimal_cost": int(labels[start]),
                "returned_cost": result.cost,
                "expanded": result.stats.expanded,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_research(args.run, args.output)
