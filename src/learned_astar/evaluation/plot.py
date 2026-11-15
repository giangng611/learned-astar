"""Recreate the result figures from saved measurements and checkpoints."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learned_astar.data import load_record, read_manifest
from learned_astar.environments import Grid
from learned_astar.models import load_model, predict_distances
from learned_astar.reproducibility import configure_torch
from learned_astar.search import astar
from learned_astar.search.learned import learned_astar

LABELS = {
    "bfs": "BFS",
    "dijkstra": "Dijkstra",
    "astar_manhattan": "Manhattan A*",
    "weighted_astar_1.25": "Weighted 1.25",
    "weighted_astar_1.5": "Weighted 1.5",
    "weighted_astar_2.0": "Weighted 2.0",
    "learned_mean": "Learned (3 seeds)",
}
COLORS = ["#94A3B8", "#64748B", "#1D4E89", "#F2B880", "#DF8E46", "#BC6425", "#008A78"]


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(results, training, data, checkpoints, output):
    results, training, output = Path(results), Path(training), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / "summary.json").read_text())
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    main = summary["splits"]["test"]
    names = list(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("completion_ms", "expanded", "path_ratio"),
        ("Mean total query time (ms)", "Mean expanded cells", "Mean path / optimal cost"),
        strict=True,
    ):
        values = [main["methods"][n][metric]["mean"] for n in names]
        ax.barh([LABELS[n] for n in names], values, color=COLORS)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(left=0)
        for index, value in enumerate(values):
            ax.text(
                value,
                index,
                f" {value:.3f}" if metric != "expanded" else f" {value:.1f}",
                va="center",
                fontsize=10,
            )
        ax.margins(x=0.2)
    save_figure(fig, output, "main_comparison")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for path in sorted(training.glob("history_*.json")):
        history = json.loads(path.read_text())
        seed = path.stem.split("_")[1]
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_loss"] for r in history],
            label=f"Seed {seed}",
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation Smooth L1 loss",
        title="Validation loss by training seed",
        yscale="log",
    )
    ax.legend(frameon=False)
    save_figure(fig, output, "training")
    split_names = {
        "test": "32×32 main",
        "density_40": "40% obstacles",
        "size_16": "16×16",
        "size_64": "64×64",
        "walls": "Walls",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("runtime_ratio", "expansion_ratio"),
        ("Total time / Manhattan A*", "Expansions / Manhattan A*"),
        strict=True,
    ):
        for offset, name, color in (
            (-0.12, "learned_mean", "#008A78"),
            (0.12, "weighted_astar_1.5", "#DF8E46"),
        ):
            values = [summary["splits"][s]["paired_95ci"][name][metric] for s in split_names]
            means = np.array([v["mean"] for v in values])
            errors = np.array(
                [[v["mean"] - v["low"] for v in values], [v["high"] - v["mean"] for v in values]]
            )
            ax.errorbar(
                means,
                np.arange(len(values)) + offset,
                xerr=errors,
                fmt="o",
                capsize=3,
                color=color,
                label=LABELS[name],
            )
        ax.axvline(1, color="#64748B", linestyle="--")
        ax.set(yticks=range(len(split_names)), yticklabels=list(split_names.values()), xlabel=title)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=10)
    save_figure(fig, output, "generalization")
    # The first main-test query is fixed before inspecting any algorithm result.
    record = read_manifest(data, "test")[0]
    arrays = load_record(data, record)
    blocked = arrays["blocked"]
    grid = Grid(*blocked.shape, frozenset(map(tuple, np.argwhere(blocked).tolist())))
    goal = tuple(map(int, arrays["goals"][0]))
    start = tuple(map(int, arrays["starts"][0]))
    labels = arrays["distances"][0]
    configure_torch(4550, 1)
    model, meta = load_model(Path(checkpoints) / "seed_7.pt")
    prediction = predict_distances(model, grid, goal)
    reference_result = astar(grid, start, goal)
    learned_result = learned_astar(grid, start, goal, model)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    vmax = max(float(labels.max()), float(prediction[~blocked].max()))
    for ax, values, title in zip(
        axes[:2],
        (labels.astype(float), prediction),
        ("Exact distance to goal", "Predicted distance · seed 7"),
        strict=True,
    ):
        shown = np.ma.masked_where((labels < 0), values)
        ax.set_facecolor("#263445")
        im = ax.imshow(shown, vmin=0, vmax=vmax, cmap="viridis")
        ax.scatter(goal[1], goal[0], c="#EC4899", marker="*", s=100)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[:2], shrink=0.75, label="Distance (moves)")
    axes[2].imshow(blocked, cmap="Greys", vmin=0, vmax=1)
    for result, color, label in (
        (reference_result, "#1D4E89", "Manhattan A*"),
        (learned_result, "#D77A24", "Learned seed 7"),
    ):
        if result.path:
            points = np.asarray(result.path)
            axes[2].plot(
                points[:, 1],
                points[:, 0],
                color=color,
                linewidth=2,
                label=f"{label}: cost {result.cost}",
            )
    axes[2].scatter(
        [start[1], goal[1]], [start[0], goal[0]], c=["#008A78", "#EC4899"], s=70, zorder=5
    )
    axes[2].set_title("Same query, returned paths")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=10, frameon=False)
    save_figure(fig, output, "distance_example")
    (output / "example.json").write_text(
        json.dumps(
            {
                "layout_id": record["layout_id"],
                "query": 0,
                "selection": "First main-test map, first start; seed 7 fixed in advance",
                "seed": meta["seed"],
                "start": start,
                "goal": goal,
                "astar_cost": reference_result.cost,
                "learned_cost": learned_result.cost,
            },
            indent=2,
        )
        + "\n"
    )
    with gzip.open(results / "queries.jsonl.gz", "rt") as stream:
        query_rows = [json.loads(line) for line in stream]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name, color in (
        ("weighted_astar_1.5", "#DF8E46"),
        ("learned_seed_7", "#008A78"),
        ("learned_seed_17", "#1D4E89"),
        ("learned_seed_27", "#8B5CF6"),
    ):
        values = sorted(
            r["path_ratio"]
            for r in query_rows
            if r["split"] == "test" and r["method"] == name and r["path_ratio"] is not None
        )
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=name.replace("_", " "),
            color=color,
        )
    ax.set(
        xlabel="Path cost / optimal cost",
        ylabel="Fraction of found paths",
        title="Path quality on the main test set",
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=10)
    save_figure(fig, output, "path_quality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_figures(args.results, args.training, args.data, args.checkpoints, args.output)


if __name__ == "__main__":
    main()
