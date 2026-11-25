"""Verify sessions, retain evidence, and report setup-inclusive runtime."""

import argparse
import gzip
import json
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learned_astar.evaluation.summarize import bootstrap_mean, distribution
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.plot import save
from learned_astar.research.runtime import methods

LABELS = {
    "manhattan": "Original Manhattan",
    "prepared_manhattan": "Prepared Manhattan",
    "fresh_mean": "Fresh CNN / ties",
    "cached_mean": "Reused CNN / original search",
    "prepared_mean": "Reused CNN / prepared search",
    "prepared_exact": "Prepared exact A*",
    "bfs_policy": "Reverse-BFS policy",
}
SPLITS = {
    "test": "Random 32",
    "walls": "Walls 32",
    "density_40": "Dense 32",
    "size_16": "Random 16",
    "size_64": "Random 64",
}
FIELDS = (
    "total_seconds",
    "setup_seconds",
    "search_seconds",
    "graph_seconds",
    "prediction_seconds",
    "reference_seconds",
)


def load_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    metadata = json.loads((root / "measurement.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != metadata[f"{name}_sha256"]:
            raise ValueError("Measurement checksum mismatch")
    raw, queries = load_rows(root / "sessions.jsonl.gz"), load_rows(root / "queries.jsonl.gz")
    if len(raw) != metadata["sessions"] or len(queries) != metadata["query_records"]:
        raise ValueError("Record counts changed")
    grouped = defaultdict(list)
    for row in raw:
        grouped[row["layout_id"], row["batch_size"], row["method"]].append(row)
    medians = []
    for group in grouped.values():
        if len(group) != config["repetitions"] or {r["repetition"] for r in group} != set(
            range(config["repetitions"])
        ):
            raise ValueError("Missing or duplicate repetitions")
        if len({r["result_sha256"] for r in group}) != 1:
            raise ValueError("Repeated paths or counters changed")
        row = {k: v for k, v in group[0].items() if k != "repetition"}
        row.update({field: float(np.median([r[field] for r in group])) for field in FIELDS})
        medians.append(row)
    lookup = {(r["layout_id"], r["batch_size"], r["method"]): r for r in medians}
    covered = defaultdict(set)
    for row in medians:
        covered[row["layout_id"], row["batch_size"]].add(row["method"])
    for (layout, size), names in covered.items():
        if names != set(methods(config["seeds"], size)):
            raise ValueError("Methods are not paired")
        for first, second in [("manhattan", "prepared_manhattan")] + [
            (f"cached_{s}", f"prepared_{s}") for s in config["seeds"]
        ]:
            if (
                lookup[layout, size, first]["result_sha256"]
                != lookup[layout, size, second]["result_sha256"]
            ):
                raise ValueError("Prepared search changed behavior")
    if any(r["cost"] != (r["optimal_cost"] if r["optimal_cost"] >= 0 else None) for r in queries):
        raise ValueError("Nonoptimal query")
    rng = np.random.default_rng(config["bootstrap_seed"])
    summary = {
        "metadata": metadata,
        "groups": {},
        "verification": {
            "repetitions_and_checksums": True,
            "prepared_behavior_matches": True,
            "all_path_costs_optimal": True,
            "session_medians": len(medians),
        },
    }
    for split in SPLITS:
        summary["groups"][split] = {}
        for size in config["batch_sizes"]:
            groups = defaultdict(lambda: defaultdict(list))
            for row in medians:
                if row["split"] != split or row["batch_size"] != size:
                    continue
                name = row["method"]
                aggregate = (
                    name.rsplit("_", 1)[0] + "_mean" if name.rsplit("_", 1)[-1].isdigit() else name
                )
                groups[aggregate][row["layout_id"]].append(row)
            outputs = {}
            for name, maps in groups.items():
                metrics = defaultdict(list)
                for layout, rows in maps.items():
                    for field in FIELDS:
                        metrics[field.replace("_seconds", "_ms_per_query")].append(
                            np.mean([r[field] for r in rows]) * 1000 / size
                        )
                    total = np.mean([r["total_seconds"] for r in rows])
                    for ref in ("manhattan", "prepared_manhattan", "bfs_policy"):
                        metrics[f"ratio_to_{ref}"].append(
                            total / lookup[layout, size, ref]["total_seconds"]
                        )
                outputs[name] = {
                    key: bootstrap_mean(values, rng, config["bootstrap_samples"])
                    if key.startswith("ratio_")
                    else distribution(values)
                    for key, values in metrics.items()
                }
            reference = [
                r
                for r in queries
                if r["split"] == split and r["batch_size"] == size and r["method"] == "manhattan"
            ]
            summary["groups"][split][str(size)] = {
                "methods": outputs,
                "queries": len(reference),
                "reachable": sum(r["optimal_cost"] >= 0 for r in reference),
            }
    return summary, medians


def figures(summary, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    selected = ["manhattan", "prepared_manhattan", "cached_mean", "prepared_mean", "bfs_policy"]
    colors = ["#777777", "#185c8b", "#d69a4b", "#b64d25", "#2e8b65"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, split in zip(axes, ("test", "walls"), strict=True):
        for name, color in zip(selected, colors, strict=True):
            ax.plot(
                [1, 4, 16, 64],
                [
                    summary["groups"][split][str(n)]["methods"][name]["total_ms_per_query"]["mean"]
                    for n in (1, 4, 16, 64)
                ],
                marker="o",
                color=color,
                label=LABELS[name],
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Queries sharing one map and goal",
            ylabel="Mean total ms per query",
            title=SPLITS[split],
        )
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fresh maps · all preparation included · 3-seed learned means")
    save(fig, output, "reuse_runtime")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, reference, title in zip(
        axes,
        ("manhattan", "prepared_manhattan"),
        ("Against original Manhattan", "Against prepared Manhattan"),
        strict=True,
    ):
        for offset, (name, color) in enumerate(
            zip(("prepared_mean", "bfs_policy"), (colors[3], colors[4]), strict=True)
        ):
            values = [
                summary["groups"][s]["64"]["methods"][name][f"ratio_to_{reference}"] for s in SPLITS
            ]
            ax.errorbar(
                np.arange(5) + (offset - 0.5) * 0.15,
                [v["mean"] for v in values],
                yerr=[
                    [v["mean"] - v["low"] for v in values],
                    [v["high"] - v["mean"] for v in values],
                ],
                fmt="o",
                capsize=3,
                color=color,
                label=LABELS[name],
            )
        ax.axhline(1, color="#555555", linestyle="--")
        ax.set_xticks(range(5), SPLITS.values(), rotation=25)
        ax.set(title=title, ylabel="Paired total-time ratio (lower is better)")
        ax.legend(frameon=True, facecolor="white", framealpha=1, edgecolor="none", fontsize=9)
    fig.suptitle("64-query sessions · whole-map bootstrap 95% intervals")
    save(fig, output, "fair_comparison")


def report(summary, output_path):
    random_methods = summary["groups"]["test"]["64"]["methods"]
    wall_methods = summary["groups"]["walls"]["64"]["methods"]
    random_ratio = random_methods["prepared_mean"]
    wall_ratio = wall_methods["prepared_mean"]
    lines = [
        "# Runtime and repeated-query results",
        "",
        "This experiment follows the [fixed runtime plan](runtime_reuse.md). It keeps the "
        "three route-trained models unchanged and measures entire sessions, including map "
        "preparation and one prediction or reverse BFS where needed. It uses 300 new maps, "
        "excluding all 5,180 earlier layout groups. Each session contains 1, 4, 16, or 64 "
        "different starts sharing a goal.",
        "",
        "Prepared search and reuse made the learned method faster than the original "
        "Manhattan implementation in 64-query sessions. However, equally prepared Manhattan "
        "was still faster in every condition. The learned method's mean paired time ratio "
        f"was {random_ratio['ratio_to_prepared_manhattan']['mean']:.3f} on random 32×32 maps "
        f"and {wall_ratio['ratio_to_prepared_manhattan']['mean']:.3f} on wall maps. "
        "Reverse BFS with stored next steps was fastest among the measured methods in "
        "these 64-query sessions. Thus the largest runtime gain came from implementation "
        "and reuse, rather than from learned ordering.",
        "",
        "## Complete session cost",
        "",
        "Values are milliseconds per query: the measured session time divided by its query "
        "count, then averaged across maps. Five shuffled repetitions supply each session's "
        "median. Learned rows average seeds 7, 17, and 27. These workloads include unreachable "
        "queries. Model loading is outside timing; all map and goal preparation is inside.",
        "",
    ]
    for split in ("test", "walls"):
        lines += [
            f"### {SPLITS[split]}",
            "",
            "| Method | 1 query | 4 queries | 16 queries | 64 queries |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, label in LABELS.items():
            values = [summary["groups"][split][str(n)]["methods"].get(name) for n in (1, 4, 16, 64)]
            lines.append(
                f"| {label} | "
                + " | ".join(f"{v['total_ms_per_query']['mean']:.4f}" if v else "—" for v in values)
                + " |"
            )
        lines.append("")
    lines += [
        "Fresh CNN predictions are measured only for single-query sessions. Reused "
        "predictions retain the original queue implementation; prepared search additionally "
        "uses integer cell IDs, precomputed neighbors, and once-validated score fields. "
        "Prepared Manhattan receives the same implementation changes. Reverse-BFS policy "
        "stores next steps and returns actual routes without an A* queue.",
        "",
        "## Comparison on every condition",
        "",
        "The table uses 64-query sessions. Ratios pair each map with its baseline, averaging "
        "the three learned models inside that map. Brackets are 95% intervals from 2,000 "
        "whole-map bootstrap samples. A ratio below one means less total time. Comparing "
        "against both Manhattan implementations prevents attributing a shared code speedup "
        "to learning.",
        "",
        "| Condition | Reachable / queries | Learned / original Manhattan | "
        "Learned / prepared Manhattan | BFS policy / prepared Manhattan |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, label in SPLITS.items():
        group = summary["groups"][split]["64"]
        values = [
            group["methods"][method][f"ratio_to_{reference}"]
            for method, reference in (
                ("prepared_mean", "manhattan"),
                ("prepared_mean", "prepared_manhattan"),
                ("bfs_policy", "prepared_manhattan"),
            )
        ]
        lines.append(
            f"| {label} | {group['reachable']} / {group['queries']} | "
            + " | ".join(f"{v['mean']:.3f} [{v['low']:.3f}, {v['high']:.3f}]" for v in values)
            + " |"
        )
    lines += [
        "",
        "## Where time goes",
        "",
        "The following stage times are per query in 64-query sessions. Setup includes map "
        "preparation, a prediction or BFS where applicable, and planner creation. Stage "
        "medians are summarized independently, so they need not sum exactly to the total.",
        "",
        "| Condition | Method | Setup ms/query | Search or path-following ms/query |",
        "| --- | --- | ---: | ---: |",
    ]
    for split in ("test", "walls"):
        for name in ("prepared_manhattan", "prepared_mean", "bfs_policy"):
            row = summary["groups"][split]["64"]["methods"][name]
            lines.append(
                f"| {SPLITS[split]} | {LABELS[name]} | {row['setup_ms_per_query']['mean']:.4f} | "
                f"{row['search_ms_per_query']['mean']:.4f} |"
            )
    lines += [
        "",
        "## Scope and checks",
        "",
        "At 64 queries, the learned search phase alone averaged "
        f"{random_ratio['search_ms_per_query']['mean']:.4f} ms versus prepared Manhattan's "
        f"{random_methods['prepared_manhattan']['search_ms_per_query']['mean']:.4f} ms on "
        f"random maps, and {wall_ratio['search_ms_per_query']['mean']:.4f} ms versus "
        f"{wall_methods['prepared_manhattan']['search_ms_per_query']['mean']:.4f} ms on walls. "
        "Its larger setup cost erased these small savings. Reusing predictions without "
        "changing the original queue implementation also remained slower than original "
        "Manhattan. The model was not retrained because this experiment targeted execution "
        "cost, while the previous round already established improved move choices.",
        "",
        "The prepared implementations returned exactly the same paths and all search "
        "counters as their original counterparts on every measured query. This is an "
        "implementation comparison, not a new training result. All methods returned "
        "optimal paths and correct reachability; no limits were reached. The BFS policy "
        "still does work during preprocessing and while following paths. Session records "
        "retain its preprocessing expansion count, while query records retain path steps.",
        "",
        "Reuse assumes a static map and unchanged goal. Any change requires new preparation. "
        "The results do not establish benefits for moving obstacles or changing goals. "
        "Intervals describe maps for three fixed models on this CPU. Correlated batch sizes "
        "are separate workload settings rather than extra independent samples.",
        "",
        "![Runtime as queries are reused](../results/research_v4/reuse_runtime.png)",
        "",
        "![Comparison with both Manhattan implementations]"
        "(../results/research_v4/fair_comparison.png)",
        "",
        f"Verified {summary['metadata']['sessions']:,} timed sessions and "
        f"{summary['metadata']['timed_queries']:,} query executions. The "
        "[archive](../results/research_v4/) includes raw session repetitions, collapsed "
        "medians, query outcomes, checksums, dataset provenance, and PNG/PDF figures. "
        "The [third experiment](search_learning_results.md) remains the source for "
        "claims about learning quality and expansion reductions.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v4"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v4"))
    parser.add_argument("--report", type=Path, default=Path("docs/runtime_results.md"))
    args = parser.parse_args()
    summary, medians = analyze(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "sessions.jsonl.gz",
        "queries.jsonl.gz",
        "protocol.toml",
        "fingerprints.json",
        "dataset_verified.json",
        "measurement.json",
    ):
        shutil.copy2(args.run / name, args.output / name)
    shutil.copy2(args.run / "fresh_test/metadata.json", args.output / "dataset_metadata.json")
    with gzip.open(args.output / "manifest.jsonl.gz", "wb") as stream:
        stream.write((args.run / "fresh_test/manifest.jsonl").read_bytes())
    with gzip.open(args.output / "session_medians.jsonl.gz", "wt") as stream:
        for row in medians:
            stream.write(json.dumps(row) + "\n")
    save_json(args.output / "summary.json", summary)
    figures(summary, args.output)
    report(summary, args.report)
    print(json.dumps(summary["metadata"], indent=2))


if __name__ == "__main__":
    main()
