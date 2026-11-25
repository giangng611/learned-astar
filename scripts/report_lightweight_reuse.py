"""Verify and summarize the frozen width-by-reuse comparison."""

import argparse
import gzip
import json
import shutil
import tomllib
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_runtime_reuse import FIELDS, SPLITS
from run_lightweight_reuse import CONTROLS, method_map

from learned_astar.evaluation.summarize import bootstrap_mean, distribution
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.plot import save

LABELS = {
    "manhattan": "Original Manhattan",
    "prepared_manhattan": "Prepared Manhattan",
    "prepared_exact": "Prepared exact A*",
    "bfs_policy": "Reverse-BFS policy",
    "width_16": "CNN 16 channels",
    "width_8": "CNN 8 channels",
    "width_4": "CNN 4 channels",
}
COLORS = {
    "prepared_manhattan": "#354555",
    "width_16": "#c45a32",
    "width_8": "#246da1",
    "width_4": "#9a5ea0",
    "bfs_policy": "#288464",
}


def rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def session_key(row):
    return row["layout_id"], row["batch_size"], row["method"]


def expected_keys(config, manifest):
    return {
        (record["layout_id"], size, name)
        for record in manifest
        for size in config["batch_sizes"]
        for name in method_map(config)
    }


def collapse_sessions(raw, config, manifest):
    grouped = defaultdict(list)
    splits = {r["layout_id"]: r["split"] for r in manifest}
    for row in raw:
        if row["split"] != splits.get(row["layout_id"]):
            raise ValueError("Session layout or split changed")
        values = [row[field] for field in FIELDS]
        if not np.isfinite(values).all() or min(values) < 0 or row["total_seconds"] <= 0:
            raise ValueError("Invalid session timing")
        if row["setup_seconds"] + row["search_seconds"] > row["total_seconds"]:
            raise ValueError("Setup or search lies outside total time")
        grouped[session_key(row)].append(row)
    if set(grouped) != expected_keys(config, manifest):
        raise ValueError("Missing or extra session coverage")
    result = {}
    for key, group in grouped.items():
        if len(group) != config["repetitions"] or {r["repetition"] for r in group} != set(
            range(config["repetitions"])
        ):
            raise ValueError("Missing or duplicate repetitions")
        if len({r["result_sha256"] for r in group}) != 1:
            raise ValueError("Repeated paths or counts changed")
        if len({r["preprocess_expanded"] for r in group}) != 1:
            raise ValueError("Repeated preprocessing counts changed")
        collapsed = {k: v for k, v in group[0].items() if k != "repetition"}
        collapsed.update({field: median(r[field] for r in group) for field in FIELDS})
        result[key] = collapsed
    for layout, size, name in result:
        if (
            name == "manhattan"
            and result[layout, size, name]["result_sha256"]
            != result[layout, size, "prepared_manhattan"]["result_sha256"]
        ):
            raise ValueError("Prepared Manhattan changed paths or counts")
    return result


def verify_queries(path, config, manifest):
    expected = expected_keys(config, manifest)
    splits = {r["layout_id"]: r["split"] for r in manifest}
    outcomes, endpoints = {}, {}
    query_count = 0
    for key, iterator in groupby(rows(path), key=session_key):
        if key not in expected or key in outcomes:
            raise ValueError("Missing, extra, or duplicate query group")
        group = list(iterator)
        layout, size, name = key
        if len(group) != size or [r["query"] for r in group] != list(range(size)):
            raise ValueError("Missing or duplicate queries")
        for row in group:
            if row["split"] != splits[layout]:
                raise ValueError("Query split changed")
            qkey = (layout, row["query"])
            endpoint = (tuple(row["start"]), tuple(row["goal"]), row["optimal_cost"])
            if qkey in endpoints and endpoints[qkey] != endpoint:
                raise ValueError("Methods or batch sizes use different queries")
            endpoints[qkey] = endpoint
            cost = row["optimal_cost"] if row["optimal_cost"] >= 0 else None
            if row["cost"] != cost or row["status"] != ("found" if cost is not None else "no_path"):
                raise ValueError("Incorrect path cost or reachability")
            if name == "bfs_policy":
                if row["expanded"] is not None or row["path_steps"] != (cost or 0):
                    raise ValueError("BFS work counters changed")
            elif not isinstance(row["expanded"], int) or row["expanded"] <= 0:
                raise ValueError("Invalid expansion count")
        outcomes[key] = [r["expanded"] for r in group]
        query_count += size
    if set(outcomes) != expected:
        raise ValueError("Missing query coverage")
    for layout, size, name in outcomes:
        if (
            name == "manhattan"
            and outcomes[layout, size, name] != outcomes[layout, size, "prepared_manhattan"]
        ):
            raise ValueError("Prepared Manhattan query counts changed")
    for record in manifest:
        points = [endpoints[record["layout_id"], q] for q in range(max(config["batch_sizes"]))]
        if len({p[0] for p in points}) != len(points) or len({p[1] for p in points}) != 1:
            raise ValueError("Starts are not distinct or goal changed")
        if any(p[0] == p[1] for p in points):
            raise ValueError("Start equals goal")
    return outcomes, endpoints, query_count


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    metadata = json.loads((root / "measurement.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != metadata[f"{name}_sha256"]:
            raise ValueError("Measurement checksum mismatch")
    manifest_path = root / "fresh_test/manifest.jsonl"
    if digest(manifest_path) != metadata["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(manifest) != metadata["maps"] or len({r["layout_id"] for r in manifest}) != len(
        manifest
    ):
        raise ValueError("Missing or duplicate layouts")
    for path, checksum in json.loads((root / "fingerprints.json").read_text()).items():
        if digest(path) != checksum:
            raise ValueError(f"Frozen input changed: {path}")
    lookup = collapse_sessions(rows(root / "sessions.jsonl.gz"), config, manifest)
    outcomes, endpoints, count = verify_queries(root / "queries.jsonl.gz", config, manifest)
    if (
        count != metadata["query_records"]
        or len(lookup) * config["repetitions"] != metadata["sessions"]
        or count * config["repetitions"] != metadata["timed_queries"]
    ):
        raise ValueError("Recorded workload counts changed")
    families = {name: [name] for name in CONTROLS} | {
        f"width_{w}": [f"width_{w}_seed_{s}" for s in config["seeds"]] for w in config["widths"]
    }
    # Also retain each seed; the main summaries average seeds inside each map.
    families.update({name: [name] for name in method_map(config) if name not in CONTROLS})
    rng = np.random.default_rng(config["bootstrap_seed"])
    summary = {"metadata": metadata, "groups": {}, "verification": {"passed": True}}
    for split in SPLITS:
        layouts = [r["layout_id"] for r in manifest if r["split"] == split]
        summary["groups"][split] = {}
        for size in config["batch_sizes"]:
            output = {}
            for family, names in families.items():
                metrics = defaultdict(list)
                for layout in layouts:
                    selected = [lookup[layout, size, name] for name in names]
                    total = np.mean([r["total_seconds"] for r in selected])
                    for field in FIELDS:
                        metrics[field.replace("_seconds", "_ms_per_query")].append(
                            np.mean([r[field] for r in selected]) * 1000 / size
                        )
                    for baseline in ("manhattan", "prepared_manhattan", "bfs_policy"):
                        metrics[f"ratio_to_{baseline}"].append(
                            total / lookup[layout, size, baseline]["total_seconds"]
                        )
                    metrics["preprocess_expanded"].append(
                        np.mean([r["preprocess_expanded"] for r in selected])
                    )
                    if family != "bfs_policy":
                        reference = outcomes[layout, size, "manhattan"]
                        reachable = [q for q in range(size) if endpoints[layout, q][2] >= 0]
                        metrics["expanded_per_query"].append(
                            np.mean([outcomes[layout, size, name] for name in names])
                        )
                        if reachable:
                            metrics["reachable_expansion_ratio"].append(
                                np.mean(
                                    [
                                        outcomes[layout, size, name][q] / reference[q]
                                        for name in names
                                        for q in reachable
                                    ]
                                )
                            )
                output[family] = {
                    key: bootstrap_mean(values, rng, config["bootstrap_samples"])
                    if "ratio" in key
                    else distribution(values)
                    for key, values in metrics.items()
                }
            summary["groups"][split][str(size)] = {
                "maps": len(layouts),
                "queries": len(layouts) * size,
                "reachable": sum(
                    endpoints[layout, q][2] >= 0 for layout in layouts for q in range(size)
                ),
                "methods": output,
            }
    summary["verification"].update(
        session_medians=len(lookup),
        checksums_and_frozen_inputs=True,
        paired_queries_and_batch_prefixes=True,
        optimal_costs_and_reachability=True,
        prepared_manhattan_matches_original=True,
        repeated_paths_and_counts=True,
    )
    return summary, list(lookup.values()), config


def figures(summary, config, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    sizes = config["batch_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, split in zip(axes, ("test", "walls"), strict=True):
        for method, color in COLORS.items():
            values = [summary["groups"][split][str(n)]["methods"][method] for n in sizes]
            ax.plot(
                sizes,
                [v["total_ms_per_query"]["mean"] for v in values],
                marker="o",
                color=color,
                label=LABELS[method],
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Starts sharing one map and goal",
            ylabel="Mean total ms per query",
            title=SPLITS[split],
        )
        ax.set_xticks(sizes, [str(n) for n in sizes])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Smaller frozen models · all setup included · three-seed means")
    save(fig, output, "width_reuse_runtime")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, size in zip(axes, (64, 128), strict=True):
        for offset, method in enumerate(("width_16", "width_8", "width_4", "bfs_policy")):
            values = [
                summary["groups"][s][str(size)]["methods"][method]["ratio_to_prepared_manhattan"]
                for s in SPLITS
            ]
            ax.errorbar(
                np.arange(5) + (offset - 1.5) * 0.13,
                [v["mean"] for v in values],
                yerr=[
                    [v["mean"] - v["low"] for v in values],
                    [v["high"] - v["mean"] for v in values],
                ],
                fmt="o",
                capsize=3,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.axhline(1, color="#555555", linestyle="--")
        ax.set_xticks(range(5), SPLITS.values(), rotation=25)
        ax.set(
            title=f"{size} starts per session",
            ylabel="Paired total-time ratio to prepared Manhattan",
        )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Whole-map pointwise 95% intervals · below one favors the compared method")
    save(fig, output, "width_reuse_comparison")


def interval(value):
    return f"{value['mean']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def report(summary, config, destination):
    findings = []
    for size in (64, 128):
        gains = [
            f"{LABELS[method]} on {label}"
            for split, label in SPLITS.items()
            for method in ("width_16", "width_8", "width_4")
            if summary["groups"][split][str(size)]["methods"][method][
                "ratio_to_prepared_manhattan"
            ]["high"]
            < 1
        ]
        findings.append(
            f"At {size} starts, pointwise intervals lie below prepared Manhattan for "
            + "; ".join(gains)
            + "."
            if gains
            else f"At {size} starts, no learned width has a pointwise interval entirely "
            "below prepared Manhattan in any tested condition."
        )
    bfs_wins = sum(
        min(
            LABELS,
            key=lambda name: summary["groups"][split]["64"]["methods"][name]["total_ms_per_query"][
                "mean"
            ],
        )
        == "bfs_policy"
        for split in SPLITS
    )
    wall_128 = summary["groups"]["walls"]["128"]["methods"]
    eight_ratio = wall_128["width_8"]["ratio_to_prepared_manhattan"]
    lines = [
        "# Smaller models with shared predictions: results",
        "",
        "This round follows the [fixed plan](lightweight_reuse.md). It reuses nine existing "
        "route-trained checkpoints across 16, 8, and 4 channels. No weights changed and no "
        "new model was selected. The 300 new maps exclude all 5,900 earlier layout groups. "
        "Each map has one goal and 128 distinct starts; smaller sessions use the same prefixes.",
        "",
        " ".join(findings),
        "",
        f"Reverse BFS with stored next steps has the lowest mean complete time in "
        f"{bfs_wins} of five conditions at 64 starts. The results do not establish a general "
        "learned runtime advantage over either prepared Manhattan or this stronger "
        "classical control.",
        "",
        "The 8-channel model is close to prepared Manhattan on wall maps at 128 starts: "
        f"its paired ratio is {interval(eight_ratio)}. Its mean time is "
        f"{wall_128['width_8']['total_ms_per_query']['mean']:.4f} ms versus "
        f"{wall_128['prepared_manhattan']['total_ms_per_query']['mean']:.4f} ms. "
        "The mean of map-wise ratios and the ratio of overall means use different "
        "weighting, so they can fall on opposite sides of one. The paired interval "
        "does not establish a speedup.",
        "",
        "## Complete time per query",
        "",
        "Values are milliseconds per query, including all preparation and returned paths. "
        "Each session uses the median of five shuffled repetitions on one CPU thread. "
        "Learned rows average three seeds within each map, then average maps. Unreachable "
        "queries remain included. Model loading is outside timing. Prepared Manhattan is "
        "the main baseline because it uses the same search implementation as the models.",
        "",
    ]
    sizes = config["batch_sizes"]
    for split in ("test", "walls"):
        lines += [
            f"### {SPLITS[split]}",
            "",
            "| Method | "
            + " | ".join(f"{n} start" + ("s" if n != 1 else "") for n in sizes)
            + " |",
            "| --- | " + " | ".join("---:" for _ in sizes) + " |",
        ]
        for method, label in LABELS.items():
            values = [
                summary["groups"][split][str(n)]["methods"][method]["total_ms_per_query"]["mean"]
                for n in sizes
            ]
            lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        lines.append("")
    lines += [
        "## Paired comparison with prepared Manhattan",
        "",
        "A ratio below one means less complete time. Brackets are pointwise 95% intervals "
        "from 2,000 whole-map bootstrap samples. The primary workload is 64 starts; 128 was "
        "also fixed before testing. An interval crossing one does not establish an advantage. "
        "The grid contains many comparisons, so isolated gains are exploratory and need "
        "confirmation. Batch sizes share maps and are not independent samples.",
        "",
    ]
    for size in (64, 128):
        lines += [
            f"### {size} starts",
            "",
            "| Condition | CNN 16 | CNN 8 | CNN 4 | BFS policy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for split, label in SPLITS.items():
            methods = summary["groups"][split][str(size)]["methods"]
            lines.append(
                f"| {label} | "
                + " | ".join(
                    interval(methods[m]["ratio_to_prepared_manhattan"])
                    for m in ("width_16", "width_8", "width_4", "bfs_policy")
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Setup, search, and search effort",
        "",
        "The following table uses the primary 64-start workload. Component medians are "
        "computed separately, so setup and search need not sum exactly to total time. "
        "Expansion ratios use reachable queries, pair each query with Manhattan, average "
        "queries and model seeds within a map, and then average maps. This separates the "
        "quality of learned ordering from prediction cost.",
        "",
        "| Condition | Method | Setup ms/query | Search ms/query | Reachable expansion ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for method in ("prepared_manhattan", "width_16", "width_8", "width_4", "bfs_policy"):
            value = summary["groups"][split]["64"]["methods"][method]
            effort = (
                interval(value["reachable_expansion_ratio"])
                if "reachable_expansion_ratio" in value
                else "See preprocessing counts"
            )
            lines.append(
                f"| {SPLITS[split]} | {LABELS[method]} | "
                f"{value['setup_ms_per_query']['mean']:.4f} | "
                f"{value['search_ms_per_query']['mean']:.4f} | {effort} |"
            )
    lines += [
        "",
        "BFS policy work is not zero: it expands cells once during preprocessing "
        "and then follows stored steps for each returned route. Raw session records retain "
        "preprocessing expansions, and query records retain path steps.",
        "",
        "## What to try next",
        "",
        "Smaller predictions and reuse reduce the runtime gap, but the learned search "
        "savings remain limited. The 8-channel model retains useful ordering on random "
        "and wall 32×32 maps, while both smaller widths expand more cells on 64×64 maps. "
        "Repeating the same training for more epochs would not directly lower inference "
        "cost or address the missing larger-map training examples.",
        "",
        "A focused next experiment should first check how much equal-priority ordering "
        "can save using exact labels on existing validation maps. If there is useful "
        "room, train one 8-channel pilot with new difficult and larger training layouts, "
        "keeping the architecture and inference cost fixed. Use validation search time "
        "and optimal paths as the gate before spending time on three seeds and another "
        "fresh test set. Keep current test maps out of training and tuning. This is a "
        "proposed experiment, not a measured improvement.",
        "",
        "## Checks and scope",
        "",
        f"Verified {summary['metadata']['sessions']:,} timed sessions containing "
        f"{summary['metadata']['timed_queries']:,} query executions. All returned paths had "
        "optimal cost and correct reachability; no search limits were reached. Repetitions "
        "returned identical paths and counters. Original and prepared Manhattan agreed "
        "exactly. The archive verifier checks hashes, complete method coverage, timing "
        "repetitions, paired endpoints, and batch prefixes.",
        "An independent standard-library check also recomputed all 19,500 session "
        "medians and 3,600 component and paired-ratio means from the raw records.",
        "",
        "Reuse requires a static map and unchanged goal. These results do not establish "
        "a benefit for changing goals or obstacles. They describe fixed checkpoints on "
        "this CPU workload and are not evidence from another training round. The previous "
        "validation selection remains 4 channels; all widths are reported without choosing "
        "a winner from this test. Per-seed results and all five workloads are in summary.json.",
        "",
        "![Runtime by reuse count](../results/research_v6/width_reuse_runtime.png)",
        "",
        "![Paired comparison with prepared Manhattan]"
        "(../results/research_v6/width_reuse_comparison.png)",
        "",
        "The [evidence archive](../results/research_v6/) retains raw timings, query outcomes, "
        "session medians, provenance, summaries, and PNG/PDF figures.",
        "",
    ]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v6"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v6"))
    parser.add_argument("--report", type=Path, default=Path("docs/lightweight_reuse_results.md"))
    args = parser.parse_args()
    summary, medians, config = analyze(args.run)
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
    figures(summary, config, args.output)
    report(summary, config, args.report)
    print(json.dumps(summary["metadata"], indent=2))


if __name__ == "__main__":
    main()
