"""Verify complete landmark sessions and summarize paired validation measurements."""

import argparse
import gzip
import json
import math
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_landmark_timing import (
    COMPONENTS,
    FAMILIES,
    METHODS,
    TIMINGS,
    choose_records,
    fingerprints,
    query_map,
    signature,
)

from learned_astar.environments import validate_path
from learned_astar.evaluation.summarize import bootstrap_mean
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchStats

LABELS = {
    "manhattan": "Manhattan",
    "manhattan_cnn": "Manhattan + CNN",
    "landmark": "Landmarks",
    "landmark_cnn": "Landmarks + CNN",
    "bfs_policy": "BFS policy",
}
FAMILY_LABELS = dict(zip(FAMILIES, ("Random 32", "Walls 32", "Random 64", "Walls 64"), strict=True))


def rows(path):
    with gzip.open(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def median_session(values, repetitions):
    if len(values) != repetitions or {r["repetition"] for r in values} != set(range(repetitions)):
        raise ValueError("Missing or duplicate repetitions")
    if len({r["result_sha256"] for r in values}) != 1:
        raise ValueError("Repeated results disagree")
    if len({r["preprocess_expanded"] for r in values}) != 1:
        raise ValueError("Preparation counts disagree")
    for row in values:
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in TIMINGS):
            raise ValueError("Invalid timer")
        if (
            row["total_seconds"] <= 0
            or row["setup_seconds"] + row["search_seconds"] > row["total_seconds"] + 1e-12
        ):
            raise ValueError("Total time excludes work")
        if sum(row[k] for k in COMPONENTS) > row["setup_seconds"] + 1e-12:
            raise ValueError("Setup time excludes work")
    # Odd repetitions: retain one real median session, including its additive components.
    return sorted(values, key=lambda r: (r["total_seconds"], r["repetition"]))[repetitions // 2]


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    records = json.loads((root / "policy_manifest.json").read_text())
    if records != choose_records(config) or fingerprints(config, records) != json.loads(
        (root / "fingerprints.json").read_text()
    ):
        raise ValueError("Frozen inputs changed")
    measurement = json.loads((root / "measurement.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != measurement[f"{name}_sha256"]:
            raise ValueError("Raw evidence checksum mismatch")
    maximum = max(config["batch_sizes"])
    dataset = {}
    for record in records:
        grid, goal, starts, labels = query_map(Path(record["directory"]), record, maximum)
        dataset[record["layout_id"]] = (
            f"{record['kind']}_{grid.width}",
            grid,
            goal,
            starts,
            labels,
        )
    expected = {
        (layout, size, name)
        for layout in dataset
        for size in config["batch_sizes"]
        for name in METHODS
    }
    grouped = defaultdict(list)
    for row in rows(root / "sessions.jsonl.gz"):
        key = row["layout_id"], row["batch_size"], row["method"]
        if key not in expected or row["family"] != dataset[key[0]][0]:
            raise ValueError("Unexpected session")
        grouped[key].append(row)
    if set(grouped) != expected:
        raise ValueError("Missing session coverage")
    medians = {
        key: median_session(values, config["repetitions"]) for key, values in grouped.items()
    }
    outcomes = defaultdict(dict)
    for row in rows(root / "queries.jsonl.gz"):
        layout, name, q = row["layout_id"], row["method"], row["query"]
        if layout not in dataset or name not in METHODS or not 0 <= q < maximum:
            raise ValueError("Unexpected query")
        family, grid, goal, starts, labels = dataset[layout]
        if q in outcomes[layout, name] or row["batch_size"] != maximum or row["family"] != family:
            raise ValueError("Duplicate or mismatched query")
        if (
            row["start"] != list(starts[q])
            or row["goal"] != list(goal)
            or row["optimal_cost"] != int(labels[starts[q]])
        ):
            raise ValueError("Query endpoints disagree with dataset")
        cost = int(labels[starts[q]])
        if cost >= 0:
            if row["status"] != "found" or row["cost"] != cost:
                raise ValueError("Wrong optimal cost")
            validate_path(grid, tuple(map(tuple, row["path"])), starts[q], goal, cost)
        elif row["status"] != "no_path" or row["cost"] is not None or row["path"]:
            raise ValueError("Wrong reachability")
        if (row["stats"] is None) != (name == "bfs_policy"):
            raise ValueError("Wrong counter type")
        outcomes[layout, name][q] = row
    if set(outcomes) != {(layout, name) for layout in dataset for name in METHODS} or any(
        len(v) != maximum for v in outcomes.values()
    ):
        raise ValueError("Incomplete query evidence")
    for (layout, size, name), median in medians.items():
        prefix = [outcomes[layout, name][q] for q in range(size)]
        reconstructed = [
            {
                "status": r["status"],
                "cost": r["cost"],
                "path": r["path"],
                "stats": SearchStats(**r["stats"]) if r["stats"] else None,
            }
            for r in prefix
        ]
        if signature(reconstructed) != median["result_sha256"]:
            raise ValueError("Stored paths/counters do not match a session prefix")
    session_count = len(expected) * config["repetitions"]
    timed_count = len(dataset) * len(METHODS) * sum(config["batch_sizes"]) * config["repetitions"]
    if (measurement["sessions"], measurement["stored_queries"], measurement["timed_queries"]) != (
        session_count,
        len(outcomes) * maximum,
        timed_count,
    ):
        raise ValueError("Workload totals disagree")
    rng = np.random.default_rng(config["bootstrap_seed"])
    summary = {
        "measurement": measurement,
        "config": config,
        "families": {},
        "verification": {
            "passed": True,
            "session_medians": len(medians),
            "paths_and_prefixes_checked": True,
            "frozen_inputs_checked": True,
        },
    }
    for family in FAMILIES:
        layouts = sorted(layout for layout, value in dataset.items() if value[0] == family)
        summary["families"][family] = {}
        for size in config["batch_sizes"]:
            result = {
                "unique_queries": len(layouts) * size,
                "unreachable_queries": sum(
                    int(dataset[layout][4][s]) < 0
                    for layout in layouts
                    for s in dataset[layout][3][:size]
                ),
                "methods": {},
            }
            for name in METHODS:
                values = defaultdict(list)
                for layout in layouts:
                    row = medians[layout, size, name]
                    for key in TIMINGS:
                        values[key.replace("_seconds", "_ms_per_query")].append(
                            row[key] * 1000 / size
                        )
                    values["preprocess_visits_per_session"].append(row["preprocess_expanded"])
                    if name != "bfs_policy":
                        queries = [outcomes[layout, name][q] for q in range(size)]
                        values["search_expansions_per_query"].append(
                            np.mean([r["stats"]["expanded"] for r in queries])
                        )
                    for ref in ("manhattan", "landmark", "bfs_policy"):
                        values[f"ratio_to_{ref}"].append(
                            row["total_seconds"] / medians[layout, size, ref]["total_seconds"]
                        )
                    if name != "bfs_policy":
                        reachable = [
                            q for q in range(size) if outcomes[layout, name][q]["optimal_cost"] > 0
                        ]
                        for ref in ("manhattan", "landmark"):
                            if reachable:
                                values[f"expansion_ratio_to_{ref}"].append(
                                    np.mean(
                                        [
                                            outcomes[layout, name][q]["stats"]["expanded"]
                                            / outcomes[layout, ref][q]["stats"]["expanded"]
                                            for q in reachable
                                        ]
                                    )
                                )
                result["methods"][name] = {
                    metric: bootstrap_mean(v, rng, config["bootstrap_samples"])
                    for metric, v in values.items()
                }
            summary["families"][family][str(size)] = result
    return summary, list(medians.values())


def estimate(value):
    return f"{value['mean']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def report(summary, destination):
    measurement = summary["measurement"]
    lines = [
        "# Step 2: complete-time landmark results",
        "",
        "This is the [fixed validation pilot](landmark_timing.md): 40 existing maps, one frozen "
        "8-channel model seed, four session sizes, and seven repetitions. "
        "No training or fresh-test "
        "selection took place. All methods returned optimal costs and correct reachability.",
        "",
        "Landmark-only search wins on the measured wall families once enough starts "
        "share a goal, but loses on the random families. The frozen CNN makes landmarks "
        "slower in all four 128-start comparisons. Reverse BFS has the lowest mean "
        "128-start time in every family. The improvement over Manhattan therefore "
        "belongs to the classical landmark estimate, not the model.",
        "",
        "## Complete time",
        "",
        "Values are mean milliseconds per query after taking each map's median session. "
        "All map/goal setup is charged once per session, including the four landmark BFS runs "
        "and CNN prediction when used. Lower is better.",
        "",
    ]
    for size in (1, 128):
        lines += [
            f"### {size} start(s) sharing one goal",
            "",
            "| Method | Random 32 | Walls 32 | Random 64 | Walls 64 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in METHODS:
            values = [
                summary["families"][f][str(size)]["methods"][name]["total_ms_per_query"]["mean"]
                for f in FAMILIES
            ]
            lines.append(f"| {LABELS[name]} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        lines.append("")
    lines += [
        "## Does preparation pay off?",
        "",
        "These are paired landmark-only / Manhattan time ratios, averaged equally over maps. "
        "A value below one favors landmarks. Brackets show pointwise 95% whole-map bootstrap "
        "intervals, using 10 maps in each family. "
        "An interval crossing one does not establish a win.",
        "",
        "| Starts | Random 32 | Walls 32 | Random 64 | Walls 64 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for size in summary["config"]["batch_sizes"]:
        lines.append(
            f"| {size} | "
            + " | ".join(
                estimate(
                    summary["families"][f][str(size)]["methods"]["landmark"]["ratio_to_manhattan"]
                )
                for f in FAMILIES
            )
            + " |"
        )
    lines += [
        "",
        "## Does the CNN add value?",
        "",
        "At 128 starts, compare landmark + CNN directly with landmark-only. This isolates "
        "the added prediction cost and tie ordering. Expansion ratios use reachable queries, "
        "averaged within each map first; timing includes unreachable queries too.",
        "",
        "| Condition | Landmark time / Manhattan | CNN addition: time / landmarks | "
        "Landmark expansions / Manhattan | CNN addition: expansions / landmarks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        methods = summary["families"][family]["128"]["methods"]
        base, learned = methods["landmark"], methods["landmark_cnn"]
        lines.append(
            f"| {FAMILY_LABELS[family]} | {estimate(base['ratio_to_manhattan'])} | "
            f"{estimate(learned['ratio_to_landmark'])} | "
            f"{estimate(base['expansion_ratio_to_manhattan'])} | "
            f"{estimate(learned['expansion_ratio_to_landmark'])} |"
        )
    lines += [
        "",
        "## Where the time goes",
        "",
        "The 128-start breakdown below is in milliseconds per query. Each component comes "
        "from the same repetition as the median total. Setup includes graph creation, landmark "
        "or reverse BFS preparation, prediction, and score/planner construction. Small timer "
        "overhead can make setup plus search differ slightly from total.",
        "",
        "| Condition | Method | Setup | Of which CNN | Search/path output | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        for name in METHODS:
            row = summary["families"][family]["128"]["methods"][name]
            numbers = [
                row[f"{key}_ms_per_query"]["mean"]
                for key in ("setup", "prediction", "search", "total")
            ]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {LABELS[name]} | "
                + " | ".join(f"{v:.4f}" for v in numbers)
                + " |"
            )
    lines += [
        "",
        "## Scope and retained evidence",
        "",
        f"The run completed {measurement['sessions']:,} sessions "
        f"and {measurement['timed_queries']:,} "
        f"query executions in {measurement['wall_seconds']:.1f} seconds of measurement-stage wall "
        "time, including checks and writing evidence. This wall time is not a method latency.",
        "",
        f"Loading the model weights once took {measurement['model_load_seconds'] * 1000:.2f} ms. "
        "This is separate from session time, which assumes resident weights. For a first session "
        "in an initialized process, add that load cost to the session total, or divide it by the "
        "number of starts before adding it to the per-query value. This single load observation "
        "is not a cold-start benchmark; Python/library startup and disk-cache conditions were "
        "not controlled. Dataset loading, warmup, result verification, and saving are outside "
        "all method timers.",
        "",
        "All full paths were checked against recomputed BFS labels. Saved paths and counters "
        "reconstruct every session signature, including shorter query prefixes and all seven "
        "repetitions. Source, model, map-array, and raw-file hashes were checked. Unreachable "
        "counts, all timing components, and every batch size are in summary.json.",
        "",
        "These are small validation results, not held-out generalization claims. Confidence "
        "intervals describe variation over the sampled maps and are not simultaneous intervals "
        "for every comparison. Four-landmark selection is fixed, not tuned here. The workload "
        "shares a map and goal; it does not establish performance with changing goals or free "
        "offline preprocessing.",
        "",
        "The 128-start workload contains 5,120 distinct map/start/goal queries. "
        "There are 10 unreachable queries in random 32 and six in random 64; all wall "
        "queries are reachable. Shorter workloads use prefixes of these same starts. "
        "The baseline is the equally prepared Manhattan implementation, so these "
        "numbers should not be read as timings for every possible A* implementation. "
        "Absolute latencies should be compared within this run, not across earlier rounds.",
        "",
        "The first measured batch size with an interval wholly below Manhattan is "
        "64 starts for walls 32 and 16 starts for walls 64. These are sampled workload "
        "sizes, not exact break-even thresholds. No such advantage appears for the "
        "random families through 128 starts. Ratios are averaged per map; dividing "
        "the two overall mean latencies can give a different ratio.",
        "",
        "The next research question should address a setting where preprocessing "
        "cannot simply solve the shared-goal problem. Reusing one map across changing "
        "goals could be examined separately. More training of the current tie model "
        "is not justified by this pilot alone. A fresh test and more model seeds "
        "would be needed before a general model-performance claim.",
        "",
        "![Landmark complete-time comparison](../results/primary_step2/landmark_timing.png)",
        "",
        "The [evidence archive](../results/primary_step2/) retains the protocol, selected maps, "
        "source hashes, raw sessions, full query paths, median sessions, and summaries.",
        "",
    ]
    destination.write_text("\n".join(lines))


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = summary["config"]["batch_sizes"]
    colors = ("#176b91", "#c76e23", "#389272", "#8764a8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for family, color in zip(FAMILIES, colors, strict=True):
        values = [
            summary["families"][family][str(s)]["methods"]["landmark"]["ratio_to_manhattan"]
            for s in sizes
        ]
        means = np.array([v["mean"] for v in values])
        axes[0].plot(sizes, means, marker="o", color=color, label=FAMILY_LABELS[family])
        axes[0].fill_between(
            sizes, [v["low"] for v in values], [v["high"] for v in values], alpha=0.12, color=color
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Starts sharing one goal",
        ylabel="Complete time / Manhattan",
        title="Do landmarks repay preparation?",
    )
    axes[0].set_xticks(sizes, [str(s) for s in sizes])
    axes[0].legend(frameon=False)
    for i, (family, color) in enumerate(zip(FAMILIES, colors, strict=True)):
        value = summary["families"][family]["128"]["methods"]["landmark_cnn"]["ratio_to_landmark"]
        axes[1].errorbar(
            i,
            value["mean"],
            yerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]],
            fmt="o",
            color=color,
            capsize=4,
        )
    axes[1].set_xticks(range(4), list(FAMILY_LABELS.values()), rotation=15)
    axes[1].set(
        ylabel="Complete time: landmarks + CNN / landmarks",
        title="Does the frozen CNN add value? · 128 starts",
    )
    for ax in axes:
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Policy validation · 40 maps · setup included · lower ratios are better")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"landmark_timing.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/primary_step2"))
    parser.add_argument("--output", type=Path, default=Path("results/primary_step2"))
    args = parser.parse_args()
    summary, medians = analyze(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "protocol.toml",
        "policy_manifest.json",
        "fingerprints.json",
        "measurement.json",
        "sessions.jsonl.gz",
        "queries.jsonl.gz",
    ):
        shutil.copy2(args.run / name, args.output / name)
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "median_sessions.json", medians)
    figures(summary, args.output)
    report(summary, Path("docs/landmark_timing_results.md"))
    print(json.dumps(summary["verification"]))


if __name__ == "__main__":
    main()
