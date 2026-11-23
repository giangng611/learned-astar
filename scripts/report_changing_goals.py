"""Verify changing-goal evidence and report paired complete session times."""

import argparse
import json
import math
import random
import shutil
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from report_landmark_timing import FAMILY_LABELS, LABELS, estimate, median_session, rows
from run_changing_goals import fingerprints, grid_for
from run_landmark_timing import FAMILIES, METHODS, TIMINGS, choose_records, signature

from learned_astar.environments import validate_path
from learned_astar.evaluation.summarize import bootstrap_mean
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchStats
from learned_astar.search.prepared import PreparedGrid, ReversePolicy


def verify_workloads(records, config, workloads):
    """Regenerate endpoints and verify stored labels using the prepared BFS control."""
    indexed = {(w["layout_id"], w["goal_count"]): w for w in workloads}
    expected = {(r["layout_id"], g) for r in records for g in config["goal_counts"]}
    if len(indexed) != len(workloads) or set(indexed) != expected:
        raise ValueError("Wrong workload coverage")
    grids = {}
    for record in records:
        grid = grids[record["layout_id"]] = grid_for(record)
        prepared = PreparedGrid.build(grid)
        rng = random.Random(record["map_seed"] + config["query_seed"])
        goals = rng.sample(list(grid.free_cells()), max(config["goal_counts"]))
        excluded = set(goals)
        starts = rng.sample(
            [c for c in grid.free_cells() if c not in excluded], config["queries_per_session"]
        )
        fields = [ReversePolicy(prepared, goal).distances for goal in goals]
        for count in config["goal_counts"]:
            work = indexed[record["layout_id"], count]
            expected_queries = [
                {
                    "start": list(start),
                    "goal": list(goals[q % count]),
                    "optimal_cost": fields[q % count][prepared.index(start)],
                }
                for q, start in enumerate(starts)
            ]
            if (
                work["queries"] != expected_queries
                or work["family"] != f"{record['kind']}_{grid.width}"
            ):
                raise ValueError("Endpoint or BFS-label mismatch")
    return indexed, grids


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    records = json.loads((root / "policy_manifest.json").read_text())
    if records != choose_records(config) or fingerprints(config, records) != json.loads(
        (root / "fingerprints.json").read_text()
    ):
        raise ValueError("Frozen inputs changed")
    metadata = json.loads((root / "measurement.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != metadata[f"{name}_sha256"]:
            raise ValueError("Raw checksum mismatch")
    if (
        digest(root / "workloads.json")
        != json.loads((root / "workloads_verified.json").read_text())["sha256"]
    ):
        raise ValueError("Workload checksum mismatch")
    workloads, grids = verify_workloads(
        records, config, json.loads((root / "workloads.json").read_text())
    )
    expected = {(layout, goals, name) for layout, goals in workloads for name in METHODS}
    grouped = defaultdict(list)
    for row in rows(root / "sessions.jsonl.gz"):
        key = row["layout_id"], row["goal_count"], row["method"]
        if key not in expected or row["family"] != workloads[key[:2]]["family"]:
            raise ValueError("Unexpected session")
        if row["goals_prepared"] != row["goal_count"] or row["other_seconds"] < 0:
            raise ValueError("Invalid goal cache or overhead")
        if not math.isclose(
            row["setup_seconds"] + row["search_seconds"] + row["other_seconds"],
            row["total_seconds"],
            abs_tol=1e-12,
        ):
            raise ValueError("Unaccounted time")
        grouped[key].append(row)
    if set(grouped) != expected:
        raise ValueError("Missing session coverage")
    medians = {
        key: median_session(values, config["repetitions"]) for key, values in grouped.items()
    }
    outcomes = defaultdict(dict)
    n = config["queries_per_session"]
    for row in rows(root / "queries.jsonl.gz"):
        key = row["layout_id"], row["goal_count"], row["method"]
        q = row["query"]
        if key not in expected or not 0 <= q < n or q in outcomes[key]:
            raise ValueError("Unexpected or duplicate query")
        truth = workloads[key[:2]]["queries"][q]
        if any(row[k] != truth[k] for k in ("start", "goal", "optimal_cost")):
            raise ValueError("Query differs from frozen workload")
        if row["family"] != workloads[key[:2]]["family"]:
            raise ValueError("Incorrect family")
        if truth["optimal_cost"] >= 0:
            if row["status"] != "found" or row["cost"] != truth["optimal_cost"]:
                raise ValueError("Wrong optimal cost")
            validate_path(
                grids[key[0]],
                tuple(map(tuple, row["path"])),
                tuple(row["start"]),
                tuple(row["goal"]),
                row["cost"],
            )
        elif row["status"] != "no_path" or row["cost"] is not None or row["path"]:
            raise ValueError("Wrong reachability")
        if (row["stats"] is None) != (key[2] == "bfs_policy"):
            raise ValueError("Wrong statistics type")
        outcomes[key][q] = row
        if len(outcomes[key]) == n:
            results = [
                {
                    "status": r["status"],
                    "cost": r["cost"],
                    "path": r["path"],
                    "stats": SearchStats(**r["stats"]) if r["stats"] else None,
                }
                for r in (outcomes[key][j] for j in range(n))
            ]
            if signature(results) != medians[key]["result_sha256"]:
                raise ValueError("Saved paths or counters differ from measured sessions")
            # Full paths stay in the archive; discard them after verification to bound memory.
            for saved in outcomes[key].values():
                del saved["path"]
    if set(outcomes) != expected or any(len(v) != n for v in outcomes.values()):
        raise ValueError("Missing query coverage")
    if (metadata["sessions"], metadata["stored_queries"], metadata["timed_queries"]) != (
        len(expected) * config["repetitions"],
        len(expected) * n,
        len(expected) * n * config["repetitions"],
    ):
        raise ValueError("Wrong workload totals")
    summary = {
        "config": config,
        "measurement": metadata,
        "families": {},
        "verification": {
            "passed": True,
            "median_sessions": len(medians),
            "labels_independently_recomputed": True,
            "all_paths_and_signatures_verified": True,
        },
        "analysis_sha256": {
            p: digest(p)
            for p in ("scripts/report_changing_goals.py", "scripts/report_landmark_timing.py")
        },
    }
    rng = np.random.default_rng(config["bootstrap_seed"])
    for family in FAMILIES:
        layouts = sorted(r["layout_id"] for r in records if f"{r['kind']}_{r['width']}" == family)
        summary["families"][family] = {}
        for goals in config["goal_counts"]:
            group = {
                "queries": len(layouts) * n,
                "unreachable": sum(
                    q["optimal_cost"] < 0
                    for layout in layouts
                    for q in workloads[layout, goals]["queries"]
                ),
                "methods": {},
            }
            for name in METHODS:
                metrics = defaultdict(list)
                for layout in layouts:
                    value = medians[layout, goals, name]
                    for field in (*TIMINGS, "other_seconds"):
                        metrics[field.replace("_seconds", "_ms_per_query")].append(
                            value[field] * 1000 / n
                        )
                    metrics["preprocess_visits_per_session"].append(value["preprocess_expanded"])
                    for reference in ("manhattan", "landmark", "bfs_policy"):
                        metrics[f"ratio_to_{reference}"].append(
                            value["total_seconds"]
                            / medians[layout, goals, reference]["total_seconds"]
                        )
                    if name != "bfs_policy":
                        reachable = [
                            q
                            for q in range(n)
                            if outcomes[layout, goals, name][q]["optimal_cost"] > 0
                        ]
                        for reference in ("manhattan", "landmark"):
                            if reachable:
                                metrics[f"expansion_ratio_to_{reference}"].append(
                                    np.mean(
                                        [
                                            outcomes[layout, goals, name][q]["stats"]["expanded"]
                                            / outcomes[layout, goals, reference][q]["stats"][
                                                "expanded"
                                            ]
                                            for q in reachable
                                        ]
                                    )
                                )
                group["methods"][name] = {
                    metric: bootstrap_mean(v, rng, config["bootstrap_samples"])
                    for metric, v in metrics.items()
                }
            summary["families"][family][str(goals)] = group
    # Independent point-estimate check using standard-library medians and means.
    checked = 0
    for family, groups in summary["families"].items():
        for goals, group in groups.items():
            for name, metrics in group["methods"].items():
                keys = [
                    k
                    for k in grouped
                    if k[1:] == (int(goals), name) and grouped[k][0]["family"] == family
                ]
                totals = {
                    k: statistics.median(r["total_seconds"] for r in grouped[k]) for k in keys
                }
                assert math.isclose(
                    statistics.fmean(totals.values()) * 1000 / n,
                    metrics["total_ms_per_query"]["mean"],
                    rel_tol=1e-12,
                )
                checked += 1
                for reference in ("manhattan", "landmark", "bfs_policy"):
                    ratios = [
                        v
                        / statistics.median(
                            r["total_seconds"] for r in grouped[k[0], k[1], reference]
                        )
                        for k, v in totals.items()
                    ]
                    assert math.isclose(
                        statistics.fmean(ratios),
                        metrics[f"ratio_to_{reference}"]["mean"],
                        rel_tol=1e-12,
                    )
                    checked += 1
    summary["verification"]["point_estimates_cross_checked"] = checked
    return summary, list(medians.values())


def report(summary, destination):
    measurement = summary["measurement"]
    lines = [
        "# Step 3: changing-goal results",
        "",
        "This [fixed validation pilot](changing_goals.md) reuses the same 40 maps and frozen "
        "8-channel seed-7 model. Every session has 128 queries, with 1, 8, or 128 distinct "
        "goals. No model was trained and no fresh test set was used.",
        "",
        "With 128 distinct goals, landmark-only search is slower than Manhattan in all "
        "four families. The CNN adds further cost. Manhattan has the lowest mean time "
        "on the random families, while reverse BFS has the lowest mean on walls. "
        "Reusing the landmark index does not cover the cost of constructing full "
        "goal-specific score fields in this implementation.",
        "",
        "## Complete query time",
        "",
        "Values are mean milliseconds per query after taking each map's median session. "
        "They include map preparation, all goal-cache misses, search, path output, and loop "
        "overhead. The model is resident. Lower is better.",
        "",
        "| Condition | Goals | Manhattan | Manhattan + CNN | Landmarks | "
        "Landmarks + CNN | BFS policy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        for goals in summary["config"]["goal_counts"]:
            methods = summary["families"][family][str(goals)]["methods"]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {goals} | "
                + " | ".join(
                    f"{methods[name]['total_ms_per_query']['mean']:.4f}" for name in METHODS
                )
                + " |"
            )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Ratios are computed per map, then averaged equally over 10 maps per family. "
        "Brackets are pointwise 95% whole-map bootstrap intervals. Below one favors the "
        "numerator; an interval crossing one does not establish a difference. These ratios "
        "can differ from dividing the overall mean latencies above.",
        "",
        "| Condition | Goals | Landmarks / Manhattan | BFS / Manhattan | "
        "Landmarks + CNN / landmarks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        for goals in summary["config"]["goal_counts"]:
            m = summary["families"][family][str(goals)]["methods"]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {goals} | "
                f"{estimate(m['landmark']['ratio_to_manhattan'])} | "
                f"{estimate(m['bfs_policy']['ratio_to_manhattan'])} | "
                f"{estimate(m['landmark_cnn']['ratio_to_landmark'])} |"
            )
    lines += [
        "",
        "## What happens when every query has a new goal?",
        "",
        "This breakdown uses 128 goals. All values are milliseconds per query. Setup includes "
        "the graph, landmark index, goal score fields, predictions, and BFS preparation where "
        "applicable. Other time includes cache lookups and loop overhead. Components come "
        "from the same repetition as the median total.",
        "",
        "| Condition | Method | Setup | Of which CNN | Search/output | Other | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in FAMILIES:
        for name in METHODS:
            row = summary["families"][family]["128"]["methods"][name]
            numbers = [
                row[f"{k}_ms_per_query"]["mean"]
                for k in ("setup", "prediction", "search", "other", "total")
            ]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {LABELS[name]} | "
                + " | ".join(f"{v:.4f}" for v in numbers)
                + " |"
            )
    lines += [
        "",
        "## Correctness and scope",
        "",
        f"The run completed {measurement['sessions']:,} sessions and "
        f"{measurement['timed_queries']:,} query executions in "
        f"{measurement['wall_seconds']:.1f} seconds of measurement-stage wall time, including "
        "checking and saving evidence. All methods returned optimal costs or correct no-path "
        "results. The archive retains 76,800 full query outcomes and 600 median sessions. "
        "Every repeated path and search counter matched.",
        "",
        "Verification regenerated the deterministic endpoints, recomputed labels through a "
        "separate prepared-BFS implementation, checked every saved path, and reconstructed "
        "each session signature. It also checked all frozen hashes and cross-checked 240 "
        "latency and paired-ratio means with standard-library calculations.",
        "",
        f"One observed model load took {measurement['model_load_seconds'] * 1000:.2f} ms. "
        "That one-time cost is reported separately from resident-model session times. Divide "
        "it by 128 to obtain the extra per-query charge for a first session in an initialized "
        "process. Python/library startup, dataset I/O, warmup, and evidence verification "
        "are outside all method timers; this is not a cold-start measurement.",
        "",
        "The changing-goal workload also changes query endpoints and difficulty. Method "
        "comparisons within a workload are paired, but differences across goal counts should "
        "not be attributed solely to cache reuse. These are small validation results from "
        "one model seed, not held-out generalization claims. The implementation builds full "
        "goal-specific score fields; more efficient on-demand landmark implementations were "
        "not tested. Goal caches also use memory, whose peak usage was not benchmarked.",
        "",
        "The one- and eight-goal workloads still favor landmark-only over Manhattan "
        "on walls, but reverse BFS is faster there. When every goal is new, the "
        "landmark index itself is still built only once; most setup time moves to "
        "goal-specific score construction. The CNN likewise needs a new prediction "
        "per goal. These measurements identify implementation costs, not an inherent "
        "limit of all landmark heuristics or learned search methods.",
        "",
        "Given the remaining project week, stop expanding this research branch. "
        "Retain the negative model result and the classical wall-map improvement "
        "as separate findings. Prioritize a clear final report, an updated presentation, "
        "and a reproducible terminal demonstration. On-demand landmark scoring can be "
        "listed as future work without claiming an unmeasured speedup.",
        "",
        "![Changing-goal comparison](../results/primary_step3/changing_goals.png)",
        "",
        "The [archive](../results/primary_step3/) contains settings, selected maps, query "
        "labels, source hashes, raw sessions and paths, medians, and complete summaries.",
        "",
    ]
    destination.write_text("\n".join(lines))


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    goals = summary["config"]["goal_counts"]
    names = ("manhattan_cnn", "landmark", "landmark_cnn", "bfs_policy")
    colors = ("#c76e23", "#176b91", "#8764a8", "#389272")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, family in zip(axes.flat, FAMILIES, strict=True):
        for name, color in zip(names, colors, strict=True):
            values = [
                summary["families"][family][str(g)]["methods"][name]["ratio_to_manhattan"]
                for g in goals
            ]
            ax.plot(goals, [v["mean"] for v in values], marker="o", color=color, label=LABELS[name])
            ax.fill_between(
                goals,
                [v["low"] for v in values],
                [v["high"] for v in values],
                color=color,
                alpha=0.12,
            )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set(
            xscale="log",
            yscale="log",
            title=FAMILY_LABELS[family],
            xlabel="Distinct goals per 128 queries",
            ylabel="Complete time / Manhattan",
        )
        ax.set_xticks(goals, list(map(str, goals)))
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Changing goals · policy validation · all preparation charged · lower is better")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"changing_goals.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/primary_step3"))
    parser.add_argument("--output", type=Path, default=Path("results/primary_step3"))
    args = parser.parse_args()
    summary, medians = analyze(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "protocol.toml",
        "policy_manifest.json",
        "fingerprints.json",
        "workloads.json",
        "workloads_verified.json",
        "measurement.json",
        "sessions.jsonl.gz",
        "queries.jsonl.gz",
    ):
        shutil.copy2(args.run / name, args.output / name)
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "median_sessions.json", medians)
    figures(summary, args.output)
    report(summary, Path("docs/changing_goals_results.md"))
    print(json.dumps(summary["verification"]))


if __name__ == "__main__":
    main()
