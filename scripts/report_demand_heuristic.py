"""Verify the full-field/on-demand experiment and retain paired evidence."""

import argparse
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from report_changing_goals import verify_workloads
from report_landmark_timing import estimate, median_session, rows
from run_demand_heuristic import METHODS, fingerprints, grid_for
from run_landmark_timing import TIMINGS, choose_records, signature

from learned_astar.data import read_manifest
from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import validate_path
from learned_astar.evaluation.summarize import bootstrap_mean
from learned_astar.research.experiment import digest, save_json
from learned_astar.search import SearchStats


def verify_scope(root, config, records):
    scope = json.loads((root / "scope.json").read_text())
    if scope["split"] == "policy_validation":
        if records != choose_records(config):
            raise ValueError("Validation selection changed")
        return
    if scope["split"] != "fresh_test":
        raise ValueError("Unknown evidence scope")
    separation = json.loads((root / "separation.json").read_text())
    excluded = set(separation["excluded_groups"])
    seen = set()
    groups = defaultdict(list)
    for record in records:
        grid = grid_for(record)
        blocked = occupancy(grid)
        group = layout_id(blocked, rotations=True)
        if record["split"] != "test" or group != record["group_id"]:
            raise ValueError("Wrong test layout identity")
        if layout_id(blocked) != record["layout_id"] or group in excluded or group in seen:
            raise ValueError("Test overlap or changed layout")
        seen.add(group)
        groups[record["kind"], grid.width].append(record)
    settings = config["fresh_test"]
    expected = {(kind, size) for kind in ("random", "walls") for size in settings["sizes"]}
    if set(groups) != expected or any(
        len(v) != settings["maps_per_family"] for v in groups.values()
    ):
        raise ValueError("Wrong fresh-test family coverage")
    sources = (
        tomllib.loads(Path(settings["exclusion_config"]).read_text())["sources"]
        + settings["additional_sources"]
    )
    original = {r["group_id"] for source in sources for r in read_manifest(source)}
    if original != excluded or separation["earlier_groups"] != len(original):
        raise ValueError("Exclusion evidence changed")
    if separation["sources"] != {s: digest(Path(s) / "manifest.jsonl") for s in sources}:
        raise ValueError("Earlier manifests changed")


def gate_passed(summary):
    gate = summary["config"]["gate"]
    if not summary["measurement"]["all_correct"] or not summary["verification"]["passed"]:
        return False
    for family in gate["families"]:
        value = summary["families"][family][str(gate["goals"])]["methods"]["landmark_demand"][
            "ratio_to_landmark"
        ]
        if value["mean"] > gate["maximum_mean_ratio_to_eager"] or value["high"] >= 1:
            return False
    return True


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    records = json.loads((root / "policy_manifest.json").read_text())
    verify_scope(root, config, records)
    if fingerprints(config, records) != json.loads((root / "fingerprints.json").read_text()):
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
        if row["score_entries"] < 0:
            raise ValueError("Negative score-cache count")
        grouped[key].append(row)
    if set(grouped) != expected:
        raise ValueError("Missing session coverage")
    medians = {
        key: median_session(values, config["repetitions"]) for key, values in grouped.items()
    }
    for layout, goals in workloads:
        for base in ("manhattan", "landmark"):
            eager, demand = medians[layout, goals, base], medians[layout, goals, base + "_demand"]
            if eager["result_sha256"] != demand["result_sha256"]:
                raise ValueError("Full-field and on-demand paths or counters differ")
            if not 0 < demand["score_entries"] <= eager["score_entries"]:
                raise ValueError("Invalid score-cache size")
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
        "scope": json.loads((root / "scope.json").read_text()),
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
            for p in (
                "scripts/report_demand_heuristic.py",
                "scripts/report_changing_goals.py",
                "scripts/report_landmark_timing.py",
            )
        },
    }
    rng = np.random.default_rng(config["bootstrap_seed"])
    for family in sorted({f"{r['kind']}_{r['width']}" for r in records}):
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
                    metrics["score_entries_per_session"].append(value["score_entries"])
                    for reference in ("manhattan", "manhattan_demand", "landmark", "bfs_policy"):
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
                for reference in ("manhattan", "manhattan_demand", "landmark", "bfs_policy"):
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
    summary["gate_passed"] = gate_passed(summary)
    return summary, list(medians.values())


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(summary["families"], key=lambda x: (int(x.split("_")[1]), x))
    fig, axes = plt.subplots(
        len(families) // 2,
        2,
        figsize=(11, 3.5 * len(families) // 2),
        layout="constrained",
        squeeze=False,
    )
    goals = summary["config"]["goal_counts"]
    for ax, family in zip(axes.flat, families, strict=True):
        for method, label, color in (
            ("landmark", "Full-field landmarks", "#9b74b0"),
            ("landmark_demand", "On-demand landmarks", "#176b91"),
            ("bfs_policy", "Reverse BFS", "#389272"),
        ):
            values = [
                summary["families"][family][str(g)]["methods"][method]["ratio_to_manhattan_demand"]
                for g in goals
            ]
            means = [v["mean"] for v in values]
            ax.errorbar(
                range(len(goals)),
                means,
                yerr=[
                    [v["mean"] - v["low"] for v in values],
                    [v["high"] - v["mean"] for v in values],
                ],
                marker="o",
                capsize=3,
                color=color,
                label=label,
            )
        ax.axhline(1, color="#444444", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(goals)), goals)
        ax.set_xlabel("Distinct goals per 128-query session")
        ax.set_ylabel("Complete time / on-demand Manhattan")
        ax.set_title(family.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"{summary['scope']['split'].replace('_', ' ').title()} — paired map means, 95% intervals"
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"demand_comparison.{extension}", dpi=170)
    plt.close(fig)


def tables(summary, output):
    lines = [
        "# On-demand heuristic measurements",
        "",
        f"Scope: {summary['scope']['split']}. All preparation and path output are included.",
        "Ratios pair whole-map median sessions; intervals are pointwise "
        "95% map bootstrap intervals.",
        "",
        "| Family | Goals | Method | Total ms/query | Time / on-demand Manhattan | "
        "Time / full-field landmarks | Stored score entries/session |",
        "| --- | ---: | --- | ---: | --- | --- | ---: |",
    ]
    for family, groups in summary["families"].items():
        for goals, group in groups.items():
            for method, values in group["methods"].items():
                lines.append(
                    f"| {family} | {goals} | {method} | "
                    f"{values['total_ms_per_query']['mean']:.4f} | "
                    f"{estimate(values['ratio_to_manhattan_demand'])} | "
                    f"{estimate(values['ratio_to_landmark'])} | "
                    f"{values['score_entries_per_session']['mean']:.1f} |"
                )
    (output / "tables.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summary, medians = analyze(args.run)
    save_json(args.run / "summary.json", summary)
    save_json(args.run / "median_sessions.json", medians)
    tables(summary, args.run)
    figures(summary, args.run)
    print(
        json.dumps(
            {
                "scope": summary["scope"],
                "gate_passed": summary["gate_passed"],
                "measurement": summary["measurement"],
                "verification": summary["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
