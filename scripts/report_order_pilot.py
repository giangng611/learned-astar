"""Verify ordering diagnostics and apply the predeclared validation gate."""

import argparse
import gzip
import json
import shutil
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_order_pilot import FAMILIES, fingerprints, policy_records, query_map

from learned_astar.evaluation.summarize import bootstrap_mean
from learned_astar.research.experiment import digest, save_json


def rows(path):
    with gzip.open(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def summarize_headroom(root, config):
    verified = json.loads((root / "headroom_verified.json").read_text())
    if verified["sha256"] != digest(root / "headroom.jsonl.gz"):
        raise ValueError("Headroom checksum mismatch")
    raw = list(rows(root / "headroom.jsonl.gz"))
    expected = {
        (r["layout_id"], q)
        for _, r in policy_records(root, config)
        for q in range(config["headroom_starts"])
    }
    if len(raw) != len(expected) or {(r["layout_id"], r["query"]) for r in raw} != expected:
        raise ValueError("Headroom coverage mismatch")
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in raw:
        if r["floor"] != r["expanded"]["attained_floor"]:
            raise ValueError("Unattained expansion floor")
        if r["optimal_cost"] <= 0:
            continue
        baseline = r["expanded"]["manhattan"]
        for name, expanded in r["expanded"].items():
            values[r["family"]][name][r["layout_id"]].append(expanded / baseline)
        values[r["family"]]["mandatory_fraction"][r["layout_id"]].append(r["mandatory"] / baseline)
    rng = np.random.default_rng(config["bootstrap_seed"])
    return {
        "queries": len(raw),
        "reachable": sum(r["optimal_cost"] >= 0 for r in raw),
        "exact_detour_attains_floor": all(r["expanded"]["exact_detour"] == r["floor"] for r in raw),
        "families": {
            family: {
                name: bootstrap_mean(
                    [np.mean(v) for v in maps.values()], rng, config["bootstrap_samples"]
                )
                for name, maps in methods.items()
            }
            for family, methods in values.items()
        },
    }, raw


def apply_gate(summary, config):
    candidates = []
    for rule in ("distance", "detour"):
        name = f"candidate_{rule}"
        overall = summary["overall"][name]
        groups = {
            f: summary["families"][f][str(config["primary_batch_size"])][name] for f in FAMILIES
        }
        checks = {
            "all_paths_correct": summary["measurement"]["all_correct"],
            "mean_gain_at_least_two_percent": overall["ratio_to_frozen_distance"]["mean"]
            <= config["gate"]["maximum_mean_time_ratio"],
            "runtime_interval_below_frozen": overall["ratio_to_frozen_distance"]["high"] < 1,
            "runtime_interval_below_matched_control": overall[f"ratio_to_control_{rule}"]["high"]
            < 1,
            "no_family_runtime_regression_over_three_percent": all(
                g["ratio_to_frozen_distance"]["mean"] <= config["gate"]["maximum_group_regression"]
                for g in groups.values()
            ),
            "walls_32_interval_below_manhattan": groups["walls_32"]["ratio_to_prepared_manhattan"][
                "high"
            ]
            < 1,
            "no_32_expansion_regression_over_three_percent": all(
                groups[f]["expansion_ratio_to_frozen"]["mean"]
                <= config["gate"]["maximum_group_regression"]
                for f in ("random_32", "walls_32")
            ),
        }
        candidates.append(
            {
                "rule": rule,
                "checks": checks,
                "eligible": all(checks.values()),
                "runtime_ratio": overall["ratio_to_frozen_distance"]["mean"],
            }
        )
    eligible = [c for c in candidates if c["eligible"]]
    return {
        "expand_to_more_seeds": bool(eligible),
        "selected_rule": min(eligible, key=lambda c: c["runtime_ratio"])["rule"]
        if eligible
        else None,
        "candidates": candidates,
        "evidence": "policy validation only; no new test results",
    }


def analyze(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    if fingerprints(config) != json.loads((root / "fingerprints.json").read_text()):
        raise ValueError("Frozen source changed")
    diagnostic, audit = summarize_headroom(root, config)
    metadata = json.loads((root / "measurement.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != metadata[f"{name}_sha256"]:
            raise ValueError("Runtime checksum mismatch")
    for checkpoint in metadata["checkpoints"].values():
        if digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed")
    policy = policy_records(root, config)
    expected = {
        (r["layout_id"], size, name)
        for _, r in policy
        for size in config["batch_sizes"]
        for name in metadata["methods"]
    }
    grouped = defaultdict(list)
    for row in rows(root / "sessions.jsonl.gz"):
        grouped[row["layout_id"], row["batch_size"], row["method"]].append(row)
    if set(grouped) != expected:
        raise ValueError("Session coverage mismatch")
    medians = {}
    timing_fields = ("total_seconds", "setup_seconds", "search_seconds", "prediction_seconds")
    for key, values in grouped.items():
        if len(values) != config["repetitions"] or {v["repetition"] for v in values} != set(
            range(config["repetitions"])
        ):
            raise ValueError("Missing or duplicate repetitions")
        if len({v["result_sha256"] for v in values}) != 1:
            raise ValueError("Repeated paths or counts changed")
        for v in values:
            if (
                any(not np.isfinite(v[f]) or v[f] < 0 for f in timing_fields)
                or v["total_seconds"] <= 0
                or v["setup_seconds"] + v["search_seconds"] > v["total_seconds"]
            ):
                raise ValueError("Invalid timing")
        medians[key] = {f: statistics.median(v[f] for v in values) for f in timing_fields}
    outcomes = defaultdict(dict)
    endpoints = {}
    for directory, record in policy:
        _, goal, starts, labels = query_map(directory, record, config["session_starts"])
        for q, start in enumerate(starts):
            endpoints[record["layout_id"], q] = (list(start), list(goal), int(labels[start]))
    count = 0
    for row in rows(root / "queries.jsonl.gz"):
        key = row["layout_id"], row["batch_size"], row["method"]
        q = row["query"]
        if key not in expected or q in outcomes[key] or not 0 <= q < row["batch_size"]:
            raise ValueError("Invalid or duplicate query")
        if (row["start"], row["goal"], row["optimal_cost"]) != endpoints[row["layout_id"], q]:
            raise ValueError("Query does not match the frozen dataset")
        cost = row["optimal_cost"] if row["optimal_cost"] >= 0 else None
        if row["cost"] != cost or row["status"] != ("found" if cost is not None else "no_path"):
            raise ValueError("Incorrect query outcome")
        outcomes[key][q] = row["stats"]["expanded"] if row["stats"] else None
        count += 1
    if (
        set(outcomes) != expected
        or any(len(v) != k[1] for k, v in outcomes.items())
        or count != metadata["query_records"]
        or count * config["repetitions"] != metadata["timed_queries"]
        or len(medians) * config["repetitions"] != metadata["sessions"]
    ):
        raise ValueError("Workload counts mismatch")
    for row in audit:
        key = row["layout_id"], config["primary_batch_size"], "prepared_manhattan"
        if outcomes[key][row["query"]] != row["expanded"]["manhattan"]:
            raise ValueError("Headroom and runtime queries do not agree")
    rng = np.random.default_rng(config["bootstrap_seed"] + 1)
    summary = {"headroom": diagnostic, "measurement": metadata, "families": {}, "overall": {}}
    map_values = defaultdict(lambda: defaultdict(dict))
    baselines = ("prepared_manhattan", "frozen_distance", "control_distance", "control_detour")
    for family in FAMILIES:
        layouts = [r["layout_id"] for _, r in policy if f"{r['kind']}_{r['width']}" == family]
        summary["families"][family] = {}
        for size in config["batch_sizes"]:
            summary["families"][family][str(size)] = {}
            for name in metadata["methods"]:
                metrics = defaultdict(list)
                for layout in layouts:
                    value = medians[layout, size, name]
                    for field in timing_fields:
                        metrics[field.replace("_seconds", "_ms_per_query")].append(
                            value[field] * 1000 / size
                        )
                    for reference in baselines:
                        metric = f"ratio_to_{reference}"
                        metrics[metric].append(
                            value["total_seconds"]
                            / medians[layout, size, reference]["total_seconds"]
                        )
                    if name != "bfs_policy":
                        reachable = [q for q in range(size) if endpoints[layout, q][2] >= 0]
                        if reachable:
                            for ref, metric in (
                                ("prepared_manhattan", "expansion_ratio_to_manhattan"),
                                ("frozen_distance", "expansion_ratio_to_frozen"),
                            ):
                                metrics[metric].append(
                                    np.mean(
                                        [
                                            outcomes[layout, size, name][q]
                                            / outcomes[layout, size, ref][q]
                                            for q in reachable
                                        ]
                                    )
                                )
                summary["families"][family][str(size)][name] = {
                    metric: bootstrap_mean(values, rng, config["bootstrap_samples"])
                    for metric, values in metrics.items()
                }
                if size == config["primary_batch_size"]:
                    for metric in (f"ratio_to_{ref}" for ref in baselines):
                        map_values[name][metric][family] = metrics[metric]
    for name, metrics in map_values.items():
        summary["overall"][name] = {}
        for metric, families in metrics.items():
            draws = []
            for values in families.values():
                a = np.asarray(values)
                draws.append(
                    a[rng.integers(len(a), size=(config["bootstrap_samples"], len(a)))].mean(axis=1)
                )
            combined = np.mean(draws, axis=0)
            summary["overall"][name][metric] = {
                "mean": float(np.mean([np.mean(v) for v in families.values()])),
                "low": float(np.quantile(combined, 0.025)),
                "high": float(np.quantile(combined, 0.975)),
                "families": len(families),
            }
    summary["gate"] = apply_gate(summary, config)
    summary["verification"] = {
        "passed": True,
        "session_medians": len(medians),
        "dataset_endpoints_and_prefixes_checked": True,
        "headroom_matches_runtime_queries": True,
        "frozen_sources_and_weights_checked": True,
    }
    return summary, config


def figures(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from learned_astar.research.plot import save

    labels = ["Random 32", "Walls 32", "Random 64", "Walls 64"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for offset, (method, label, color) in enumerate(
        [
            ("learned_8", "Frozen CNN 8", "#2472a4"),
            ("attained_floor", "Best possible ties", "#cb763c"),
            ("exact_primary", "Exact primary heuristic", "#28856a"),
        ]
    ):
        values = [summary["headroom"]["families"][f][method] for f in FAMILIES]
        axes[0].bar(
            np.arange(4) + (offset - 1) * 0.23,
            [v["mean"] for v in values],
            width=0.21,
            label=label,
            color=color,
        )
    axes[0].axhline(1, color="#555555", linestyle="--")
    axes[0].set(
        title="Reachable expansion ratios to Manhattan", ylabel="Paired map mean (lower is better)"
    )
    axes[0].set_xticks(range(4), labels, rotation=20)
    axes[0].legend(fontsize=8, frameon=False)
    for offset, (method, label, color) in enumerate(
        [
            ("frozen_distance", "Frozen CNN 8", "#2472a4"),
            ("control_distance", "Continue old data", "#888888"),
            ("candidate_distance", "New data mixture", "#cb763c"),
        ]
    ):
        values = [
            summary["families"][f]["128"][method]["ratio_to_prepared_manhattan"] for f in FAMILIES
        ]
        axes[1].errorbar(
            np.arange(4) + (offset - 1) * 0.15,
            [v["mean"] for v in values],
            yerr=[[v["mean"] - v["low"] for v in values], [v["high"] - v["mean"] for v in values]],
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    axes[1].axhline(1, color="#555555", linestyle="--")
    axes[1].set(title="Complete time at 128 starts", ylabel="Paired ratio to prepared Manhattan")
    axes[1].set_xticks(range(4), labels, rotation=20)
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.3)
    fig.suptitle("Policy validation only · ordering floor and one-seed training pilot")
    save(fig, output, "ordering_and_pilot")


def interval(value):
    return f"{value['mean']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def report(summary, root, destination):
    headroom = summary["headroom"]
    gate = summary["gate"]
    controls = {
        name: json.loads((root / name / "summary.json").read_text())
        for name in ("control", "candidate")
    }
    labels = dict(zip(FAMILIES, ("Random 32", "Walls 32", "Random 64", "Walls 64"), strict=True))
    lines = [
        "# Ordering limits and training-pilot results",
        "",
        "This round follows the [fixed protocol](order_pilot.md). It uses policy validation "
        "to measure the expansion limit of Manhattan tie ordering and to assess a small "
        "training change. These are exploratory validation results from one model seed, "
        "not a new held-out test result.",
        "",
        "The new training mixture lowers validation loss, but does not establish a "
        "complete-time improvement over the frozen model or matched training control. "
        "Neither candidate passes the fixed gate, so the experiment stops before "
        "additional seeds or a new test set.",
        "",
        "## The ordering limit",
        "",
        f"The audit checked {headroom['queries']:,} queries on 120 validation maps. The "
        "start-aware reference attained the calculated expansion floor on every query. "
        "The exact-detour ordering also attained that floor on every measured query. "
        "Exhaustive tests separately cover every 3×3 obstacle layout and endpoint pair. "
        "The floor applies to this implementation's four-way unit-cost grids with "
        "consistent Manhattan primary priorities; it is not a general bound for "
        "arbitrary heuristics.",
        "",
        "| Condition | Frozen CNN 8 | Perfect distance ties | Best possible ties | "
        "Exact primary | Maximum tie saving |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, label in labels.items():
        data = headroom["families"][family]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{data[m]['mean']:.3f}"
                for m in ("learned_8", "exact_distance", "attained_floor", "exact_primary")
            )
            + f" | {(1 - data['attained_floor']['mean']) * 100:.1f}% |"
        )
    lines += [
        "",
        "Values are map-paired expansion ratios on reachable queries; lower is better. "
        "The maximum saving is relative to ordinary Manhattan, not an additional saving "
        "on top of the current model. The two BFS-based references use exact information "
        "and are count diagnostics, not deployable learned methods or free runtime baselines.",
        "",
        "Wall maps leave little room for tie ordering. Most excess work there comes from "
        "Manhattan's primary priority, which tie scores cannot change. Exact primary "
        "distances reduce expansions much more, showing where a different search design "
        "could help. Random 64×64 maps leave more room, but the existing model does not "
        "capture that opportunity.",
        "",
        "## Testing detour ordering",
        "",
        "Subtracting Manhattan from the prediction produces a score for extra travel "
        "around obstacles. A constant shift preserves its order while keeping scores "
        "nonnegative. This adds no neural-network parameters. It is evaluated separately "
        "from ordinary distance ties.",
        "",
        "| Condition | Frozen distance ties | Frozen detour ties | Exact detour ties |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, label in labels.items():
        data = headroom["families"][family]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{data[m]['mean']:.3f}" for m in ("learned_8", "learned_8_detour", "exact_detour")
            )
            + " |"
        )
    lines += [
        "",
        "The exact score reaches the floor, but the learned score is much less "
        "reliable. Prediction errors that were tolerable for distance ties can cause poor "
        "ordering after subtracting Manhattan. This experiment does not establish a "
        "runtime benefit merely from changing that formula.",
        "",
        "## Matched training pilot",
        "",
        "Both branches start from the same frozen 8-channel seed-7 model and receive "
        "960 optimizer updates. The control sees only the original training pool. The "
        "candidate sees 50% original examples, 25% new wall 32×32 examples, and 25% new "
        "mixed 64×64 examples. Both use the same learning rate, loss, and architecture. "
        "Checkpoint selection uses a shared mean validation loss across 32×32 and 64×64 "
        "maps; policy-validation maps never enter training.",
        "",
        "| Branch | Best epoch | Training seconds | Combined validation loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, data in controls.items():
        history = json.loads((root / name / "history_7.json").read_text())
        best_loss = min(row["validation_loss"] for row in history)
        lines.append(
            f"| {name.title()} | {data['best_epoch']} | {data['training_seconds']:.1f} | "
            f"{best_loss:.5f} |"
        )
    lines += [
        "",
        "Both models still have 2,713 parameters. The candidate lowers the combined "
        "distance-and-route validation loss substantially, especially on the larger "
        "maps. That improvement must be checked against search effort and complete time; "
        "it does not by itself show better search.",
        "",
        "## Complete time at 128 starts",
        "",
        "All setup is included. Values below are paired ratios to prepared Manhattan, "
        "with pointwise whole-map 95% intervals. All methods use the same maps and query "
        "prefixes. Five shuffled repetitions supply each session median. Smaller workloads "
        "of 1 and 64 starts and all absolute milliseconds are retained in summary.json.",
        "",
        "| Method | Random 32 | Walls 32 | Random 64 | Walls 64 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in summary["measurement"]["methods"]:
        lines.append(
            f"| {method.replace('_', ' ')} | "
            + " | ".join(
                interval(summary["families"][f]["128"][method]["ratio_to_prepared_manhattan"])
                for f in FAMILIES
            )
            + " |"
        )
    lines += [
        "",
        "## Decision and limitations",
        "",
        (
            "A candidate passed the fixed validation gate. Additional seeds are required "
            "before generating a fresh test set."
            if gate["expand_to_more_seeds"]
            else "Neither candidate rule passed the fixed validation gate. The pilot stops here: "
            "no additional seeds were trained and no new test set was consumed. Both "
            "checkpoints and all measurements are retained; the prior selected model is unchanged."
        ),
        "",
        "| Candidate rule | Mean time / frozen distance ties | "
        "Mean time / matched control | Gate passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for candidate in gate["candidates"]:
        rule = candidate["rule"]
        values = summary["overall"][f"candidate_{rule}"]
        lines.append(
            f"| {rule} | {interval(values['ratio_to_frozen_distance'])} | "
            f"{interval(values[f'ratio_to_control_{rule}'])} | {candidate['eligible']} |"
        )
    lines += [
        "",
        "These overall ratios weight the four map families equally and resample "
        "whole maps within each family. The gate also requires a wall-32 speed advantage "
        "over prepared Manhattan and limits family-specific regressions. Every individual "
        "gate check is recorded in gate.json. A lower distance loss alone does not pass "
        "this search-time gate.",
        "",
        "The result identifies a boundary for this approach: more accurate tie scores "
        "can only recover the small final-layer margin on wall maps. A future method "
        "would need to improve the primary search priority or reduce prediction cost "
        "further, with an explicit path-quality guarantee or tradeoff. Static repeated-goal "
        "workloads should still include reverse BFS as a strong practical control. "
        "This round does not claim generalization from one seed or performance on "
        "new held-out maps.",
        "",
        "## Evidence and checks",
        "",
        f"The runtime stage completed {summary['measurement']['sessions']:,} sessions "
        f"and {summary['measurement']['timed_queries']:,} query executions. Every method "
        "returned optimal costs and correct reachability. Checks verify repeated paths "
        "and counters, all expected methods and repetitions, frozen file hashes, original "
        "dataset endpoints, query prefixes, and agreement between headroom and runtime "
        "queries. The 600 new layouts exclude all 6,200 earlier layout groups and are "
        "separated between training, checkpoint validation, and policy validation.",
        "",
        "![Ordering limits and training pilot](../results/research_v7/ordering_and_pilot.png)",
        "",
        "The [archive](../results/research_v7/) contains raw diagnostic counts, runtime "
        "repetitions, query outcomes, protocol, data manifests, source hashes, summaries, "
        "and the gate decision. Both trained checkpoints and histories are retained in "
        "artifacts/checkpoints/research_v7/.",
        "",
    ]
    destination.write_text("\n".join(lines))


def archive(root, summary, config, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "protocol.toml",
        "fingerprints.json",
        "datasets_verified.json",
        "policy_manifest.json",
        "headroom.jsonl.gz",
        "headroom_verified.json",
        "headroom_summary.json",
        "training_gate.json",
        "sessions.jsonl.gz",
        "queries.jsonl.gz",
        "measurement.json",
        "summary.json",
        "gate.json",
    ):
        shutil.copy2(root / name, output / name)
    for name in ("hard_32", "mixed_64", "policy_64"):
        shutil.copy2(root / name / "metadata.json", output / f"{name}_metadata.json")
        with gzip.open(output / f"{name}_manifest.jsonl.gz", "wb") as stream:
            stream.write((root / name / "manifest.jsonl").read_bytes())
    checkpoints = Path("artifacts/checkpoints/research_v7")
    checkpoints.mkdir(parents=True, exist_ok=False)
    for branch in ("control", "candidate"):
        shutil.copytree(root / branch, checkpoints / branch)
    figures(summary, output)
    report(summary, root, Path("docs/order_pilot_results.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("headroom", "final"))
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v7"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v7"))
    args = parser.parse_args()
    if args.stage == "headroom":
        config = tomllib.loads((args.run / "protocol.toml").read_text())
        result, _ = summarize_headroom(args.run, config)
        save_json(args.run / "headroom_summary.json", result)
        print(json.dumps(result, indent=2))
    else:
        summary, config = analyze(args.run)
        save_json(args.run / "summary.json", summary)
        save_json(args.run / "gate.json", summary["gate"])
        archive(args.run, summary, config, args.output)
        print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
