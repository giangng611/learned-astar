import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COUNT_FIELDS = (
    "expanded",
    "unique_expanded",
    "generated",
    "queue_insertions",
    "stale_pops",
    "reexpanded",
    "max_frontier",
)


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {key: None for key in ("mean", "median", "q25", "q75", "p95", "max")}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def bootstrap_mean(values, rng, samples):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "low": None, "high": None, "maps": 0}
    draws = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "maps": len(values),
    }


def collapse_repetitions(rows, repetitions):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layout_id"], row["query"], row["method"])].append(row)
    collapsed = []
    for group in groups.values():
        if len(group) != repetitions or {r["repetition"] for r in group} != set(range(repetitions)):
            raise ValueError("Missing or duplicate timing repetitions")
        row = {k: v for k, v in group[0].items() if k != "repetition"}
        statuses = {r["status"] for r in group}
        row["status"] = next(iter(statuses)) if len(statuses) == 1 else "mixed_limit"
        row["completed"] = row["status"] in ("found", "no_path")
        if row["completed"] and len({r["cost"] for r in group}) != 1:
            raise ValueError("Repeated searches disagree on path cost")
        for field in ("elapsed_seconds", *COUNT_FIELDS):
            row[field] = float(np.median([r[field] for r in group]))
        row["path_ratio"] = (
            row["cost"] / row["optimal_cost"]
            if row["status"] == "found" and row["optimal_cost"] > 0
            else None
        )
        collapsed.append(row)
    return collapsed


def describe_method(rows):
    reachable = [r for r in rows if r["optimal_cost"] >= 0]
    unreachable = [r for r in rows if r["optimal_cost"] < 0]
    paths = [r["path_ratio"] for r in rows if r["path_ratio"] is not None]
    return {
        "queries": len(rows),
        "maps": len({r["layout_id"] for r in rows}),
        "reachable_queries": len(reachable),
        "unreachable_queries": len(unreachable),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "reachable_success_rate": sum(r["status"] == "found" for r in reachable) / len(reachable)
        if reachable
        else None,
        "unreachable_correct_rate": sum(r["status"] == "no_path" for r in unreachable)
        / len(unreachable)
        if unreachable
        else None,
        "optimal_fraction_of_found": sum(p == 1 for p in paths) / len(paths) if paths else None,
        "completion_ms": distribution(
            [1000 * r["elapsed_seconds"] for r in rows if r["completed"]]
        ),
        "attempt_ms": distribution([1000 * r["elapsed_seconds"] for r in rows]),
        "expanded": distribution([r["expanded"] for r in rows if r["completed"]]),
        "path_ratio": distribution(paths),
        "reexpanded": distribution([r["reexpanded"] for r in rows if r["completed"]]),
    }


def paired_intervals(rows, rng, samples):
    baseline = {(r["layout_id"], r["query"]): r for r in rows if r["method"] == "astar_manhattan"}
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        ref = baseline[(row["layout_id"], row["query"])]
        if not row["completed"] or not ref["completed"]:
            continue
        values = {
            "runtime_ratio": row["elapsed_seconds"] / ref["elapsed_seconds"],
            "expansion_ratio": row["expanded"] / ref["expanded"],
        }
        if row["path_ratio"] is not None:
            values["path_ratio"] = row["path_ratio"]
        names = [row["method"]]
        if row["method"].startswith("learned_seed_"):
            names.append("learned_mean")
        for name in names:
            for metric, value in values.items():
                grouped[name][metric][row["layout_id"]].append(value)
    # Every map has the same query and model counts in completed runs. Average
    # models and queries inside each map; bootstrap only the resulting map means.
    result = {}
    for name in sorted(grouped):
        result[name] = {}
        for metric, maps in sorted(grouped[name].items()):
            result[name][metric] = bootstrap_mean([np.mean(v) for v in maps.values()], rng, samples)
    return result


def error_search_correlation(rows, map_rows, name):
    baseline = {
        (r["layout_id"], r["query"]): r
        for r in rows
        if r["method"] == "astar_manhattan" and r["completed"]
    }
    ratios = defaultdict(list)
    for row in rows:
        key = row["layout_id"], row["query"]
        if row["method"] == name and row["completed"] and key in baseline:
            ratios[row["layout_id"]].append(row["expanded"] / baseline[key]["expanded"])
    pairs = [
        (m["prediction_errors"][name]["mae"], np.mean(ratios[m["layout_id"]]))
        for m in map_rows
        if m["layout_id"] in ratios
    ]
    values = np.asarray(pairs)
    correlation = None
    if len(values) > 1 and np.all(values.std(axis=0) > 0):
        correlation = float(np.corrcoef(values.T)[0, 1])
    return {
        "pearson_r": correlation,
        "maps": len(pairs),
        "variables": "Map MAE and mean expansion ratio relative to Manhattan A*",
    }


def summarize(directory, output):
    directory, output = Path(directory), Path(output)
    settings = json.loads((directory / "settings.json").read_text())
    if not settings.get("completed"):
        raise ValueError("Evaluation has not finished")
    raw = (directory / "runs.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != settings["runs_sha256"]:
        raise ValueError("Result checksum mismatch")
    rows = collapse_repetitions(
        (json.loads(line) for line in raw.splitlines()), settings["config"]["repetitions"]
    )
    expected_methods = {
        "bfs",
        "dijkstra",
        "astar_manhattan",
        "weighted_astar_1.25",
        "weighted_astar_1.5",
        "weighted_astar_2.0",
        *settings["checkpoints"],
    }
    query_methods = defaultdict(set)
    for row in rows:
        query_methods[(row["layout_id"], row["query"])].add(row["method"])
    if len(query_methods) != settings["queries"] or any(
        methods != expected_methods for methods in query_methods.values()
    ):
        raise ValueError("Missing query or method results")
    map_rows = [json.loads(line) for line in (directory / "maps.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(settings["config"]["bootstrap_seed"])
    summary = {"settings": settings, "splits": {}}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        methods = {
            name: describe_method([r for r in subset if r["method"] == name])
            for name in sorted({r["method"] for r in subset})
        }
        learned = [r for r in subset if r["method"].startswith("learned_seed_")]
        methods["learned_mean"] = describe_method(learned)
        methods["learned_mean"]["query_unit"] = "model-query pair; three models per query"
        selected_maps = [r for r in map_rows if r["split"] == split]
        errors = {}
        for name in settings["checkpoints"]:
            errors[name] = {
                metric: distribution([m["prediction_errors"][name][metric] for m in selected_maps])
                for metric in (
                    "mae",
                    "rmse",
                    "overestimate_fraction",
                    "mean_overestimate",
                    "max_overestimate",
                )
            }
        summary["splits"][split] = {
            "methods": methods,
            "paired_95ci": paired_intervals(subset, rng, settings["config"]["bootstrap_samples"]),
            "prediction_errors": errors,
            "error_search_correlation": {
                name: error_search_correlation(subset, selected_maps, name)
                for name in settings["checkpoints"]
            },
            "oracle_ms": distribution(
                [1000 * np.median(m["oracle_seconds"]) for m in selected_maps]
            ),
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Keep compact query evidence alongside the readable aggregate, without
    # treating repetitions as extra samples.
    with gzip.open(output / "queries.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize repeated runs and bootstrap maps")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)


if __name__ == "__main__":
    main()
