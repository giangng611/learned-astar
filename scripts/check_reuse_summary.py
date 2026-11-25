"""Independently recompute retained session medians and paired runtime means."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def check(root):
    config = tomllib.loads((root / "protocol.toml").read_text())
    metadata = json.loads((root / "measurement.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != metadata[f"{name}_sha256"]:
            raise ValueError("Raw checksum mismatch")
    grouped = defaultdict(list)
    for row in rows(root / "sessions.jsonl.gz"):
        grouped[row["split"], row["layout_id"], row["batch_size"], row["method"]].append(row)
    collapsed = {}
    fields = tuple(k for k in next(iter(grouped.values()))[0] if k.endswith("_seconds"))
    for key, values in grouped.items():
        if len(values) != config["repetitions"] or {r["repetition"] for r in values} != set(
            range(config["repetitions"])
        ):
            raise ValueError("Repetition coverage mismatch")
        collapsed[key] = {field: statistics.median(r[field] for r in values) for field in fields}
    retained = list(rows(root / "session_medians.jsonl.gz"))
    if len(retained) != len(collapsed):
        raise ValueError("Median coverage mismatch")
    seen = set()
    for row in retained:
        key = row["split"], row["layout_id"], row["batch_size"], row["method"]
        if key in seen or any(row[field] != collapsed[key][field] for field in fields):
            raise ValueError("Retained median mismatch")
        seen.add(key)
    comparisons = 0
    for split, workloads in summary["groups"].items():
        for size_text, group in workloads.items():
            size = int(size_text)
            layouts = sorted({k[1] for k in collapsed if k[0] == split and k[2] == size})
            for family, metrics in group["methods"].items():
                names = (
                    [f"{family}_seed_{seed}" for seed in config["seeds"]]
                    if family.startswith("width_") and "_seed_" not in family
                    else [family]
                )
                for field in fields:
                    actual = statistics.mean(
                        statistics.mean(
                            collapsed[split, layout, size, name][field] for name in names
                        )
                        * 1000
                        / size
                        for layout in layouts
                    )
                    reported = metrics[field.replace("_seconds", "_ms_per_query")]["mean"]
                    if not math.isclose(actual, reported, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Component summary mismatch")
                    comparisons += 1
                for reference in ("manhattan", "prepared_manhattan", "bfs_policy"):
                    actual = statistics.mean(
                        statistics.mean(
                            collapsed[split, layout, size, name]["total_seconds"] for name in names
                        )
                        / collapsed[split, layout, size, reference]["total_seconds"]
                        for layout in layouts
                    )
                    reported = metrics[f"ratio_to_{reference}"]["mean"]
                    if not math.isclose(actual, reported, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Paired ratio mismatch")
                    comparisons += 1
    return {
        "passed": True,
        "median_sessions": len(collapsed),
        "recomputed_summary_means": comparisons,
        "raw_checksums_match": True,
        "summary_sha256": digest(root / "summary.json"),
        "checker_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/research_v6"))
    args = parser.parse_args()
    result = check(args.results)
    (args.results / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
