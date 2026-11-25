"""Check archived checksums and timing arithmetic without generated output directories.

Full path and independent BFS checks are performed by report_demand_heuristic.py.
This separate standard-library check verifies the retained evidence and point estimates.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root):
    summary = json.loads((root / "summary.json").read_text())
    measurement = json.loads((root / "measurement.json").read_text())
    if summary["measurement"] != measurement:
        raise ValueError("Summary measurement differs from archive")
    for name in ("sessions", "queries"):
        if digest(root / f"{name}.jsonl.gz") != measurement[f"{name}_sha256"]:
            raise ValueError("Raw evidence checksum mismatch")
    records = json.loads((root / "policy_manifest.json").read_text())
    for record in records:
        if digest(root / "maps" / Path(record["file"]).name) != record["sha256"]:
            raise ValueError("Archived map checksum mismatch")
    groups = defaultdict(list)
    with gzip.open(root / "sessions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["family"], row["goal_count"], row["method"], row["layout_id"]].append(row)
    repetitions = summary["config"]["repetitions"]
    medians = {}
    for key, rows in groups.items():
        if len(rows) != repetitions or {r["repetition"] for r in rows} != set(range(repetitions)):
            raise ValueError("Missing or repeated timing repetition")
        if len({r["result_sha256"] for r in rows}) != 1:
            raise ValueError("Repeated search results differ")
        medians[key] = statistics.median(r["total_seconds"] for r in rows)
    expected = {
        (f"{r['kind']}_{r['width']}", g, m, r["layout_id"])
        for r in records
        for g in summary["config"]["goal_counts"]
        for m in summary["config"]["methods"]
    }
    if set(groups) != expected or len(groups) * repetitions != measurement["sessions"]:
        raise ValueError("Wrong session coverage")
    count = 0
    for family, workloads in summary["families"].items():
        for goals, workload in workloads.items():
            for method, metrics in workload["methods"].items():
                selected = {
                    k: v for k, v in medians.items() if k[:3] == (family, int(goals), method)
                }
                for metric, value in metrics.items():
                    if metric == "total_ms_per_query":
                        actual = (
                            statistics.fmean(selected.values())
                            * 1000
                            / summary["config"]["queries_per_session"]
                        )
                    elif metric.startswith("ratio_to_"):
                        reference = metric.removeprefix("ratio_to_")
                        actual = statistics.fmean(
                            v / medians[k[0], k[1], reference, k[3]] for k, v in selected.items()
                        )
                    else:
                        continue
                    if not math.isclose(actual, value["mean"], rel_tol=1e-12):
                        raise ValueError(
                            f"Incorrect summary mean: {family}/{goals}/{method}/{metric}"
                        )
                    count += 1
    return {"maps": len(records), "median_sessions": len(medians), "checked_means": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.archive), indent=2))
