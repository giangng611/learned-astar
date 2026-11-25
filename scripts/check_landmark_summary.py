"""Independently recompute the archived session medians and summary point estimates."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def check(root):
    summary = json.loads((root / "summary.json").read_text())
    groups = defaultdict(list)
    with gzip.open(root / "sessions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["layout_id"], row["batch_size"], row["method"]].append(row)
    medians = {}
    for key, rows in groups.items():
        total = statistics.median(r["total_seconds"] for r in rows)
        chosen = min(
            (r for r in rows if r["total_seconds"] == total), key=lambda r: r["repetition"]
        )
        medians[key] = chosen
    archived = json.loads((root / "median_sessions.json").read_text())
    assert len(archived) == len(medians)
    for row in archived:
        key = row["layout_id"], row["batch_size"], row["method"]
        assert medians[key] == row
    means_checked = ratios_checked = 0
    for family, batches in summary["families"].items():
        for size_string, batch in batches.items():
            size = int(size_string)
            for method, metrics in batch["methods"].items():
                selected = [
                    r
                    for (layout, n, name), r in medians.items()
                    if r["family"] == family and n == size and name == method
                ]
                assert len(selected) == summary["config"]["maps_per_family"]
                for metric, estimate in metrics.items():
                    if metric.endswith("_ms_per_query"):
                        source = metric.removesuffix("_ms_per_query") + "_seconds"
                        computed = statistics.fmean(r[source] * 1000 / size for r in selected)
                        means_checked += 1
                    elif metric.startswith("ratio_to_"):
                        reference = metric.removeprefix("ratio_to_")
                        computed = statistics.fmean(
                            r["total_seconds"]
                            / medians[r["layout_id"], size, reference]["total_seconds"]
                            for r in selected
                        )
                        ratios_checked += 1
                    else:
                        continue
                    assert math.isclose(computed, estimate["mean"], rel_tol=1e-12), metric
    result = {
        "passed": True,
        "median_sessions": len(medians),
        "timing_means_checked": means_checked,
        "paired_means_checked": ratios_checked,
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "independent_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/primary_step2"))
    args = parser.parse_args()
    print(json.dumps(check(args.run)))
