"""Width-only controls and validation-only selection by complete query time."""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from learned_astar.evaluation.summarize import bootstrap_mean
from learned_astar.models import load_model
from learned_astar.research.experiment import digest


def make_spec(root, config, selection=None):
    models, plan, aggregates = {}, {"astar_manhattan": {"model": None, "mode": "manhattan"}}, {}
    manifest_hash = digest(Path(config["development"]) / "manifest.jsonl")
    for width in config["widths"]:
        family = f"width_{width}"
        aggregates[family] = []
        directory = (
            Path(config["reference_models"])
            if width == config["reference_width"]
            else Path(root) / family
        )
        for seed in config["training"]["seeds"]:
            name = f"{family}_seed_{seed}"
            path = directory / f"seed_{seed}.pt"
            model, meta = load_model(path)
            expected = {"channels": width, "residual": False, "obstacles": True}
            if model.model_config != expected or meta["seed"] != seed:
                raise ValueError("Checkpoint width or seed differs from the protocol")
            if meta["dataset_manifest_sha256"] != manifest_hash:
                raise ValueError("Models did not use the same development dataset")
            expected_training = {**config["training"], "seeds": tuple(config["training"]["seeds"])}
            if any(meta["training_config"][k] != v for k, v in expected_training.items()):
                raise ValueError("Checkpoint training settings differ from the protocol")
            models[name] = str(path)
            plan[name] = {"model": name, "mode": "tie"}
            aggregates[family].append(name)
    return {
        "models": models,
        "search_plan": plan,
        "aggregates": aggregates,
        "selection": selection,
        "strict_holdout": True,
    }


def paired_widths(directory, config):
    with gzip.open(Path(directory) / "queries.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    reference = config["reference_width"]
    indexed = {(r["layout_id"], r["query"], r["method"]): r for r in rows}
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if not row["method"].startswith("width_"):
            continue
        _, width, _, seed = row["method"].split("_")
        ref = indexed[row["layout_id"], row["query"], f"width_{reference}_seed_{seed}"]
        if not row["completed"] or not ref["completed"]:
            continue
        for field in ("elapsed_seconds", "prediction_seconds", "search_seconds"):
            metrics[row["split"]][width, field + "_ratio"][row["layout_id"]].append(
                row[field] / ref[field]
            )
        if row["optimal_cost"] >= 0:
            metrics[row["split"]][width, "reachable_expansion_ratio"][row["layout_id"]].append(
                row["expanded"] / ref["expanded"]
            )
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"] + 1)
    result = {}
    for split, fields in sorted(metrics.items()):
        result[split] = {}
        for (width, field), maps in sorted(fields.items()):
            result[split].setdefault(width, {})[field] = bootstrap_mean(
                [np.mean(v) for v in maps.values()], rng, config["evaluation"]["bootstrap_samples"]
            )
    return result


def choose_width(summary, paired, config):
    if not summary["settings"]["validation"]:
        raise ValueError("Width selection requires validation, not test results")
    candidates = []
    for width in config["widths"]:
        if width == config["reference_width"]:
            continue
        groups = [summary["splits"][s]["methods"][f"width_{width}"] for s in ("random", "walls")]
        correct = all(
            g["reachable_success_rate"] == 1
            and g["optimal_fraction_of_found"] == 1
            and g["unreachable_correct_rate"] in (1, None)
            and g["statuses"].get("found", 0) + g["statuses"].get("no_path", 0) == g["queries"]
            for g in groups
        )
        expansions = [g["paired_95ci"]["reachable_expansion_ratio"]["mean"] for g in groups]
        score = float(
            np.mean(
                [
                    paired[s][str(width)]["elapsed_seconds_ratio"]["mean"]
                    for s in ("random", "walls")
                ]
            )
        )
        candidates.append(
            {
                "width": width,
                "all_correct": correct,
                "expansion_ratios_to_manhattan": dict(
                    zip(("random", "walls"), expansions, strict=True)
                ),
                "retains_search_gain": all(v is not None and v < 1 for v in expansions),
                "runtime_ratio_to_reference": score,
                "eligible": correct
                and all(v is not None and v < 1 for v in expansions)
                and score < 1,
            }
        )
    eligible = [c for c in candidates if c["eligible"]]
    selected = (
        min(eligible, key=lambda c: (c["runtime_ratio_to_reference"], c["width"]))
        if eligible
        else None
    )
    return {
        "width": selected["width"] if selected else config["reference_width"],
        "smaller_model_selected": selected is not None,
        "candidates": candidates,
        "criterion": "Equal random/wall mean paired total-time ratio to the fixed 16-channel "
        "control, with correct optimal paths and fewer expansions than Manhattan in both groups.",
    }
