"""Predeclared factorial comparison of learning objectives and search rules."""

import json
from pathlib import Path

import numpy as np

from learned_astar.research.experiment import digest, evaluate_research


def make_spec(root, config, *, selection=None):
    root = Path(root)
    old = Path(config["sources"]["distance_models"])
    models, plan, aggregates = {}, {}, {}
    for name, mode in (
        ("astar_manhattan", "manhattan"),
        ("astar_fifo", "fifo"),
        ("weighted_astar_1.5", 1.5),
        ("oracle_lookup", "oracle"),
        ("oracle_tie", "oracle_tie"),
        ("exact_bfs_fresh", "exact"),
    ):
        plan[name] = {"model": None, "mode": mode}
    for family in ("distance", "route"):
        for mode in ("raw", "tie"):
            aggregates[f"{family}_{mode}_mean"] = []
        for seed in (7, 17, 27):
            name = f"{family}_seed_{seed}"
            path = (
                old / ("direct" if seed == 7 else "replicates") / f"seed_{seed}.pt"
                if family == "distance"
                else root / "route" / f"seed_{seed}.pt"
            )
            if not path.is_file():
                raise ValueError(f"Missing frozen checkpoint: {path}")
            models[name] = str(path)
            for mode in ("raw", "tie"):
                method = f"{name}_{mode}"
                plan[method] = {"model": name, "mode": mode}
                aggregates[f"{family}_{mode}_mean"].append(method)
    models["no_obstacles_seed_7"] = str(old / "no_obstacles/seed_7.pt")
    plan["no_obstacles_seed_7_tie"] = {"model": "no_obstacles_seed_7", "mode": "tie"}
    if selection and selection["family"] is not None:
        aggregates["selected_mean"] = aggregates[f"{selection['family']}_tie_mean"].copy()
    return {
        "models": models,
        "search_plan": plan,
        "aggregates": aggregates,
        "selection": selection,
        "strict_holdout": True,
    }


def evaluate_stage(directory, spec, output, config, *, validation, exclude_groups=()):
    return evaluate_research(
        directory, spec, output, config, validation=validation, exclude_groups=exclude_groups
    )


def choose_tie_family(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    if not summary["settings"]["validation"]:
        raise ValueError("Model selection requires policy validation, not test results")
    candidates = []
    for family in ("distance", "route"):
        groups = [
            summary["splits"][kind]["methods"][f"{family}_tie_mean"] for kind in ("random", "walls")
        ]
        valid = all(
            g["reachable_success_rate"] == 1
            and g["statuses"].get("found", 0) + g["statuses"].get("no_path", 0) == g["queries"]
            and g["optimal_fraction_of_found"] == 1
            and g["unreachable_correct_rate"] in (1, None)
            for g in groups
        )
        score = float(
            np.mean([g["paired_95ci"]["reachable_expansion_ratio"]["mean"] for g in groups])
        )
        candidates.append(
            {"family": family, "quality_gate_passed": valid, "expansion_score": score}
        )
    eligible = [c for c in candidates if c["quality_gate_passed"]]
    selected = (
        min(eligible, key=lambda c: (c["expansion_score"], c["family"]))
        if eligible
        else {
            "family": None,
            "quality_gate_passed": False,
            "expansion_score": None,
        }
    )
    return {
        **selected,
        "improves_expansions_on_validation": bool(eligible) and selected["expansion_score"] < 1,
        "candidates": candidates,
        "validation_summary_sha256": digest(summary_path),
    }
