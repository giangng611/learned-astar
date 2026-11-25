"""Probe intermediate goal counts on new development maps with a frozen selector."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import torch
from prepare_selector_data import METHODS, collect, prepare, summarize, verify_inputs
from train_selector import FEATURES, RuntimeSelector, hand_rule, predict_one, timed_choice

from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json


def prepare_probe(root, previous, model_root):
    verify_inputs(previous)
    checkpoint = model_root / "model.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if metadata["dataset_sha256"] != digest(previous / "dataset.json"):
        raise ValueError("Checkpoint does not match the previous dataset")
    inputs = root.with_name(root.name + "_inputs")
    if root.exists() or inputs.exists():
        raise FileExistsError("Probe already exists; continue with measure or evaluate")
    separation = json.loads((previous / "separation.json").read_text())
    records = json.loads((previous / "manifest.json").read_text())
    inputs.mkdir(parents=True)
    save_json(inputs / "separation.json", separation)
    save_json(inputs / "policy_manifest.json", records)
    shutil.copyfile(checkpoint, inputs / "model.pt")
    config = inputs / "protocol.toml"
    config.write_text(
        "seed = 129501\nquery_seed = 129502\norder_seed = 129503\nsizes = [32, 64]\n"
        "train_per_family = 4\nvalidation_per_family = 2\nqueries_per_session = 128\n"
        "goal_counts = [16, 32, 64]\nrepetitions = 3\ncpu_threads = 1\n"
        f"exclusion_archive = {json.dumps(str(inputs))}\n"
    )
    prepare(root, config)
    # The inherited split labels are retained for possible later development.
    # This diagnostic evaluates all 24 maps and performs no fitting on either split.
    frozen = json.loads((root / "fingerprints.json").read_text())
    frozen[str(inputs / "model.pt")] = digest(inputs / "model.pt")
    save_json(root / "fingerprints.json", frozen)


def evaluate(root):
    output = root / "selector_evaluation.json"
    if output.exists():
        raise FileExistsError("Evaluation already exists; read selector_evaluation.json")
    summarize(root)
    checkpoint = root.with_name(root.name + "_inputs") / "model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["features"] != list(FEATURES) or saved["methods"] != list(METHODS):
        raise ValueError("Checkpoint schema mismatch")
    configure_torch(7, 1)
    model = RuntimeSelector(saved["hidden_channels"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    mean, scale = (
        np.asarray(saved["mean"], dtype=np.float32),
        np.asarray(saved["scale"], dtype=np.float32),
    )
    rows = json.loads((root / "dataset.json").read_text())
    predictions = []
    with torch.inference_mode():

        def predictor(values):
            return predict_one(model, values, mean, scale)

        for _ in range(20):
            predictor(rows[0]["features"])
            hand_rule(rows[0]["features"])
        for row in rows:
            chosen, inference = timed_choice(predictor, row["features"], 31)
            manual, rule_time = timed_choice(hand_rule, row["features"], 31)
            times = row["median_seconds"]
            predictions.append(
                {
                    "group_id": row["group_id"],
                    "family": row["family"],
                    "goal_count": row["goal_count"],
                    "chosen": chosen,
                    "rule": manual,
                    "feature_seconds": row["feature_seconds"],
                    "inference_seconds": inference,
                    "seconds": {
                        **times,
                        "learned": times[chosen] + row["feature_seconds"] + inference,
                        "rule": times[manual] + row["feature_seconds"] + rule_time,
                        "oracle": min(times.values()),
                    },
                }
            )

    def aggregate(subset):
        groups = sorted({r["group_id"] for r in subset})
        costs = [
            {
                m: statistics.mean(r["seconds"][m] for r in subset if r["group_id"] == group)
                for m in (*METHODS, "learned", "rule", "oracle")
            }
            for group in groups
        ]
        return {
            "mean_session_ms": {
                m: 1000 * statistics.mean(c[m] for c in costs)
                for m in (*METHODS, "learned", "rule", "oracle")
            },
            "mean_map_ratio": {
                f"learned_over_{m}": statistics.mean(c["learned"] / c[m] for c in costs)
                for m in ("rule", "bfs_policy", "oracle")
            },
            "fastest_method_choices": sum(
                r["seconds"][r["chosen"]] == r["seconds"]["oracle"] for r in subset
            ),
            "rows": len(subset),
        }

    result = {
        "stage": "development_probe_not_final_test",
        "model_sha256": digest(checkpoint),
        "timing": "recombined_medians_not_live_end_to_end",
        "maps": len({r["group_id"] for r in predictions}),
        "trained": False,
        "overall": aggregate(predictions),
        "by_goal_count": {
            str(g): aggregate([r for r in predictions if r["goal_count"] == g])
            for g in sorted({r["goal_count"] for r in predictions})
        },
        "dataset_sha256": digest(root / "dataset.json"),
    }
    verify_inputs(root)
    save_json(root / "selector_predictions.json", predictions)
    save_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "measure", "evaluate"))
    parser.add_argument("--run", type=Path, default=Path("outputs/selector_transition"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/selector_pilot"))
    parser.add_argument("--model", type=Path, default=Path("outputs/selector_model"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_probe(args.run, args.previous, args.model)
    elif args.stage == "measure":
        collect(args.run)
    else:
        evaluate(args.run)


if __name__ == "__main__":
    main()
