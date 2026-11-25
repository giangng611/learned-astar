"""Train a tiny runtime predictor on separate selector development maps."""

import argparse
import copy
import json
import math
import statistics
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from prepare_selector_data import METHODS, summarize
from torch import nn

from learned_astar.reproducibility import configure_torch
from learned_astar.research.experiment import digest, save_json

FEATURES = (
    "height",
    "width",
    "obstacle_fraction",
    "max_line_obstacles",
    "line_obstacle_std",
    "queries",
    "unique_goals",
    "mean_manhattan_fraction",
)


class RuntimeSelector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURES), width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, values):
        return self.layers(values)


def fit_scaler(values):
    """Caller supplies training rows only; constant features keep a scale of one."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    return mean, np.where(scale > 1e-8, scale, 1.0)


def hand_rule(values):
    """Fixed before collecting this pilot; no family labels or runtime targets."""
    if values["unique_goals"] <= 8:
        return "bfs_policy"
    return "landmark_demand" if values["max_line_obstacles"] >= 0.8 else "manhattan_demand"


def predict_one(model, values, mean, scale):
    array = np.asarray([values[k] for k in FEATURES], dtype=np.float32)
    inputs = torch.from_numpy(((array - mean) / scale).astype(np.float32))
    return METHODS[int(model(inputs).argmin().item())]


def timed_choice(function, values, repetitions):
    samples, choice = [], None
    for _ in range(repetitions):
        begun = time.perf_counter()
        current = function(values)
        samples.append(time.perf_counter() - begun)
        if choice is not None and choice != current:
            raise ValueError("Prediction changed between timing repetitions")
        choice = current
    return choice, statistics.median(samples)


def train(data_root, output, config_path):
    config = tomllib.loads(config_path.read_text())
    for key in ("hidden_channels", "epochs", "patience", "inference_repetitions"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["cpu_threads"] != 1 or not 0 < config["learning_rate"] < 1:
        raise ValueError("Use one CPU thread and a learning rate between zero and one")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    # Recheck saved paths, source hashes, coverage, and median arithmetic before training.
    summarize(data_root)
    rows = json.loads((data_root / "dataset.json").read_text())
    if {r["split"] for r in rows} != {"train", "validation"}:
        raise ValueError("Expected training and validation only")
    groups = {s: {r["group_id"] for r in rows if r["split"] == s} for s in ("train", "validation")}
    if groups["train"] & groups["validation"]:
        raise ValueError("Training and validation layouts overlap")
    configure_torch(config["seed"], config["cpu_threads"])
    values = np.asarray([[r["features"][k] for k in FEATURES] for r in rows], dtype=np.float32)
    times = np.asarray([[r["median_seconds"][m] for m in METHODS] for r in rows])
    train_mask = np.asarray([r["split"] == "train" for r in rows])
    if not np.isfinite(values).all() or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("Nonfinite features or invalid runtimes")
    mean, scale = fit_scaler(values[train_mask])
    logs = np.log(times)
    target_mean = float(logs[train_mask].mean())
    target_scale = max(float(logs[train_mask].std()), 1e-8)
    inputs = torch.from_numpy(((values - mean) / scale).astype(np.float32))
    targets = torch.from_numpy(((logs - target_mean) / target_scale).astype(np.float32))
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    val_indices = torch.from_numpy(np.flatnonzero(~train_mask))
    model = RuntimeSelector(config["hidden_channels"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best_loss, best_epoch, best_state, stale = [], math.inf, 0, None, 0
    begun = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(inputs[train_indices]), targets[train_indices])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.mse_loss(model(inputs[val_indices]), targets[val_indices])
            )
        if not math.isfinite(validation_loss) or not math.isfinite(float(loss.detach())):
            raise ValueError("Training diverged")
        history.append(
            {"epoch": epoch, "train_loss": float(loss.detach()), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch, best_state, stale = (
                validation_loss,
                epoch,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch}: train={loss.item():.5f}, validation={validation_loss:.5f}",
                flush=True,
            )
        if stale >= config["patience"]:
            break
    training_seconds = time.perf_counter() - begun
    model.load_state_dict(best_state)
    model.eval()
    # The fixed-method control is selected on training data, not validation.
    best_fixed = METHODS[int(times[train_mask].mean(axis=0).argmin())]
    comparisons = []
    with torch.inference_mode():

        def predictor(values):
            return predict_one(model, values, mean, scale)

        for _ in range(20):
            predictor(rows[0]["features"])
            hand_rule(rows[0]["features"])
        for row in rows:
            if row["split"] != "validation":
                continue
            choice, inference = timed_choice(
                predictor, row["features"], config["inference_repetitions"]
            )
            manual, rule_time = timed_choice(
                hand_rule, row["features"], config["inference_repetitions"]
            )
            costs = row["median_seconds"]
            comparisons.append(
                {
                    "layout_id": row["layout_id"],
                    "group_id": row["group_id"],
                    "family": row["family"],
                    "goal_count": row["goal_count"],
                    "chosen_method": choice,
                    "hand_rule_method": manual,
                    "feature_seconds": row["feature_seconds"],
                    "inference_seconds": inference,
                    "costs_seconds": {
                        **costs,
                        "learned_selector": costs[choice] + row["feature_seconds"] + inference,
                        "hand_rule": costs[manual] + row["feature_seconds"] + rule_time,
                        "best_fixed_train": costs[best_fixed],
                        "oracle": min(costs.values()),
                    },
                }
            )
    methods = (*METHODS, "learned_selector", "hand_rule", "best_fixed_train", "oracle")
    # Equal map weighting, with each map's workloads averaged first.
    map_costs = {}
    for group in sorted(groups["validation"]):
        subset = [r for r in comparisons if r["group_id"] == group]
        map_costs[group] = {
            m: statistics.mean(r["costs_seconds"][m] for r in subset) for m in methods
        }
    summary = {
        "stage": "validation_pilot_only",
        "timing": "recombined_medians_not_live_end_to_end",
        "train_maps": len(groups["train"]),
        "validation_maps": len(groups["validation"]),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int((~train_mask).sum()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": config["seed"],
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": training_seconds,
        "best_fixed_train": best_fixed,
        "mean_validation_session_ms": {
            m: 1000 * statistics.mean(c[m] for c in map_costs.values()) for m in methods
        },
        "mean_map_paired_ratio": {
            f"learned_over_{m}": statistics.mean(
                c["learned_selector"] / c[m] for c in map_costs.values()
            )
            for m in ("hand_rule", "best_fixed_train", "oracle")
        },
        "median_inference_ms": 1000
        * statistics.median(r["inference_seconds"] for r in comparisons),
        "dataset_sha256": digest(data_root / "dataset.json"),
        "data_summary_sha256": digest(data_root / "summary.json"),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.toml").write_bytes(config_path.read_bytes())
    torch.save(
        {
            "state_dict": best_state,
            "hidden_channels": config["hidden_channels"],
            "features": list(FEATURES),
            "methods": list(METHODS),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "target_mean": target_mean,
            "target_scale": target_scale,
            "best_epoch": best_epoch,
            "seed": config["seed"],
            "dataset_sha256": summary["dataset_sha256"],
        },
        output / "model.pt",
    )
    save_json(output / "history.json", history)
    save_json(output / "validation_predictions.json", comparisons)
    save_json(output / "summary.json", summary)
    save_json(
        output / "fingerprints.json",
        {
            str(p): digest(p)
            for p in [
                Path(__file__),
                config_path,
                output / "model.pt",
                data_root / "dataset.json",
                data_root / "summary.json",
            ]
        },
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/selector_pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/selector_model"))
    parser.add_argument("--config", type=Path, default=Path("configs/selector_training.toml"))
    args = parser.parse_args()
    train(args.data, args.output, args.config)


if __name__ == "__main__":
    main()
