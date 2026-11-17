import argparse
import hashlib
import json
import platform
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from learned_astar.config import require_finite, require_int
from learned_astar.data import load_record, read_manifest
from learned_astar.models import DistanceCNN, distance_scale, encode_inputs
from learned_astar.models.network import ARCHITECTURE, RESEARCH_ARCHITECTURE
from learned_astar.reproducibility import configure_torch


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (7, 17, 27)
    epochs: int = 40
    patience: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    cpu_threads: int = 4
    channels: int = 32
    residual: bool = False
    obstacles: bool = True

    def __post_init__(self):
        for key in ("epochs", "patience", "batch_size", "cpu_threads", "channels"):
            require_int(getattr(self, key), key, minimum=1)
        require_finite(self.learning_rate, "learning_rate")
        if self.learning_rate == 0:
            raise ValueError("Learning rate must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Training seeds must be nonempty and distinct")
        for seed in self.seeds:
            require_int(seed, "seed")
            if seed >= 2**32:
                raise ValueError("Seed must be less than 2**32")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if type(self.residual) is not bool or type(self.obstacles) is not bool:
            raise ValueError("Residual and obstacles options must be booleans")


def masked_loss(prediction, target, mask):
    errors = F.smooth_l1_loss(prediction, target, reduction="none")
    sizes = mask.flatten(1).sum(1)
    if torch.any(sizes == 0):
        raise ValueError("Every example must contain at least one reachable cell")
    return ((errors * mask).flatten(1).sum(1) / sizes).mean()


def load_training_tensors(directory, split):
    if split not in ("train", "validation"):
        raise ValueError("Training may only load train or validation splits")
    inputs, targets, masks, groups = [], [], [], []
    for record in read_manifest(directory, split):
        if record["status"] != "ready":
            continue
        data = load_record(directory, record)
        groups.append(record["group_id"])
        for goal, labels in zip(data["goals"], data["distances"], strict=True):
            inputs.append(encode_inputs(data["blocked"], tuple(map(int, goal))))
            targets.append(np.maximum(labels, 0).astype(np.float32) / distance_scale(*labels.shape))
            masks.append(labels >= 0)
    if not inputs:
        raise ValueError(f"No usable examples in {split}")
    return tuple(torch.from_numpy(np.stack(x)) for x in (inputs, targets, masks)), groups


@torch.inference_mode()
def validation_loss(model, tensors, batch_size):
    model.eval()
    total = 0.0
    for begin in range(0, len(tensors[0]), batch_size):
        x, y, mask = [t[begin : begin + batch_size] for t in tensors]
        total += float(masked_loss(model(x), y, mask)) * len(x)
    return total / len(tensors[0])


def train_models(directory, output, config: TrainingConfig):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configure_torch(config.seeds[0], config.cpu_threads)
    train, train_groups = load_training_tensors(directory, "train")
    validation, validation_groups = load_training_tensors(directory, "validation")
    if set(train_groups) & set(validation_groups):
        raise ValueError("Training and validation layouts overlap")
    manifest_hash = hashlib.sha256((Path(directory) / "manifest.jsonl").read_bytes()).hexdigest()
    model_config = {k: getattr(config, k) for k in ("channels", "residual", "obstacles")}
    architecture = (
        ARCHITECTURE
        if model_config == {"channels": 32, "residual": False, "obstacles": True}
        else RESEARCH_ARCHITECTURE
    )
    metadata = {
        "architecture": architecture,
        "model_config": model_config,
        "config": asdict(config),
        "dataset_manifest_sha256": manifest_hash,
        "train_examples": len(train[0]),
        "validation_examples": len(validation[0]),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summaries = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DistanceCNN(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(seed)
        best, stale, history = float("inf"), 0, []
        started = time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            order = torch.randperm(len(train[0]), generator=generator)
            total = 0.0
            for indices in order.split(config.batch_size):
                x, y, mask = [t[indices] for t in train]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(model(x), y, mask)
                if not torch.isfinite(loss):
                    raise ValueError("Training produced a nonfinite loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            val = validation_loss(model, validation, config.batch_size)
            if not np.isfinite(val):
                raise ValueError("Validation produced a nonfinite loss")
            row = {
                "epoch": epoch,
                "train_loss": total / len(train[0]),
                "validation_loss": val,
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            if val < best:
                best, stale = val, 0
                checkpoint = {
                    "architecture": architecture,
                    "model_config": model_config,
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": best,
                    "training_config": asdict(config),
                    "dataset_manifest_sha256": manifest_hash,
                    "development_layout_groups": train_groups + validation_groups,
                }
                temporary = output / f"seed_{seed}.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / f"seed_{seed}.pt")
            else:
                stale += 1
            (output / f"history_{seed}.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                f"seed={seed} epoch={epoch} train={row['train_loss']:.6f} "
                f"validation={val:.6f} seconds={row['seconds']:.2f}",
                flush=True,
            )
            if stale >= config.patience:
                break
        summary = {
            "seed": seed,
            "epochs": len(history),
            "best_validation_loss": best,
            "best_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"],
            "training_seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Train distance CNNs without reading test labels")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = (
        TrainingConfig(**tomllib.loads(args.config.read_text()))
        if args.config
        else TrainingConfig()
    )
    train_models(args.data, args.output, config)


if __name__ == "__main__":
    main()
