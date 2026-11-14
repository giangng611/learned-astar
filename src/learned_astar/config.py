"""Read and check settings shared by the demo and later experiments."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


def require_int(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def require_finite(value: object, name: str, *, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be finite and at least {minimum}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    height: int = 32
    width: int = 32
    obstacle_fraction: float = 0.25
    seed: int = 4550
    weights: tuple[float, ...] = (1.25, 1.5, 2.0)
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        require_int(self.height, "height", minimum=1)
        require_int(self.width, "width", minimum=1)
        require_int(self.seed, "seed")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        require_int(self.cpu_threads, "cpu_threads", minimum=1)
        require_finite(self.obstacle_fraction, "obstacle_fraction")
        if self.obstacle_fraction > 1:
            raise ValueError("obstacle_fraction must be at most 1")
        if not isinstance(self.weights, (list, tuple)) or not self.weights:
            raise ValueError("weights must be a nonempty list or tuple")
        for weight in self.weights:
            require_finite(weight, "weight", minimum=1.0)
        if len(set(self.weights)) != len(self.weights):
            raise ValueError("weights must not contain duplicates")
        object.__setattr__(self, "weights", tuple(self.weights))


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    unknown = values.keys() - ExperimentConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return ExperimentConfig(**values)
