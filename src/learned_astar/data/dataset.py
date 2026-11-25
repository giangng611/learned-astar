"""Generate portable NPZ files with a JSON manifest and no layout overlap."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from learned_astar.config import require_int
from learned_astar.environments import Grid, random_grid
from learned_astar.reproducibility import make_rng

from .labels import reverse_bfs, validate_labels


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 4550
    train_maps: int = 1200
    validation_maps: int = 200
    test_maps: int = 300
    extra_maps: int = 150
    size: int = 32
    goals_per_map: int = 2
    starts_per_map: int = 4
    train_wall_maps: int = 0
    validation_wall_maps: int = 0

    def __post_init__(self):
        for name in (
            "seed",
            "train_maps",
            "validation_maps",
            "test_maps",
            "extra_maps",
            "train_wall_maps",
            "validation_wall_maps",
        ):
            require_int(getattr(self, name), name)
        for name in ("size", "goals_per_map", "starts_per_map"):
            require_int(getattr(self, name), name, minimum=1)
        if self.size > 4096:
            raise ValueError("Grid size is too large")
        if any(n % 2 for n in (self.train_maps, self.validation_maps, self.test_maps)):
            raise ValueError("Main split counts must be even for equal density groups")
        if (
            self.train_maps
            + self.validation_maps
            + self.test_maps
            + self.train_wall_maps
            + self.validation_wall_maps
        ) == 0:
            raise ValueError("At least one main split must contain maps")


def occupancy(grid: Grid) -> np.ndarray:
    values = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for cell in grid.blocked:
        values[cell] = True
    return values


def layout_id(blocked: np.ndarray, *, rotations: bool = False) -> str:
    variants = [np.rot90(blocked, k) for k in range(4)] if rotations else [blocked]
    payloads = [f"{a.shape[0]}x{a.shape[1]}:".encode() + a.tobytes() for a in variants]
    return hashlib.sha256(min(payloads)).hexdigest()


def wall_grid(size: int, rng) -> Grid:
    blocked = set()
    column = rng.randrange(3, 6)
    while column < size - 2:
        gap = rng.randrange(size - 1)
        blocked.update((r, column) for r in range(size) if r not in (gap, gap + 1))
        column += rng.randrange(4, 8)
    grid = Grid(size, size, frozenset(blocked))
    turns = rng.randrange(4)
    values = np.rot90(occupancy(grid), turns)
    return Grid(size, size, frozenset(map(tuple, np.argwhere(values).tolist())))


def _specs(config):
    for split, count in [
        ("train", config.train_maps),
        ("validation", config.validation_maps),
        ("test", config.test_maps),
    ]:
        for density in (0.1, 0.25):
            yield split, config.size, density, count // 2, "random"
    yield "train", config.size, None, config.train_wall_maps, "walls"
    yield "validation", config.size, None, config.validation_wall_maps, "walls"
    for split, size, density, kind in [
        ("density_40", config.size, 0.4, "random"),
        ("size_16", 16, 0.25, "random"),
        ("size_64", 64, 0.25, "random"),
        ("walls", config.size, None, "walls"),
    ]:
        yield split, size, density, config.extra_maps, kind


def generate_dataset(output: str | Path, config: DatasetConfig, *, exclude_groups=()) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "maps").mkdir()
    rng = make_rng(config.seed)
    seen = set(exclude_groups)
    excluded_count = len(seen)
    records = []
    duplicates = 0
    for split, size, density, count, kind in _specs(config):
        for _ in range(count):
            for _attempt in range(10000):
                map_seed = rng.randrange(2**32)
                local = make_rng(map_seed)
                grid = (
                    wall_grid(size, local)
                    if kind == "walls"
                    else random_grid(size, size, density, local)
                )
                blocked = occupancy(grid)
                group_id = layout_id(blocked, rotations=True)
                if group_id not in seen:
                    seen.add(group_id)
                    break
                duplicates += 1
            else:
                raise ValueError("Cannot generate enough distinct layouts for these settings")
            identifier = layout_id(blocked)
            record = {
                "layout_id": identifier,
                "group_id": group_id,
                "split": split,
                "map_seed": map_seed,
                "kind": kind,
                "height": size,
                "width": size,
                "requested_density": density,
                "actual_density": grid.obstacle_fraction,
            }
            free = list(grid.free_cells())
            training = split in ("train", "validation")
            needed = config.goals_per_map if training else config.starts_per_map + 1
            if len(free) < needed:
                record.update(status="insufficient_free_cells", file=None)
                records.append(record)
                continue
            if training:
                goals = local.sample(free, config.goals_per_map)
                starts = []
            else:
                selected = local.sample(free, config.starts_per_map + 1)
                goals, starts = selected[:1], selected[1:]
            labels = np.stack([reverse_bfs(grid, goal) for goal in goals])
            filename = f"maps/{identifier}.npz"
            np.savez_compressed(
                output / filename,
                blocked=blocked,
                goals=np.asarray(goals, dtype=np.int32),
                starts=np.asarray(starts, dtype=np.int32).reshape(-1, 2),
                distances=labels,
            )
            record.update(
                status="ready",
                file=filename,
                sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                reachable_fractions=[float(np.count_nonzero(d >= 0) / len(free)) for d in labels],
            )
            records.append(record)
    manifest = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    (output / "manifest.jsonl").write_text(manifest)
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "duplicates_rejected": duplicates,
        "excluded_layout_groups": excluded_count,
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "label_markers": {"blocked": -2, "unreachable": -1},
    }
    # This file marks a completed generation run; partial directories cannot load.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_manifest(directory: str | Path, split: str | None = None) -> list[dict]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema")
    raw = (directory / "manifest.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    records = [json.loads(line) for line in raw.splitlines()]
    return [r for r in records if split is None or r["split"] == split]


def load_record(directory: str | Path, record: dict) -> dict[str, np.ndarray]:
    if record["status"] != "ready":
        raise ValueError("Record does not contain enough free cells")
    directory = Path(directory).resolve()
    path = (directory / record["file"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Dataset file must be inside the dataset directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Map checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_dataset(directory: str | Path) -> dict:
    records = read_manifest(directory)
    seen = set()
    checked_goals = 0
    for record in records:
        if record["group_id"] in seen:
            raise ValueError("Duplicate layout or rotated layout across records")
        seen.add(record["group_id"])
        if record["status"] != "ready":
            continue
        data = load_record(directory, record)
        blocked = data["blocked"]
        if blocked.shape != (record["height"], record["width"]) or blocked.dtype != np.bool_:
            raise ValueError("Invalid occupancy grid")
        if layout_id(blocked) != record["layout_id"]:
            raise ValueError("Layout ID mismatch")
        if layout_id(blocked, rotations=True) != record["group_id"]:
            raise ValueError("Layout group ID mismatch")
        grid = Grid(*blocked.shape, frozenset(map(tuple, np.argwhere(blocked).tolist())))
        goals = [tuple(map(int, g)) for g in data["goals"]]
        starts = [tuple(map(int, s)) for s in data["starts"]]
        if len(set(goals)) != len(goals) or len(set(starts)) != len(starts):
            raise ValueError("Endpoints must be distinct")
        if set(goals) & set(starts):
            raise ValueError("A test start equals the goal")
        for start in starts:
            grid.require_cell(start, free=True)
        if data["distances"].shape != (len(goals), *blocked.shape):
            raise ValueError("Wrong label dimensions")
        for goal, labels in zip(goals, data["distances"], strict=True):
            validate_labels(grid, goal, labels)
            checked_goals += 1
    return {
        "maps": len(records),
        "verified_goal_maps": checked_goals,
        "unique_layout_groups": len(seen),
    }
