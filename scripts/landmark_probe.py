"""Small correctness probe for an obstacle-aware, admissible primary heuristic.

This is a classical landmark control, not a learned method or a timing benchmark.
Landmarks depend only on the map. All BFS preparation must be charged in later timing.
"""

import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from learned_astar.environments import Grid, validate_path
from learned_astar.search import bfs
from learned_astar.search.prepared import PreparedAStar, PreparedGrid


def corner_landmarks(grid):
    """Snap each corner to the nearest free cell; break ties by row then column."""
    free = tuple(grid.free_cells())
    if not free:
        return ()
    corners = ((0, 0), (0, grid.width - 1), (grid.height - 1, 0), (grid.height - 1, grid.width - 1))
    chosen = []
    for row, column in corners:
        cell = min(free, key=lambda c: (abs(c[0] - row) + abs(c[1] - column), c))
        if cell not in chosen:
            chosen.append(cell)
    return tuple(chosen)


@dataclass(frozen=True)
class LandmarkIndex:
    prepared: PreparedGrid
    landmarks: tuple
    distances: tuple
    preprocess_expanded: int

    @classmethod
    def build(cls, prepared, landmarks=None):
        selected = corner_landmarks(prepared.grid) if landmarks is None else tuple(landmarks)
        indices = tuple(prepared.index(cell) for cell in selected)
        if len(set(indices)) != len(indices):
            raise ValueError("Landmarks must be distinct free cells")
        fields = []
        expanded = 0
        for source in indices:
            distances = [-1] * len(prepared.cells)
            distances[source] = 0
            queue = deque([source])
            while queue:
                cell = queue.popleft()
                expanded += 1
                for neighbor in prepared.neighbors[cell]:
                    if distances[neighbor] < 0:
                        distances[neighbor] = distances[cell] + 1
                        queue.append(neighbor)
            fields.append(tuple(distances))
        return cls(prepared, selected, tuple(fields), expanded)

    def primary(self, goal):
        """Return max(Manhattan, |d(L, v) - d(L, goal)|) for finite pairs.

        On this undirected unit-cost grid each term is admissible and consistent.
        Unreachable landmark pairs contribute nothing; -1 is never a distance.
        """
        target = self.prepared.index(goal)
        scores = [abs(r - goal[0]) + abs(c - goal[1]) for r, c in self.prepared.cells]
        for distances in self.distances:
            to_goal = distances[target]
            if to_goal < 0:
                continue
            for cell, to_cell in enumerate(distances):
                if to_cell >= 0:
                    scores[cell] = max(scores[cell], abs(to_cell - to_goal))
        return tuple(scores)


def illustrations():
    examples = (
        ("open", ["......."] * 7, (1, 1), (1, 5)),
        ("wall_detour", ["...#..."] * 6 + ["......."], (1, 1), (1, 5)),
        ("disconnected", ["...", "###", "..."], (0, 0), (2, 2)),
    )
    records = []
    for name, rows, start, goal in examples:
        grid = Grid.from_ascii(rows)
        prepared = PreparedGrid.build(grid)
        index = LandmarkIndex.build(prepared)
        scores = index.primary(goal)
        baseline = PreparedAStar(prepared, goal)
        guided = PreparedAStar(prepared, goal, primary=scores)
        reference = bfs(grid, start, goal)
        methods = {}
        for label, planner in (("manhattan", baseline), ("landmark", guided)):
            result = planner.search(start)
            if (result.status, result.cost) != (reference.status, reference.cost):
                raise ValueError("Search disagrees with BFS")
            if result.path:
                validate_path(grid, result.path, start, goal, result.cost)
            methods[label] = {
                "status": result.status.value,
                "cost": result.cost,
                "expanded": result.stats.expanded,
                "primary_at_start": planner.primary[prepared.index(start)],
                "path": result.path,
            }
        records.append(
            {
                "name": name,
                "rows": rows,
                "start": start,
                "goal": goal,
                "landmarks": index.landmarks,
                "landmark_bfs_expansions": index.preprocess_expanded,
                "stored_distance_entries": sum(map(len, index.distances)),
                "methods": methods,
            }
        )
    sources = (
        Path(__file__),
        Path("src/learned_astar/search/prepared.py"),
        Path("src/learned_astar/search/core.py"),
        Path("src/learned_astar/environments/grid.py"),
    )
    return {
        "scope": "Three hand-built illustrations; not a validation or test benchmark",
        "training_runs": 0,
        "timing_claim": False,
        "source_sha256": {
            str(path.relative_to(Path.cwd()) if path.is_absolute() else path): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
        "examples": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/landmark_step1.json"))
    args = parser.parse_args()
    result = illustrations()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for example in result["examples"]:
        base, new = (example["methods"][key] for key in ("manhattan", "landmark"))
        print(
            f"{example['name']}: cost={new['cost']}, search expansions "
            f"{base['expanded']} -> {new['expanded']}, "
            f"landmark BFS expansions={example['landmark_bfs_expansions']}"
        )
    print(f"Illustrations saved to {args.output}; no timing or generalization claim.")


if __name__ == "__main__":
    main()
