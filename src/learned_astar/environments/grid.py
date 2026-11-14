"""Four-way grids with unit movement costs."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from random import Random

from learned_astar.config import require_finite, require_int

Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Grid:
    height: int
    width: int
    blocked: frozenset[Cell] = frozenset()

    def __post_init__(self) -> None:
        require_int(self.height, "height", minimum=1)
        require_int(self.width, "width", minimum=1)
        object.__setattr__(self, "blocked", frozenset(self.blocked))
        for cell in self.blocked:
            self.require_cell(cell)

    def require_cell(self, cell: Cell, *, free: bool = False) -> None:
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(index) is not int for index in cell)
        ):
            raise ValueError("A cell must be a (row, column) tuple of integers")
        row, column = cell
        if not (0 <= row < self.height and 0 <= column < self.width):
            raise ValueError(f"Cell {cell} is outside the grid")
        if free and cell in self.blocked:
            raise ValueError(f"Cell {cell} is blocked")

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        """Yield free neighbors in north, east, south, west order."""
        row, column = cell
        for next_cell in (
            (row - 1, column),
            (row, column + 1),
            (row + 1, column),
            (row, column - 1),
        ):
            next_row, next_column = next_cell
            if (
                0 <= next_row < self.height
                and 0 <= next_column < self.width
                and next_cell not in self.blocked
            ):
                yield next_cell

    def free_cells(self) -> Iterator[Cell]:
        for row in range(self.height):
            for column in range(self.width):
                if (row, column) not in self.blocked:
                    yield row, column

    @property
    def obstacle_fraction(self) -> float:
        return len(self.blocked) / (self.height * self.width)

    @classmethod
    def from_ascii(cls, rows: Sequence[str]) -> "Grid":
        """Read '.' as free and '#' as blocked, without trimming the rows."""
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Map rows must be nonempty and have equal widths")
        if any(symbol not in ".#" for row in rows for symbol in row):
            raise ValueError("Map rows may only contain '.' and '#'")
        blocked = frozenset(
            (r, c) for r, row in enumerate(rows) for c, symbol in enumerate(row) if symbol == "#"
        )
        return cls(len(rows), len(rows[0]), blocked)


def random_grid(height: int, width: int, obstacle_fraction: float, rng: Random) -> Grid:
    """Choose exactly round(fraction * height * width) distinct blocked cells."""
    require_int(height, "height", minimum=1)
    require_int(width, "width", minimum=1)
    require_finite(obstacle_fraction, "obstacle_fraction")
    if obstacle_fraction > 1:
        raise ValueError("obstacle_fraction must be at most 1")
    count = round(obstacle_fraction * height * width)
    blocked = frozenset(divmod(index, width) for index in rng.sample(range(height * width), count))
    return Grid(height, width, blocked)


def sample_query(grid: Grid, rng: Random) -> tuple[Cell, Cell]:
    """Choose different free endpoints, without requiring a route between them."""
    free = list(grid.free_cells())
    if len(free) < 2:
        raise ValueError("A query needs at least two free cells")
    start, goal = rng.sample(free, 2)
    return start, goal


def validate_path(grid: Grid, path: Sequence[Cell], start: Cell, goal: Cell, cost: int) -> None:
    """Raise ValueError if a returned route or its reported cost is wrong."""
    grid.require_cell(start, free=True)
    grid.require_cell(goal, free=True)
    require_int(cost, "cost")
    if not path or path[0] != start or path[-1] != goal:
        raise ValueError("Path must begin at the start and end at the goal")
    for cell in path:
        grid.require_cell(cell, free=True)
    for previous, current in pairwise(path):
        if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
            raise ValueError("Each path move must go to an adjacent cell")
    if cost != len(path) - 1:
        raise ValueError("Path cost must equal the number of moves")
