"""Evaluate the existing consistent heuristic only at requested cells.

This changes when scores are computed, not the priority or search loop.
Caches belong to one map and goal and are charged to the session using them.
"""

from learned_astar.search.prepared import PreparedAStar


class DemandScores(dict):
    def __init__(self, prepared, goal, index=None):
        super().__init__()
        target = prepared.index(goal)
        if index is not None and index.prepared is not prepared:
            raise ValueError("Landmark index belongs to a different prepared map")
        self.cells = prepared.cells
        self.goal = goal
        self.fields = (
            tuple((field, field[target]) for field in index.distances if field[target] >= 0)
            if index is not None
            else ()
        )

    def __missing__(self, cell):
        row, column = self.cells[cell]
        score = abs(row - self.goal[0]) + abs(column - self.goal[1])
        for field, to_goal in self.fields:
            to_cell = field[cell]
            if to_cell >= 0:
                difference = abs(to_cell - to_goal)
                if difference > score:
                    score = difference
        self[cell] = score
        return score


class DemandAStar(PreparedAStar):
    """Use the unchanged prepared search with a per-goal sparse score cache."""

    def __init__(self, prepared, goal, *, index=None):
        self.prepared = prepared
        self.goal = prepared.index(goal)
        self.primary = DemandScores(prepared, goal, index)
        self.ties = None
