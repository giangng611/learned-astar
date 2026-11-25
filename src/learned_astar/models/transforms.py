"""Grid-specific distance transformations shared by experiments and the demo."""

import numpy as np


def manhattan_field(shape, goal):
    rows, columns = np.indices(shape)
    return np.abs(rows - goal[0]) + np.abs(columns - goal[1])


def parity_distances(prediction, goal):
    manhattan = manhattan_field(prediction.shape, goal)
    return manhattan + 2 * np.rint(np.maximum(prediction - manhattan, 0) / 2)
