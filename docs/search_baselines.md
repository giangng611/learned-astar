# Search baselines

The grid uses zero-based `(row, column)` coordinates, four movement directions, and a cost of one per move. Neighbors are checked north, east, south, then west. Start and goal must be free cells inside the grid.

## Methods

BFS uses an ordinary queue. Dijkstra uses a heuristic of zero, Manhattan A* uses horizontal and vertical distance to the goal, and Weighted A* multiplies that distance by 1.25, 1.5, or 2.0. The priority queue methods share one loop, choosing the smallest `f = g + weight * h`. Here, `g` is distance traveled and `h` estimates distance remaining. Ties prefer larger `g`, then earlier insertion.

A shorter route updates the cell's distance and parent and adds a new queue entry, even if the cell was expanded before. Old entries are skipped when their distance no longer matches the best one. Search succeeds when the goal is removed from the queue, not when first discovered. Cost is the number of moves in the returned path.

The shared loop accepts a custom heuristic for the later model stage. Its values must be finite and nonnegative, but those checks do not guarantee optimal paths. No learned model is connected yet.

## Results and limits

| State | Meaning |
| --- | --- |
| `found` | A route was found; path and cost are included. |
| `no_path` | The queue became empty without reaching the goal. |
| `expansion_limit` | The allowed number of expansions was reached. |
| `time_limit` | The time limit was reached. |

Other than `found`, results have an empty path and no cost. Invalid endpoints or heuristic values raise an error. A valid start equal to the goal returns one cell, cost zero, and no expansions, even with a zero budget.

An expansion budget counts cells whose neighbors are checked; removing the goal uses no expansion. Time limits are checked between queue removals, so an expansion or path reconstruction may finish slightly after the limit.

## Counts and timing

| Field | Definition |
| --- | --- |
| `expanded` | Entries whose neighbors were checked, including repeat expansions; excludes stale entries and goal removal. |
| `unique_expanded` | Different cells expanded. |
| `generated` | Legal neighbors considered, including repeats. |
| `queue_insertions` | Entries added, including the initial start. |
| `stale_pops` | Old entries removed after a better distance was recorded. |
| `reexpanded` | Expansions beyond each cell's first. |
| `max_frontier` | Largest queue size, including stale entries still waiting. |

Timing includes queue setup, search, and path reconstruction. Input validation, map generation, path checks, and file writing are outside it. Learned-search evaluation will also include input preparation and model inference.

## Maps and validation

Map generation selects exactly `round(obstacle_fraction * height * width)` distinct blocked cells. It samples different free endpoints without requiring a route. Fewer than two free cells produces a clear demo error; the dataset stage will record such cases.

Tests compare Dijkstra and Manhattan A* with BFS and check the Weighted A* bound of `weight * optimal_cost`. They cover all 3×3 layouts and ordered pairs of different free endpoints, random maps up to 64×64, detours, ties, reopening, stale entries, limits, and invalid inputs. The demo also checks endpoints, obstacles, adjacent moves, and path cost, and retains failed and unreachable queries.
