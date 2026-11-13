# Search baselines

The first working version uses grids with four movement directions and a cost of one per move. A cell is written as `(row, column)`, starting from zero. Neighbors are checked north, east, south, then west. Start and goal must be free cells inside the grid.

## Algorithms

BFS uses an ordinary queue and gives a shortest path for these unit-cost grids. Dijkstra uses a priority queue with a heuristic of zero. Manhattan A* uses the horizontal and vertical distance to the goal. Weighted A* multiplies that distance by a chosen weight; the starting weights are 1.25, 1.5, and 2.0.

Dijkstra and both A* versions share the same search loop. It chooses the smallest `f = g + weight * h`, where `g` is the distance traveled and `h` is the remaining-distance estimate. Ties prefer a larger `g`, then the entry added first. Keeping these rules the same makes comparisons easier to interpret.

The shared loop also accepts a custom heuristic for the later model stage. This is only a connection point for now; no learned model has been added. Custom values must be finite and nonnegative. Those checks prevent invalid queue priorities, but they do not prove that a heuristic gives optimal paths.

## Better routes and stopping

The search keeps the best distance found for each cell. When a shorter route arrives, it updates the parent and adds a new queue entry, even if the cell was expanded earlier. Old entries are skipped when their distance no longer matches the best one. This allows the search to correct an earlier choice.

The goal must be removed from the queue before the search reports success. Finding it while checking a neighbor does not stop the search. The returned cost is the number of moves in the reconstructed path. BFS follows the same stopping rule with its ordinary queue.

There are four result states:

| State | Meaning |
| --- | --- |
| `found` | A route was found. The result includes its path and cost. |
| `no_path` | The queue became empty without reaching the goal. |
| `expansion_limit` | The allowed number of expansions was reached. |
| `time_limit` | The search reached its time limit. |

Only `found` includes a path. The other states use an empty path and no cost, so a partial attempt cannot be mistaken for a completed route. Invalid endpoints or invalid heuristic values raise an error instead.

If start equals goal, the valid answer is that one cell with cost zero and no expansions. This case succeeds even with a zero budget. An expansion budget counts cells whose neighbors are checked, so the goal can still be removed after the last allowed expansion. Time limits are checked between queue removals. They are soft limits: a single expansion or path reconstruction can finish slightly after the requested time.

## What gets counted

| Field | Meaning |
| --- | --- |
| `expanded` | Current queue entries whose neighbors were checked, including repeated expansions. It excludes stale entries and the final goal removal. |
| `unique_expanded` | Different cells whose neighbors were checked. |
| `generated` | Legal neighbors considered, including ones already seen. |
| `queue_insertions` | Entries added to the queue, including the initial start. |
| `stale_pops` | Old queue entries removed after a better distance was recorded. |
| `reexpanded` | Expansions beyond a cell's first expansion. |
| `max_frontier` | Largest number of queued entries, including any stale entries still waiting. |

Search time includes queue setup, the search, and path reconstruction. Input checks, map generation, result validation, and file writing are outside it. The later learned-search evaluation must also include input preparation and model inference in its total time. The single-query demo does not replace that evaluation.

## Map generation and checks

The random generator chooses exactly `round(obstacle_fraction * height * width)` distinct blocked cells. It then samples different free endpoints without checking whether a route exists. Maps with fewer than two free cells produce a clear error in the demo. The dataset stage will record those cases when building its manifest.

The tests compare Dijkstra and Manhattan A* with BFS and check that Weighted A* stays within `weight * optimal_cost`. They cover every 3×3 obstacle layout and every ordered pair of different free endpoints. Larger examples include random maps up to 64×64 and a corridor that forces a detour. Separate tests check queue ties, stale entries, reopening, time limits, and invalid inputs.

The demo checks every returned path for correct endpoints, free cells, adjacent moves, and matching cost. It saves failed and unreachable queries too. A good result on one map is useful for checking the code, but it is not evidence that a method performs better across the full dataset.
