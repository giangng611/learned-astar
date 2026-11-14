# Development log

## November 13, 2026: phases 0 and 1

Phase 0 added the Python package, configuration, random seeds, and recorded dependency versions. Python 3.14.3, NumPy 2.5.4, and PyTorch 2.15.0 passed a CPU convolution and gradient check using one thread.

Phase 1 added repeatable grids, path validation, BFS, Dijkstra, Manhattan A*, and Weighted A*. The priority queue methods share their search loop. Results distinguish success, no route, an expansion limit, and a time limit.

Validation completed:

- All 146 tests passed, including every 3×3 layout and every pair of different free endpoints, larger maps, reopening, stale entries, and limits.
- Code style, formatting, and dependency checks passed.
- The default 32×32 demo saved its map, paths, settings, and measurements as JSON.

With seed 4550, all six baseline settings returned cost 5. BFS and Dijkstra expanded 29 cells each; Manhattan A* and all three Weighted A* settings expanded 5 each. This single query checks the program and is not a benchmark result.

## Next: phase 2

Generate exact distance labels with reverse BFS and keep each map layout in one dataset split. Save seeds, layout identifiers, goals, and unreachable-cell masks. Verify labels and file loading on a small dataset before generating the full sets.
