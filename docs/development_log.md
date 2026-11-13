# Development log

## November 13, 2026: phases 0 and 1

The project now runs from the command line. Phase 0 set up the Python package, configuration file, random seeds, local environment, and package version record. The environment check passed with Python 3.14.3, NumPy 2.5.4, and PyTorch 2.15.0 on an Apple Silicon Mac. It ran a small convolution and calculated its gradients on the CPU using one thread.

Phase 1 added the grid rules, repeatable obstacle maps, path checks, BFS, Dijkstra, Manhattan A*, and Weighted A*. The priority queue algorithms share their search code. The results distinguish a found path, no route, an expansion limit, and a time limit.

The checks completed for these phases were:

- 146 automated tests passed, including every 3×3 obstacle layout and every pair of different free endpoints.
- Larger random maps, detours, equal start and goal, invalid inputs, reopening, stale queue entries, and search limits were checked.
- Code style and formatting checks passed, and the installed packages had no dependency conflicts.
- The command-line demo ran on the default 32×32 map and saved its settings, map, paths, and counts as JSON.

For the default seed of 4550, all six baseline settings returned a path of cost 5. BFS and Dijkstra expanded 29 cells each. Manhattan A* and the three Weighted A* settings expanded 5 cells each. This is one short query used to check the program. It is not a benchmark result or evidence about the learned heuristic.

I used Codex to help set up the project, implement the baselines, write tests and documentation, and run the checks. The changes are grouped into local commits for my review. They have not been pushed.

## Next: phase 2

The next step is to build the dataset and exact distance labels. Reverse BFS will calculate the distance from each reachable cell to a chosen goal. Whole map layouts will stay in one split so the same layout cannot appear in both training and testing. The saved data will include seeds, layout identifiers, goals, unreachable cells, and split information.

I will start with a small dataset to check labels and file loading before generating the planned training, validation, and test sets. Model training and full timing experiments will follow after those checks pass.
