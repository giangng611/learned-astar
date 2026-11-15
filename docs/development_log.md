# Development log

## November 13, 2026: phases 0 and 1

Phase 0 added the Python package, configuration, random seeds, and recorded dependency versions. Python 3.14.3, NumPy 2.5.4, and PyTorch 2.15.0 passed a CPU convolution and gradient check using one thread.

Phase 1 added repeatable grids, path validation, BFS, Dijkstra, Manhattan A*, and Weighted A*. The priority queue methods share their search loop. Results distinguish success, no route, an expansion limit, and a time limit.

Validation completed:

- All 146 tests passed, including every 3×3 layout and every pair of different free endpoints, larger maps, reopening, stale entries, and limits.
- Code style, formatting, and dependency checks passed.
- The default 32×32 demo saved its map, paths, settings, and measurements as JSON.

With seed 4550, all six baseline settings returned cost 5. BFS and Dijkstra expanded 29 cells each; Manhattan A* and all three Weighted A* settings expanded 5 each. This single query checks the program and is not a benchmark result.

## November 15, 2026: phases 2–5

Phase 2 added reverse BFS labels, map-level splits, rotation-group checks, file checksums, and independent label verification. The full dataset contains 2,300 distinct layouts and 3,700 map-goal distance labels. A separate pilot checked the data and training workflow before the full runs.

Phase 3 added the 38,497-parameter CNN and masked training loss. Seeds 7, 17, and 27 each ran for 40 epochs; validation selected epochs 36, 36, and 40. Training took 18.73 minutes on CPU in total. Settings, histories, dataset provenance, and selected checkpoints are saved under `artifacts/`.

Phase 4 connected frozen predictions to the shared search loop and measured 3,600 held-out queries across nine method settings and five repetitions. All 162,000 timed runs completed in an evaluation lasting about 5.23 minutes. Every method handled reachable and unreachable queries correctly, with no time or expansion limits reached. The main learned result averaged 2.529 ms and 70.6 expansions, compared with 0.167 ms and 48.2 for Manhattan A*. The learned method therefore did not improve speed or search effort in this experiment.

Phase 5 added the terminal map shell, reusable demo maps, result figures, a 3,300-word report, and a 15-slide presentation with speaker notes. The recording guide includes a short demo and suggested section handoffs.

Validation completed:

- All 170 tests passed, including data integrity, model loading, leakage rejection, inference timing, repeated-result summaries, and shell interaction.
- Code style, formatting, and dependency checks passed.
- The stored results contain 32,400 query-method summaries and 900 map records; saved checkpoint hashes match the evaluated files.
- The final checkpoints ran successfully through the terminal comparison.
- Both submitted proposal files match their original contents.

The report and presentation describe the negative performance result, path-quality tradeoff, and limits of the tested model. The remaining group work is to rehearse, record, review, and submit the presentation.
