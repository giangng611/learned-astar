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

## November 16, 2026: follow-up research

Added a fixed experiment plan and 1,440 new development maps. Training used 800 random and 400 wall maps; validation used 160 random and 80 wall maps. Five 60-epoch runs compared direct distance prediction, a learned detour, and a no-obstacle control, then repeated the selected direct architecture with two additional seeds. Each new model has 10,033 parameters.

No candidate passed the validation path-quality gate. The predeclared fallback selected raw direct predictions for the final comparison. After freezing that choice, 600 fresh maps supplied 2,400 test queries and 204,000 timed method runs. A separate exploratory diagnostic measured local consistency without changing selection.

The seed-7 direct model reduced wall-map distance error from Manhattan's 18.67 steps to 7.17. Hiding its obstacle inputs increased error to 18.03. However, the selected three-seed configuration averaged 0.936 ms per wall query versus 0.337 ms for Manhattan A*, with 65.2% optimal paths. Learning obstacle distances did not produce a better search method in this round.

Validation completed:

- All 179 tests passed; code style, formatting, and dependency checks passed.
- The new datasets contain 2,040 verified layouts and 3,480 goal-distance maps, with no layout-group overlap with the original dataset.
- All 276,000 validation and test runs completed. Every method correctly handled reachable and unreachable queries; no search limits were reached.
- Evidence checks verified checkpoint hashes, paired query coverage, all timing repetitions, and the stored median summaries.
- Four figures, compact evidence, all five new checkpoints, and an independently chosen wall example are retained.
- Both submitted proposal files remain unchanged.

The follow-up report distinguishes improved distance prediction from search quality. The existing presentation still covers the first experiment; the new report and figures are available for its next revision.

## November 16, 2026: learning search preferences

Added a comparison of distance-only training and distance-plus-move training. The new objective accepts all neighboring steps on a shortest route. Three new 10,033-parameter models reused the mixed training data and ran for 60 epochs each. Checkpoint validation selected epochs 60, 56, and 59; total training time was 8.57 minutes.

Search now supports learned ordering of equal Manhattan priorities and an explicit FIFO control. A separate 240-map policy-validation set selected the route-trained tie family before another 600 test maps were generated. The comparison measured 19 methods, 3,360 validation and test queries, and five repetitions, totaling 319,200 timed runs.

On fresh reachable queries, the selected rule reduced paired expansions by 7.2% on random 32×32 maps and 4.7% on wall maps. All learned tie methods kept optimal path cost. The selected method remained slower overall: 0.682 ms versus Manhattan's 0.087 ms on random maps, and 0.917 ms versus 0.307 ms on walls. The 64×64 expansion interval did not establish an improvement.

Validation completed:

- All 190 tests passed, including every 3×3 layout and endpoint pair with deliberately wrong secondary scores, move-loss checks, training, and rejection of leaked policy-validation maps.
- Code style, formatting, and dependency checks passed.
- All 840 new layouts and distance maps were verified and separated from the 4,340 earlier layout groups.
- Every timed run completed with correct reachability; no search limits were reached. Evidence checks confirmed repetitions, medians, paired coverage, and checkpoint hashes.
- Three checkpoints, training histories, compact measurements, four PNG/PDF figures, and a fixed wall example are retained. The terminal tie mode agrees with the recorded example.
- Both submitted proposal files remain unchanged.

The third report separates improved move preferences, optimal path cost, reduced search effort, and the remaining runtime gap. The presentation still describes the first experiment and has not yet incorporated the two later rounds.
