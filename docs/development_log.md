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

## November 17, 2026: runtime and goal reuse

Kept the three route-trained models fixed and profiled the original search on policy-validation maps. Added prepared search using integer cell IDs, stored neighbors, and score checks performed once per planner. Both Manhattan and learned ties receive the same implementation changes. A reverse-BFS policy stores next steps and returns actual paths for repeated starts.

The fixed protocol generated 300 new maps after excluding all 5,180 earlier layout groups. Sessions used 1, 4, 16, or 64 distinct starts sharing one map and goal. All setup costs were included, with five shuffled repetitions on one CPU thread. The experiment completed 64,500 sessions and 1,279,500 query executions.

At 64 queries per session, prepared learned search averaged 0.0473 ms per query on random 32×32 maps and 0.1508 ms on wall maps. It beat original Manhattan but remained slower than prepared Manhattan, which averaged 0.0402 and 0.1439 ms. Reverse BFS with stored next steps averaged 0.0127 and 0.0137 ms and was fastest among the measured methods in this workload. The small learned search-phase saving did not cover its extra setup cost.

Validation completed:

- All 195 tests passed. Exhaustive small-grid checks compare exact paths and every search counter between original and prepared implementations, including incorrect tie scores.
- All measured methods returned optimal paths and correct reachability. Prepared searches matched their original counterparts on every measured query.
- Independent checks recomputed medians and paired ratios, verified query coverage, and confirmed frozen code, data, and model hashes.
- Code style, formatting, and dependency checks passed. Both proposal files remain unchanged.
- The updated 18-slide presentation covers all four rounds, with editable tables and charts, speaker notes, and a revised recording guide. The original deck is retained separately.

The latest report and figures explain both the implementation speedup and the remaining gap against equally optimized baselines. The terminal demo uses fresh predictions; the reused-session findings come from the recorded benchmark.

## November 18, 2026: smaller route models

Trained 8- and 4-channel CNNs with three seeds each, keeping the five convolution blocks, receptive field, data, route objective, and search rule fixed. All six runs completed 60 epochs, taking 16.83 minutes in total. The existing 16-channel checkpoints remained unchanged and were remeasured as controls.

The experiment generated 120 new policy-validation maps and 300 fresh test maps, excluding all 5,480 earlier layout groups. Validation selected 4 channels using complete query time, correct optimal paths, and mean expansion ratios below Manhattan in both map families. Its wall advantage on validation was very small. The selection was kept when the test did not confirm that advantage.

On the new random and wall 32×32 test maps, the 8-channel model reduced mean time by 35.7% and 26.8% relative to the 16-channel control. It retained paired expansion reductions of 9.1% and 3.8% versus Manhattan. The 4-channel model reduced mean time by 51.0% and 37.2%, but showed no clear wall expansion improvement and expanded 5.2% more cells on 64×64 maps. Both smaller widths remained slower overall than Manhattan. These timings include a fresh prediction per query and are separate from the earlier reuse experiment.

Validation completed:

- All 198 tests passed, including selection safeguards, shared training controls, and map-level paired comparisons.
- All 84,000 timed validation and test runs returned optimal paths and correct reachability. Repeated counters and query endpoints matched.
- Evidence checks verified raw repetitions, medians, checkpoint hashes, separated layouts, and independent paired runtime means.
- Both smaller widths loaded into the terminal and handled the detour and unreachable examples correctly.
- The six new checkpoints, training histories, raw measurements, report, and PNG/PDF comparison figure are retained. Both submitted proposal files remain unchanged.

The result identifies a practical tradeoff: 8 channels retain more of the useful search ordering, while 4 channels minimize prediction cost among the tested widths. The formal validation selection remains 4 channels; no model is reselected using test results.

## November 18, 2026: smaller models with shared predictions

Combined the nine existing 16-, 8-, and 4-channel checkpoints with prepared search. No new training was needed. The fixed comparison used 300 new maps, excluding all 5,900 earlier layout groups, and sessions with 1, 4, 16, 64, or 128 distinct starts sharing a goal. All preparation costs were included. The run completed 97,500 sessions and 4,153,500 query executions in 8.57 minutes, including measurement checks and writing raw evidence.

At 64 starts, every learned width remained slower than equally prepared Manhattan. The 8-channel model's paired time ratios were 1.099 on random 32×32 maps and 1.029 on wall maps. At 128 starts, its wall ratio was 1.001 with a 95% interval of 0.981–1.019, consistent with similar speed but not an established advantage. Reverse BFS with stored next steps had the lowest mean complete time in all five 64-start conditions.

The 8-channel model still reduced paired reachable expansions by 5.9% on random 32×32 maps and 4.2% on walls at 64 starts. On 64×64 maps, 8 and 4 channels expanded 2.8% and 5.9% more cells. Smaller inference and reuse narrow the gap, but neither establishes an overall runtime win. The previous validation choice remains unchanged.

Validation completed:

- All 202 tests passed; code style, formatting, and dependency checks passed.
- All timed methods returned optimal paths and correct reachability. Repeated paths and counters agreed, and original and prepared Manhattan matched exactly.
- Evidence verification checked all methods, repetitions, endpoints, batch prefixes, model hashes, and data separation.
- An independent check recomputed 19,500 session medians and 3,600 summary means from the raw records.
- Raw evidence, per-seed summaries, and two PNG/PDF figures are retained. Both figures were visually checked, and both submitted proposal files remain unchanged.

The next proposed step is a limited 8-channel training pilot using difficult and larger training layouts, after checking remaining tie-ordering headroom on validation. A validation improvement should be required before expanding to three seeds and another fresh test set.
