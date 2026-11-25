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

## November 20, 2026: ordering limits and a bounded training pilot

Measured the minimum expansions attainable by changing tie order while keeping Manhattan primary priority. The exact-information reference reached the calculated floor on all 1,920 queries from 120 validation maps. Relative to ordinary Manhattan, the maximum paired expansion savings were 10.7% on random 32×32 maps, 5.3% on walls 32×32, 19.1% on random 64×64, and 4.3% on walls 64×64. These are count limits on the measured queries, not runtime gains. Exact detour scores also reached the floor, but subtracting Manhattan from learned predictions made ordering worse.

Generated 600 new maps, separated from all 6,200 earlier layout groups. Two branches continued the same 8-channel seed-7 checkpoint for 960 updates each. The control used the original data; the candidate mixed original examples with new wall and larger maps. Training took 41.5 seconds in total. The candidate reduced the shared distance-and-route validation loss from the control's 0.07376 to 0.02848, but this did not translate into a clear search-time benefit.

The runtime comparison completed 16,200 sessions and 1,042,200 query executions in about 3.4 minutes. At 128 starts per shared goal, the candidate's distance-tie time ratio to the frozen model was 0.999, with a 95% interval of 0.991–1.007. The learned detour rule was slower, with a ratio of 1.378. Neither passed the fixed validation gate. No additional seeds were trained, no new held-out test set was used, and the earlier model selection remains unchanged.

Validation completed:

- All 206 tests passed, including exhaustive 3×3 layout and endpoint checks for the expansion floor and safeguards for the runtime gate.
- Every timed method returned optimal costs and correct reachability. Repeated paths and counters agreed.
- Evidence checks verified complete repetitions, query prefixes, dataset separation, and frozen source and checkpoint hashes.
- Code style, formatting, and dependency checks passed. Both submitted proposal files remain unchanged.
- Both checkpoints, training histories, raw measurements, summaries, and the PNG/PDF comparison figure are retained.

This pilot supports stopping further tie-only training under the current setup. A separate future experiment could change the primary priority or predict intermediate goals, with path quality measured explicitly. Repeated-goal workloads should retain reverse BFS as a practical comparison. The present findings are validation evidence from one model seed.

## November 22, 2026: stronger primary heuristic, correctness step

Added a small classical landmark prototype using corner landmarks and exact BFS distances. Taking the maximum of Manhattan and landmark lower bounds preserves admissibility and consistency on the project's undirected unit-cost grids. This changes primary search priorities without relying on unverified model predictions. It is a control method, not a learned improvement.

All 210 tests passed, including exhaustive bound and path checks over every 3×3 obstacle layout and all 11,520 free endpoint pairs. Three hand-built illustrations retain maps, paths, source hashes, and preparation counts. On the wall example, search expansions fall from 25 to 14 with the same optimal 14-step path, but preparing the four landmarks requires 172 BFS visits. No runtime advantage is claimed.

No model was trained, no dataset was generated, and no validation or test benchmark was run. Existing algorithms, checkpoints, and submitted proposals remain unchanged. The next step is a small validation comparison with complete setup time, including landmark-only and landmark-plus-frozen-model controls.

## November 22, 2026: complete-time landmark validation

Measured five prepared methods on 40 existing policy-validation maps, balanced across random/wall and 32×32/64×64 families. Sessions used 1, 16, 64, or 128 distinct starts sharing a goal. Seven shuffled repetitions completed 5,600 sessions and 292,600 query executions in 120.8 seconds, including checking and recording evidence. Each session paid for graph preparation, landmark BFS or reverse BFS, CNN prediction when applicable, and full path output. The separately observed one-time model load was 16.11 ms.

At 128 starts, landmark-only paired runtime ratios to Manhattan were 1.246 on random 32, 0.480 on walls 32, 1.187 on random 64, and 0.371 on walls 64. The wall intervals were below one; random-map intervals were above one. The first measured workloads establishing a wall benefit were 64 starts for 32×32 and 16 starts for 64×64. Single-query landmark sessions remained roughly twice as slow as prepared Manhattan.

Adding the frozen CNN to landmarks increased paired complete time by 6.6%, 10.1%, 7.2%, and 21.4% in the four families. It also increased reachable expansions on both wall groups. Reverse BFS had the lowest mean 128-start time in every family. The measured speed benefit therefore comes from the classical landmark bound, not the model. This pilot does not justify more training of the current tie model.

Validation completed:

- All 213 tests passed; code style, formatting, and dependency checks passed.
- Every measured query returned an optimal cost or correct no-path result. All shorter sessions matched the corresponding saved 128-start path and counter prefixes.
- Verification checked 800 session medians, raw repetitions, full paths, source and weight hashes, selected arrays, and BFS labels.
- An independent checker recomputed 640 timing means and 240 paired ratio means from the raw sessions.
- Raw evidence, full paths, median sessions, report, and a visually checked PNG/PDF figure are retained. Both submitted proposals remain unchanged.

No model was trained and no fresh test set was used. Further work should distinguish fixed-goal reuse from changing-goal workloads before spending more training resources. These results cover only one model seed and 10 validation maps per family.

## November 23, 2026: changing goals and closing the experiment branch

Kept the same 40 validation maps and frozen 8-channel model. Each session contained 128 queries, with 1, 8, or 128 distinct goals. All methods reused their prepared graph and cached goal-specific state; landmark methods built the four-landmark index once per session. New goals paid for new score fields, predictions, or reverse-BFS policies. Five shuffled repetitions completed 3,000 sessions and 384,000 query executions in 337.1 seconds, including checks and evidence output. The separately observed one-time model load was 16.94 ms.

When every query had a different goal, landmark-only paired time ratios to Manhattan were 3.337 on random 32, 1.738 on walls 32, 3.506 on random 64, and 1.400 on walls 64. All intervals were above one. Adding the CNN made complete sessions 2.22–2.33 times as slow as landmark-only. Manhattan had the lowest mean time on random maps; reverse BFS had the lowest mean on walls. Landmarks still improved over Manhattan on walls with one or eight goals, but reverse BFS was faster there.

The breakdown identifies full goal-specific score construction as the main remaining landmark cost. Reusing the index does not make this preparation free. The CNN also needs a new prediction per goal. These findings apply to the measured implementation; an on-demand landmark implementation was not tested. Changing goal counts also changes endpoints and difficulty, so only within-workload method comparisons are paired.

Validation completed:

- All 216 tests passed, including per-goal cache reuse, changes of goal, unreachable queries, deterministic endpoints, and rejection of altered labels.
- Every path had optimal cost or a correct no-path result. Repeated paths and all search counters agreed.
- Verification independently regenerated endpoints and BFS labels, checked 76,800 saved query outcomes and 600 session medians, and cross-checked 240 latency and paired-ratio means.
- Code style, formatting, dependency, source-hash, and raw-evidence checks passed. The PNG/PDF comparison figure was visually checked.
- Both submitted proposals and every existing checkpoint remain unchanged. No training or fresh-test evaluation was performed.

This closes the bounded changing-goal diagnostic. With roughly one week remaining, the next work should consolidate the research conclusions, update the presentation, and rehearse the terminal demo rather than start another model sweep. Keep the learned-model findings separate from the classical landmark improvement.

## November 23, 2026: research consolidation, days 1–2

Added a research synthesis covering the learning task, search methods, fair timing, fresh-test findings, and later validation diagnostics. It separates useful learned ordering from complete-time performance and attributes the landmark benefit to the classical method. The original report remains available with a scope note; both submitted proposal files are unchanged.

Added a discussion guide explaining model inputs, training labels, hidden channels, optimal paths, runtime overhead, and the limits of additional tie training. It connects the main explanations to existing figures and records which comparisons support each conclusion. README navigation now starts with the synthesis and correctly identifies the reproduction command as the original v1 pipeline. The recording guide explicitly states that the current slide deck covers through v4 and needs the later findings added.

Checks completed:

- Compared 26 numerical expressions in the synthesis with archived JSON summaries and training histories, including paired intervals, width timings, the ordering floor, and landmark cost components.
- Checked all 57 local link targets in the five report and navigation files, plus Markdown whitespace.
- Confirmed both proposal hashes and the frozen 8-channel pilot checkpoint hash against their recorded values.
- Confirmed the edited report and guide files contain no AI attribution. No experiment data, source code, checkpoints, or slide files changed.

This was a documentation consolidation, so the algorithm test suite and benchmarks were not rerun. The latest implementation validation remains the 216 passing tests recorded for the changing-goal step. No new training, data generation, or held-out evaluation was needed.

The next planned deliverable is an updated presentation and a rehearsed terminal demonstration. Further research remains optional: start with a specific hypothesis, a matched control, and a bounded validation pilot. On-demand landmark scoring targets an observed preparation cost; learning to rank actual tied search cells would require a separate objective and would still face the measured tie-ordering limit. Neither is reported as implemented or successful.

## November 24, 2026: on-demand scores and a fresh larger-map test

Implemented per-goal score caches for Manhattan and landmark heuristics using the unchanged prepared A* search loop. Exhaustive 3×3 checks verify identical paths, costs, and all search counters against full-field evaluation. Additional tests cover larger sparse caches, goal separation, disconnected components, search limits, the validation gate, and archive corruption. Full-field landmark scores use floats and on-demand scores use integers; the study includes representation and lookup costs as well as avoided score evaluation.

The fixed validation study reused 40 maps and the previous changing-goal endpoints. It completed 3,000 sessions and 384,000 timed query executions in 198.9 seconds. At 128 distinct goals, on-demand landmark/full-field landmark ratios were 0.273 [0.264, 0.282] on walls 32 and 0.258 [0.248, 0.269] on walls 64. Both passed the predeclared gate before fresh-test generation.

The frozen implementation then evaluated 120 fresh maps: 20 random and 20 wall layouts at each of 32×32, 64×64, and 128×128. All 6,800 earlier layout groups and their rotations were excluded. Five shuffled repetitions completed 9,000 sessions and 1,152,000 timed query executions in 1,687.9 seconds, including checking and evidence writing in the stage wall time. No model was trained, and test outcomes did not change the methods or settings.

At 128 distinct goals, on-demand landmark time divided by on-demand Manhattan time was 0.516 [0.496, 0.535], 0.376 [0.351, 0.401], and 0.327 [0.305, 0.348] on the three wall sizes. Landmarks also beat full-field Manhattan and reverse BFS there. On random maps at that workload, on-demand Manhattan remained faster. With one or eight shared goals, reverse BFS had the lowest mean time in all six families. This is a classical implementation benefit, not a learned improvement.

Checks completed:

- All 222 tests passed in 20.24 seconds. Code style, formatting, and dependency checks passed.
- Every measured path had optimal cost or correct reachability. Full-field/on-demand pairs and repeated executions matched in complete paths and search counters.
- Full verification checked 307,200 saved outcomes, regenerated endpoints and BFS labels, verified source and raw-evidence hashes, and checked 2,400 median sessions.
- Independent arithmetic checked 300 validation and 450 test timing means and ratios. A separate standard-library checker also reproduced these values directly from the archives without local generated datasets.
- Both archives retain map arrays, workload labels, raw repetitions, full paths, source hashes, summaries, tables, and visually checked PNG/PDF figures.

The results and synthesis now distinguish this fresh-test confirmation from the earlier validation pilots. The measured timer includes preparation and query processing, but excludes final cache disposal and application startup. Peak memory and vectorized full-field controls remain unmeasured. Both submitted proposals and the existing checkpoints remain unchanged.

A next learning experiment could choose among the three methods from map features and the known query workload. It should use separate training data, charge feature and prediction costs, and compare against a simple hand-written selector. No such selector has been trained or evaluated yet. The presentation still needs the later research added.

## November 25, 2026: local method-selection pilot commands

Added a separate data pipeline for 48 training and 24 validation maps. It excludes all 6,920 earlier layout groups, including rotations, and collects three repeated timings for each optimal method on batches with 1, 8, or 128 goals. Completed maps can be retained after an interruption. Raw timings, full first-repetition paths, counters, input hashes, and map-level splits are saved for later checks.

Added a 99-parameter network that predicts the three log runtimes from map and workload features. Normalization uses training rows only. One seed and at most 200 updates keep the initial run small; validation loss selects the saved checkpoint. The comparison includes the three fixed methods, a training-selected fixed method, a predeclared hand-written rule, and an oracle reference. Initial selector costs combine measured feature and inference times with method medians, so they are explicitly reported as development estimates rather than live end-to-end test results.

Six new tests cover rotated exclusions, separate layouts, reproducible workloads, interrupted-run recovery, altered evidence, path checks, normalization, checkpoint reloads, and timing-cost addition. The full suite passed 228 tests, and code style and formatting checks passed. The submitted proposal files remain unchanged. Only tiny test runs were executed; full data collection and training are left for the local PyCharm run.

## November 25, 2026: intermediate-goal diagnostic

The local pilot completed and selected epoch 188. Its recombined mean validation session estimates were 9.527 ms for the learned selector and 9.506 ms for the fixed rule. Added commands to probe 16, 32, and 64 goals on 24 new development maps while retaining the checkpoint and original rule. The new layouts exclude all 6,992 earlier groups. Results are grouped by goal count, and no fitting occurs in this diagnostic.

All seven selector tests passed, including a small complete diagnostic that checks layout separation, unchanged checkpoint bytes, saved results, and rejection of altered inputs. Style and formatting checks passed. The full diagnostic remains pending local execution; previous experiment scripts and model files were not changed.

## Return to the learned-heuristic question: frontier pilot

The method-selection diagnostic completed locally and remains a separate extension. The next pilot returns to obstacle-aware distance prediction inside A*, following the submitted proposal. Added active-frontier sampling and an ordering loss based on queued path cost plus exact BFS distance. Equal teacher scores and unreachable cells are excluded. The labels rank route cost, not the number of expansions needed to finish the search.

The control and candidate start from the same saved eight-channel CNN and receive the same 128 training maps, batches, and 20 epochs. Only the candidate receives the additional frontier loss. Another 32 validation maps provide diagnostics; both branches retain their final epoch rather than selecting different checkpoints. The development comparison uses 48 separate 32×32 selector maps, one new goal and four starts per map, fresh inference on every query, and three repetitions. Baselines include Manhattan, BFS, three Weighted A* settings, and the unchanged parent CNN. These maps are development evidence, not a final test set.

All 233 tests passed. New checks cover every free endpoint pair on 3×3 layouts, strict teacher comparisons, stale-entry handling through agreement with ordinary A*, masked gradients, zero goal scores, reproducible sampling, and a complete small training/evaluation run. Source checks and proposal hashes passed, and a read-only check confirmed the default map counts and separation from the parent CNN's development groups. No existing experiment script, submitted proposal, or model weight file was changed. The full pilot remains pending local execution in PyCharm.
