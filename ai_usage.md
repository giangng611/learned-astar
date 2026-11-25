# AI usage

## Implementation and troubleshooting

Codex was used for the following technical work:

- Setting up the Python package, dependencies, configurations, and random seeds.
- Implementing grid generation, path checks, BFS, Dijkstra, Manhattan A*, Weighted A*, and learned A*.
- Implementing reverse BFS labels, layout separation, checksums, and dataset verification.
- Implementing the distance CNN, masked training loss, validation checkpoint selection, and repeated training runs.
- Building the evaluation pipeline, including inference timing, path-quality checks, repeated measurements, map-level bootstrap intervals, and result figures.
- Adding the interactive terminal shell, ASCII map input, model loading, and saved query output.
- Creating and running correctness tests, investigating implementation errors, and resolving dependency and formatting issues.

Checks include 170 passing tests, verification of 2,300 layouts and 3,700 distance maps, code style and formatting checks, and a dependency check. Training used three seeds with separately selected validation checkpoints. The experiment settings, training histories, selected models, and measured results are retained with the project.

## Follow-up implementation

Codex was also used to implement mixed random and wall datasets, direct and residual model variants, the no-obstacle control, exact-distance comparisons, and distance rounding. Technical checks cover fresh-test separation, validation-only selection, local consistency diagnostics, timing summaries, and model provenance. Troubleshooting included making repeated CPU thread configuration safe when running several training jobs in one process. The completed model from the interrupted sequence was retained, and the remaining jobs resumed without changing their settings.

The follow-up completed five training runs, verified 2,040 new layouts and 3,480 distance maps, and measured 276,000 validation and test runs. All 179 tests passed. Model files, training histories, and compact measurement evidence are retained for review and reproduction.

## Search-learning implementation

Codex was used to implement the neighboring-move objective, learned tie ordering with Manhattan priority, FIFO and exact-tie controls, separate policy-validation maps, paired objective comparisons, and terminal support. Technical checks covered optimality under incorrect secondary scores, masked gradients, multiple valid moves, frozen checkpoint checks, and data separation.

This round completed three 60-epoch training runs and verified 840 new layouts. All 319,200 timed validation and test runs completed, and all 190 tests passed. Checkpoints, histories, compact evidence, and result figures are retained.

## Runtime implementation

Codex was used to profile repeated search work, implement prepared A* and the reverse-BFS policy, and build the session benchmark. Technical checks covered frozen map-goal state, prediction reuse, exact agreement with reference paths and counters, source hashes, and independent timing summaries.

The experiment verified 300 new layouts and completed 64,500 timed sessions containing 1,279,500 query executions. All 195 tests passed. The three trained model files were unchanged; compact evidence, diagnostic profiles, and figures are retained.

## Smaller-model implementation

Codex was used to implement the width-only experiment, runtime-based validation selection, matched seed comparisons, evidence checks, and checkpoint archiving. Technical checks cover unchanged training settings, correct optimal paths, map-level weighting, and rejection of selection using test results.

Six new 60-epoch training runs completed. The experiment verified 420 new maps and 84,000 timed runs. All 198 tests passed, and both smaller widths loaded into the terminal demo. Raw measurements, all six checkpoints, and training histories are retained.

## Smaller-model reuse implementation

Codex was used to connect the frozen model widths to prepared search, prevent checkpoint collisions across seeds and widths, stream benchmark evidence, and implement paired runtime analysis and independent verification. Technical checks cover prediction reuse, complete query coverage, batch prefixes, raw checksums, and unchanged paths and counters.

This round reused all nine checkpoints without retraining. It verified 300 new maps and 97,500 sessions containing 4,153,500 query executions. All 202 tests passed. An independent check recomputed 19,500 session medians and 3,600 summary means. Raw evidence, per-seed summaries, and figures are retained.

## Ordering-limit and training-pilot implementation

Codex was used to implement the expansion-floor calculation, exact and learned detour ordering, matched continued-training branches, and a fixed runtime gate. Technical checks cover exhaustive small-grid cases, equivalent score ordering, data separation, checkpoint provenance, repeated path agreement, and whole-map runtime comparisons.

The experiment verified 600 new maps and audited 1,920 validation queries. Two training branches completed 960 updates each in 41.5 seconds total. The runtime stage measured 16,200 sessions containing 1,042,200 query executions, and all 206 tests passed. Neither candidate passed the gate, so additional seeds and a new test were not run. Both checkpoints, training histories, raw measurements, and figures are retained.

## Landmark correctness prototype

Codex was used to implement a classical landmark lower-bound prototype, handle disconnected landmark distances, and add correctness checks against BFS. Tests verify admissibility, consistency, optimal paths, deterministic corner selection, and reuse across goals. The project passed 210 tests, including all 11,520 free endpoint pairs across 3×3 obstacle layouts. Three illustrative cases retain paths, source hashes, and preparation counts. This step involved no training or runtime benchmark.

## Landmark timing implementation

Codex was used to implement complete-session timing, fixed validation-map selection, path and prefix verification, map-level comparisons, and an independent summary checker. The run measured 5,600 sessions containing 292,600 query executions on 40 existing validation maps. Checks confirmed 800 session medians, 640 timing means, and 240 paired means. All 213 tests passed. No model was trained; model loading was measured separately from sessions using resident weights. Raw timings, full paths, source hashes, summaries, and figures are retained.

## Changing-goal implementation

Codex was used to implement per-goal caches, deterministic changing-goal workloads, complete-session timing, and independent endpoint and BFS-label verification. The experiment reused 40 validation maps and measured 3,000 sessions containing 384,000 query executions. Checks covered 76,800 saved query outcomes, 600 session medians, and 240 cross-checked summary means. All 216 tests passed. No model was trained; existing checkpoints were unchanged. Raw timings, paths, workload labels, source hashes, and figures are retained.

## On-demand heuristic implementation

Codex was used to implement per-goal sparse heuristic caches, full-field/on-demand comparisons, a validation gate, and fresh-map generation with layout exclusions. Technical checks compare complete paths and every search counter, verify BFS costs independently, retain timing components and score-entry counts, and check paired summaries. The implementation reuses the existing A* search loop. All 222 tests passed, including exhaustive small-grid cases and rejection of altered archive means or raw evidence. No neural weights were changed.

The validation and test runs completed 12,000 sessions and 1,536,000 timed query executions on 40 existing validation maps and 120 fresh maps. Fresh layouts exclude all 6,800 earlier groups. Checks covered 307,200 saved query outcomes, 2,400 median sessions, and 750 timing means and ratios. A separate standard-library checker verifies archived map and raw-evidence hashes and timing arithmetic without the original dataset directories.
