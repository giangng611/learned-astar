# Step 2: complete-time landmark comparison

The [correctness step](primary_heuristic_step1.md) showed that landmark bounds can reduce search while retaining optimal paths. This pilot measures whether the reduction pays for preparation, and whether the frozen CNN adds a benefit beyond the landmark method alone. No models are trained and no new test set is used.

## Fixed comparison

Use 40 maps from the previous policy-validation archive: 10 random and 10 wall maps at each of 32×32 and 64×64. Select the first layout IDs in sorted order within each family, balancing random 32×32 between five maps at each existing density. Selection does not use timing, path difficulty, or prediction error. Existing training/test separation remains unchanged.

Each map uses its stored goal and the same 128 deterministic distinct starts as the ordering pilot. Keep unreachable starts. Sessions contain the first 1, 16, 64, or 128 starts. Each session prepares its own map and goal state. There is no reuse across measured sessions or methods.

Compare these five methods using the same prepared graph and pathfinding implementation:

- Manhattan A*.
- Manhattan A* with the frozen 8-channel seed-7 distance model breaking ties.
- A* with the maximum of Manhattan and four corner-landmark lower bounds.
- The same landmark A* with the same frozen model breaking ties.
- Reverse BFS with stored next steps, returning actual paths.

The landmark rule and corner selection are unchanged from step 1. Model predictions never replace the safe primary bounds. Comparing the two landmark methods isolates the added model cost and search effect. Comparing landmark-only against Manhattan measures the classical heuristic change.

## Timing and verification

Use one CPU thread and seven repetitions. Shuffle map order, then shuffle all method/batch-size combinations within each repetition. Warm each method three times on open 32×32 and 64×64 grids before measurement. The fixed run has 5,600 measured sessions and 292,600 query executions.

Session time starts before graph preparation and ends after every returned path is built. Retain separate graph, landmark BFS, prediction, planner/score construction, and search times. Feature encoding and output checks are included in prediction. Landmark selection is included in landmark preparation. Reverse BFS preparation is included for its control. Search timing includes result construction and path reconstruction; BFS path following is measured in the same part of its session.

Map files and the model are loaded before sessions. Report the observed one-time model loading cost separately; the session results describe an initialized process with resident model weights. They are not cold application-start timings. Dataset loading, Python/library startup, warmup, result verification, and saving evidence are excluded equally. No offline landmark index is provided for free. A change of goal across queries is a different workload and is not measured here.

Recompute BFS labels before measurement. Verify each returned path and cost after its timer stops, including no-path cases. Repeated paths and all counters must match. Retain full paths for the 128-start session and require every shorter session to match its corresponding prefix. Freeze code, checkpoint, selected arrays, and protocol before timing.

## Summaries and interpretation

For each map, method, and batch size, use the median total time over seven repetitions. Take component times from that same median repetition so the breakdown remains additive. Divide by the batch size for time per query, then average maps equally within each family. Compare methods using paired per-map ratios, with 2,000 whole-map bootstrap draws for pointwise 95% intervals. Reachable expansion ratios are first averaged within each map. Report unreachable counts separately.

No model or method is selected for a final test in this pilot. A small or uncertain timing difference is not an established advantage. Fewer expansions alone are insufficient: a useful model contribution must improve complete time against landmark-only, and a practical improvement must also be assessed against Manhattan and reverse BFS. Results cover one model seed, 10 maps per family, this CPU, and repeated starts sharing a fixed goal.

Settings are in [landmark_timing.toml](../configs/landmark_timing.toml). Run `scripts/run_landmark_timing.py prepare` and then `evaluate` from the repository root using the project Python environment. The stages refuse to overwrite existing measurements or run after frozen inputs change.
