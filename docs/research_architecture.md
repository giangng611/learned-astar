# Implementation plan

Updated November 14, 2026

Phases 0 and 1 are complete: environment setup, grid generation, baseline search, tests, and a command-line demo. The remaining work is the dataset, learned heuristic, and experiments. The [proposal](proposal.md) gives the research question and references.

## Code and search

Python code is in `src/learned_astar/`. Grid generation is in `environments/`, search in `search/`, and the demo in `cli.py`. Later stages will add `data/`, `models/`, `training/`, `evaluation/`, and `visualization/`. Settings, helper commands, and tests belong in `configs/`, `scripts/`, and `tests/`.

The main task uses static 32×32 grids, four movement directions, and unit costs. BFS uses a separate queue. Dijkstra, Manhattan A*, and Weighted A* share a priority queue loop; learned A* will use it too. Search prefers smaller `f`, then larger `g`, then earlier insertion. Better routes reopen cells, stale entries are skipped, and success requires removing the goal from the queue. The [baseline notes](search_baselines.md) define counts, validation, and limit behavior.

BFS, Dijkstra, and Manhattan A* return shortest paths under these rules. Weighted A* uses Manhattan distance with weights 1.25, 1.5, and 2.0. With reopening and goal-removal stopping, its cost is at most the weight times the optimal cost.

Learned A* will use `f = g + h_model` without an optimality guarantee. Nonnegative predictions or a mixture with Manhattan distance do not establish that guarantee. With finite predictions, reopening, and no work limit, the finite-grid search can still find an existing route. An optional later variant could use learned values only to break ties in Manhattan A* priority, preserving optimality.

## Model and training

A small convolutional neural network (CNN) will predict distances for the whole grid once per query. Its five inputs are obstacles, a goal marker, signed row and column offsets from the goal, and Manhattan distance. Offsets are divided by `max(H - 1, 1)` and `max(W - 1, 1)`; Manhattan distance uses `S = max(H + W - 2, 1)`.

The planned model has five 3×3 convolution layers with 32 output channels, dilation values 1, 2, 4, 8, and 16, matching padding, and ReLU. A final 1×1 layer and softplus produce nonnegative distances. It has about 39,000 parameters and a 63×63 receptive field, meaning the region each output can use. This covers the main grids but may miss distant obstacles on 64×64 maps.

Reverse BFS supplies exact distance targets divided by `S`. Targets are not clipped at one because detours can exceed `S`. Search multiplies predictions by `S` and sets the goal value to zero.

Smooth L1 loss uses reachable free cells only. It is averaged within each map-goal example, then across the batch, so larger reachable regions do not receive more weight. Blocked and unreachable cells are masked out of the loss; their predictions will not be used to prune search.

Initial training settings are Adam, learning rate 0.001, batch size 16, and at most 100 epochs. Training stops after 10 epochs without validation improvement. The best validation checkpoint is saved for each of three random seeds, and all three are reported. Settings may change during a small pilot but must be fixed before final test evaluation.

## Dataset

Random maps contain exactly `round(p * H * W)` blocked cells sampled without replacement. Wall and corridor maps use a separate generator. Dimensions and obstacle layouts are checked for duplicates before splitting; every goal and any rotated copies of one layout stay in the same split.

| Split | Maps | Examples per map |
| --- | --- | --- |
| Training | 1,200 | Two free goals, with distance labels for all reachable cells. |
| Validation | 200 | Two free goals, using the same label process. |
| Test | 300 | One free goal and four different free starts, excluding the goal. |

Each split uses 32×32 maps, half at 10% obstacles and half at 25%. Endpoints are sampled without requiring a route. Unreachable queries are kept, and maps with too few free cells are recorded. Labels mark blocked and unreachable cells separately. Exact test distances remain in evaluation code and are not passed to learned search.

Extra conditions start with 150 maps each: 32×32 at 40% obstacles, 16×16 and 64×64 at 25%, and 32×32 wall/corridor maps. Change one condition at a time where possible. Counts will be confirmed after a pilot, before inspecting test results; they are practical targets, not a claim of statistical significance.

Record actual obstacle fraction, reachable-cell fraction, and shortest path lengths. For reachable queries with different endpoints, also record true distance divided by Manhattan distance to describe detours. Higher obstacle density alone does not establish greater difficulty.

## Timing

The main comparison runs on the CPU with fixed thread settings. Total query time includes input preparation, model inference, conversion to search values, search, and path reconstruction. File loading, exact labels, path validation, and writing results stay outside every algorithm's timer. Model loading and training time are reported separately.

Warm up operations, vary algorithm order, and time each query five times. Use the median repetition for analysis, with fresh model predictions for each run. Repetitions are not independent test examples. Time reverse BFS separately to compare full neural prediction with computing exact distances.

If predictions are later reused across starts sharing a goal, compare with reverse BFS plus distance lookups. Include the cost of preparing either distance map.

## Analysis

The main outcomes are expanded cells and total runtime relative to Manhattan A*. Weighted A* provides a simple speed-versus-path-length comparison. Report means, medians, the middle half of measurements, slow cases, success rates, and the fraction of optimal paths.

Use 95% bootstrap confidence intervals by resampling whole maps, keeping their starts and algorithm results together. For the main interval, average the three trained models within each map, then show each model separately. This measures variation across maps for those models, not all possible training outcomes.

Path cost ratio is returned cost divided by optimal cost, only for successful queries with positive optimal cost. Report failures alongside it. Keep unreachable cases separate and label limited searches as incomplete; their elapsed time is not a completion time. Start equal to goal remains a correctness test but is excluded from ratios.

Measure mean absolute error (MAE), root mean squared error (RMSE), and the frequency and size of overestimates on reachable cells, summarized by map. Compare these errors with search behavior without treating an observed relationship as proof of cause.

## Work order and checks

1. **Complete:** environment and baseline search. The local Python 3.14.3/PyTorch 2.15.0 setup passed a CPU gradient check, and 146 tests passed. Package versions are in `requirements-lock.txt`; checks are recorded in the [development log](development_log.md).
2. **Dataset and training:** verify labels, split separation, repeatability, and checkpoint loading. Estimated three to four workdays.
3. **Integration:** connect predictions to search and complete a small run before larger experiments. Estimated one to two workdays.
4. **Experiments and analysis:** run the fixed comparison and retain all outcomes. Estimated three to four workdays plus computing time.
5. **Report and presentation:** explain results and limits. Estimated three to four workdays.

The proposal deadline is November 16, 2026; final report and presentation dates still need confirmation. Finish the main 32×32 comparison before extra conditions if time is short. Slow inference, inaccurate predictions, and weak transfer to different maps remain possible outcomes and will be reported. Changes are committed in reviewable groups before pushing.
