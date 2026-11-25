# Method-selection pilot

The next learning question is whether a small model can choose a useful search method for a known batch of queries. The choices are on-demand Manhattan A*, on-demand landmark A*, and reverse BFS with cached paths. Each choice still returns an optimal path. The model predicts runtime; it does not generate a map or replace the search algorithm.

## Development data

The first run uses 72 fresh maps: 48 for training and 24 for validation. Both splits contain random and wall maps at 32×32 and 64×64, with equal counts per family. Random maps balance obstacle densities of 0.10 and 0.25. Layouts and their rotations cannot appear in both splits. The exclusion list contains the 6,800 earlier groups and the 120 maps from the latest test. Only layout identifiers are read from that test archive; its runtime measurements are not training labels.

Each map supplies three batches of 128 queries, with 1, 8, or 128 distinct goals. Starts are fixed across batches and disjoint from the goal pool. Changing the goal count also changes some endpoints, so this is not an isolated test of goal reuse alone. All three methods run on the same queries in shuffled order, with three repetitions. This gives 216 learning rows from 72 independent maps, 1,944 timed sessions, and 248,832 query executions. Rows from one map always remain in the same split.

The runtime targets are median session times. Raw repetitions and the gap between the two fastest methods are retained because close timings may give unstable winner labels. Every timed result is checked against a BFS cost, and the first repetition's paths and search counters are saved. Repeated results must agree. A completed map is saved atomically, so an interrupted measurement can resume without repeating completed maps.

## Inputs and training

The eight inputs are height, width, obstacle fraction, the largest row or column obstacle fraction, the larger row or column obstacle-fraction standard deviation, query count, distinct goal count, and mean normalized Manhattan distance between endpoints. These come only from the map and the known query batch. Family names, map seeds, exact distances, and measured times are not prediction inputs. This pilot assumes the entire query batch is available before selecting a method; it does not cover an unknown stream of future goals.

The network has one hidden layer with eight units and three runtime outputs, for 99 trainable parameters. It learns standardized log runtimes with mean squared error and chooses the method with the smallest prediction. Input and target normalization use training rows only. The first pilot uses seed 7, at most 200 full-batch updates, and stops after 30 epochs without validation-loss improvement. The checkpoint with the lowest validation loss is saved. No previous distance-CNN weights are loaded.

A fixed rule provides a useful comparison: choose BFS when there are at most eight distinct goals; otherwise choose landmarks when the largest line obstacle fraction is at least 0.8, and Manhattan otherwise. This rule is fixed before collecting the pilot and is informed by previous experiments. Other controls are each fixed method, the single fixed method chosen using training times, and an oracle that chooses the fastest measured method for every batch. The oracle is a lower reference cost, not a deployable selector.

## Timing and interpretation

Method timings include graph preparation, landmark preparation when needed, goal caches, searches, and path construction. They retain the earlier study's boundary: final cache disposal, file operations, warm-up, and correctness checks are outside the timer. Feature extraction is measured separately from each method, and one-query selector inference includes input packing and normalization. Here “one query” to the selector means one decision for the full pathfinding batch. The rule is timed too.

The validation report adds median feature and decision costs to the chosen method's median session cost. These are recombined timing estimates, not a live end-to-end deployment benchmark. Fixed methods do not pay selector feature costs. Workloads are averaged within each map before maps are averaged. Validation is used to select the checkpoint, so its results are development evidence, not an independent test claim. A promising pilot still needs a frozen selector, fresh test maps, and live timing that includes selection. A win is not assumed.

## Run locally

Use the project Terminal in PyCharm. Run the commands in order, waiting for each to finish:

```bash
.venv/bin/python scripts/prepare_selector_data.py prepare
.venv/bin/python scripts/prepare_selector_data.py measure
.venv/bin/python scripts/train_selector.py
```

Preparation refuses to overwrite an existing run. If measurement is interrupted, rerun only the second command; it keeps completed maps and repeats any incomplete map. Use the same computer and similar load, and avoid other heavy jobs during timing. Source and input changes are rejected. The training command also refuses to overwrite an existing model directory.

Data and raw evidence are saved under `outputs/selector_pilot`. Training saves `model.pt`, `history.json`, `validation_predictions.json`, and `summary.json` under `outputs/selector_model`. The summary printed at the end is enough for an initial review. A ratio below one favors the learned selector; a ratio above one favors its comparison method. These ratios alone do not establish a reliable improvement.

The local 72-map pilot completed with checkpoint epoch 188. Its mean validation batch estimate was 9.527 ms for the learned selector and 9.506 ms for the fixed rule. This supports useful selection relative to fixed methods, but does not establish a benefit over the rule.

## Intermediate-goal diagnostic

The next probe freezes that checkpoint and measures 16, 32, and 64 distinct goals on 24 new maps. It excludes all 6,992 earlier groups, including the first selector pilot. All 24 maps are used for this development diagnostic; inherited train/validation labels are retained only for possible later work. No model is fitted or selected in this step. The original rule is unchanged, so a benefit over it would still need comparison with a rule tuned using training data before a final test.

```bash
.venv/bin/python scripts/run_selector_transition.py prepare
.venv/bin/python scripts/run_selector_transition.py measure
.venv/bin/python scripts/run_selector_transition.py evaluate
```

The measurement stage can resume. Preparation snapshots the checkpoint before timing and rejects changed sources or inputs. Evaluation saves the overall and per-goal-count results in `outputs/selector_transition/selector_evaluation.json`. These remain recombined timing estimates, not a live end-to-end test. The full probe is left for local execution.
