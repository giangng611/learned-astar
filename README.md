# Learning Heuristics for A* Search with Obstacles

This CSCI/ARTI 4550/6550 capstone compares a learned distance heuristic with Manhattan A*, Weighted A*, BFS, and Dijkstra. It measures search effort, total query time including inference, and path quality on the same queries. The workflow covers dataset generation, training with three seeds, held-out evaluation, figures, and a terminal demonstration.

Start with the [research synthesis](docs/final_report.md) for the complete findings and the [discussion guide](docs/discussion_guide.md) for a plain-language explanation. Learned tie ordering reduces some search work, but has not established a reliable complete-time win over equally prepared Manhattan. The latest classical landmark implementation has a fresh-test speed benefit on changing-goal wall workloads; this is separate from the learned results.

The [experiment architecture](docs/research_architecture.md) explains the design; the [submitted proposal](docs/proposal.md) retains the original plan. The [original report](docs/report.md) covers the first experiment. The [18-slide presentation](presentation/learned_astar_research.pptx) and [recording guide](presentation/recording_guide.md) cover research through v4 and still need the later findings added.

The next [method-selection pilot](docs/selector_pilot.md) provides local commands to generate separate development maps, measure three optimal search methods, and train a 99-parameter runtime predictor. Its full pilot has not run yet. The initial comparison uses validation data and a fixed selection rule; it is not a new test result.

## Setup

Use Python 3.11 or newer. The recorded run used Python 3.14.3, PyTorch 2.15.0, and NumPy 2.5.4 on an Apple Silicon Mac, using CPU execution.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ml,plots]'
python scripts/check_environment.py
```

To reuse the recorded versions on a matching setup, add `-c requirements-lock.txt` to the install command. Baseline search only needs Python's standard library; the learned demo needs the `ml` dependencies. Training uses four CPU threads and query evaluation uses one. Random seeds and settings are saved, but timings can differ across machines and system load.

## Demonstration

The saved models are in `artifacts/checkpoints/`. Start the shell from the project folder:

```sh
python -m learned_astar shell
```

```text
load examples/maps/detour.txt
run
path astar_manhattan
path learned_seed_7
```

Use `draw` to enter a map with `S`, `G`, `#`, and `.`, then finish with `.end`. Use `help` for other commands. The [terminal guide](docs/terminal_demo.md) includes a short recording sequence and an unreachable example.

To run one map and save all paths:

```sh
python -m learned_astar solve --map examples/maps/detour.txt --show-paths --output outputs/demo.json
```

The original baseline-only demo remains available:

```sh
python -m learned_astar demo --config configs/baseline.toml --show-map
```

Single-query timings explain the demo; use the full experiment to compare performance.

## Follow-up research

The [follow-up plan](docs/research_followup.md) compares new training data, direct distance
prediction, a learned detour added to Manhattan, and an obstacle-input control. It selects
the search settings on validation before creating a fresh test set. The original models
and measurements are retained separately. The plan includes a numerical explanation of
input and label scaling and commands for each experiment stage.

The [follow-up results](docs/research_results.md) report five new training runs,
600 fresh test maps, obstacle controls, and exact-distance references. The original
15-slide deck remains at `presentation/learned_astar.pptx`; the 18-slide research deck
covers the follow-ups through v4, not the subsequent smaller-model and landmark studies.

The [search-learning experiment](docs/search_learning.md) keeps Manhattan as the baseline,
adds training on useful neighboring moves, and compares direct heuristics with learned
ordering of equal Manhattan priorities. Its [results](docs/search_learning_results.md)
separate learning quality, optimal paths, search effort, and total time on another fresh
test set. The terminal supports this search rule with `--heuristic-mode tie`.

The [runtime experiment](docs/runtime_reuse.md) freezes these models and tests 1–64 queries
sharing a map and goal. It compares both original and prepared Manhattan, prediction
reuse, exact A*, and a reverse-BFS policy, charging all preparation costs. Its
[results](docs/runtime_results.md) show that shared implementation improvements account
for most of the speedup; equally prepared Manhattan still beats the learned method.

The [smaller-model experiment](docs/lightweight_models.md) compares 16-, 8-, and 4-channel
CNNs with the same route objective and Manhattan tie ordering. It selects a width by
fresh-query time on separate validation maps while requiring useful search ordering.
The [results](docs/lightweight_results.md) separate prediction cost, search effort,
move accuracy, and performance on new map sizes.

The [smaller-model reuse experiment](docs/lightweight_reuse.md) combines those frozen
checkpoints with prepared search. It compares 1–128 starts sharing a map and goal on
300 new maps, including equally prepared Manhattan and reverse-BFS controls. Its
[results](docs/lightweight_reuse_results.md) include setup time, per-seed evidence,
and paired comparisons for every map condition.

The [ordering-limit pilot](docs/order_pilot.md) measures the minimum expansions possible
while keeping Manhattan primary priority. It compares distance and detour tie scores,
then continues one 8-channel seed on either the old data or a mixture with new wall and
64×64 maps. Its [results](docs/order_pilot_results.md) distinguish the ordering ceiling,
training effects, and the fixed validation gate for spending resources on more seeds.
These are validation diagnostics, not another held-out test result.

The [next correctness step](docs/primary_heuristic_step1.md) tests a classical landmark
estimate as a stronger primary heuristic while keeping optimal paths. Small examples
and exhaustive checks establish correctness. The [complete-time pilot](docs/landmark_timing.md)
then measures all setup costs on 40 validation maps. Its [results](docs/landmark_timing_results.md)
show a shared-goal speed benefit on wall maps from landmarks alone; adding the frozen
CNN makes landmarks slower, and reverse BFS remains the fastest 128-start control.

The [changing-goal follow-up](docs/changing_goals.md) keeps 128 queries per map while
varying the number of goals. Its [results](docs/changing_goals_results.md) show that
building full score fields for new goals removes the landmark speed advantage when
every query has a different goal. The frozen CNN adds more cost. This bounded follow-up
completes the changing-goal diagnostic. The [synthesis](docs/final_report.md#limitations-and-next-decisions)
sets out the remaining limitations and conditions for a further bounded experiment.

The [on-demand heuristic study](docs/demand_heuristic.md) computes scores only at cells
requested by search, applying the same change to Manhattan. After a fixed validation
gate, it tests 120 fresh maps at sizes 32, 64, and 128. Its
[results](docs/demand_heuristic_results.md) show a landmark advantage over both Manhattan
variants on wall maps with 128 distinct goals. On-demand Manhattan wins on random maps
at that workload; reverse BFS has the lowest mean time with one or eight shared goals.
Paths and search counters are unchanged by on-demand evaluation. The
[archive](results/demand_heuristic/) includes all map arrays and an independent timing checker.

## Reproduce the experiment

Run the original v1 pipeline in a new directory:

```sh
python scripts/run_experiment.py --run-dir outputs/reproduction
```

This copies the configurations, generates and verifies 2,300 maps, trains three models, evaluates 3,600 queries, and saves summaries and figures. It does not rerun the later research rounds; their linked protocols above provide separate commands. Existing directories are not overwritten. Training can take several minutes per seed on this machine; evaluation runs after training finishes.

Individual stages are also available:

```sh
python -m learned_astar.data generate --config configs/dataset.toml --output datasets/main
python -m learned_astar.data verify --output datasets/main
python -m learned_astar.training.train --data datasets/main --config configs/training.toml --output checkpoints/main
python -m learned_astar.evaluation.run --data datasets/main --checkpoints checkpoints/main/seed_7.pt checkpoints/main/seed_17.pt checkpoints/main/seed_27.pt --config configs/evaluation.toml --output outputs/evaluation
python -m learned_astar.evaluation.summarize --input outputs/evaluation --output results/main
python -m learned_astar.evaluation.plot --results results/main --training checkpoints/main --data datasets/main --checkpoints checkpoints/main --output results/figures
```

Use fresh output paths when rerunning a completed stage. The repository includes compact query summaries, aggregate results, figures, training histories, and the selected checkpoints. Generated NPZ datasets remain local; the generator and saved settings reproduce them. Recent runtime and smaller-model archives also retain the raw timing repetitions. The [dataset](docs/dataset.md), [training](docs/training.md), and [evaluation](docs/evaluation.md) notes explain the methods and measurement rules.

## Checks and code layout

```sh
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pip check
```

Tests cover every 3×3 obstacle layout and every pair of different free endpoints, larger maps, reopening, search limits, labels, split separation, checkpoints, inference timing, result summaries, and shell commands.

The package separates `environments`, `search`, `data`, `models`, `training`, and `evaluation`. `terminal.py` implements the shell, while `scripts/run_experiment.py` connects the experiment stages. The [baseline notes](docs/search_baselines.md) define counts and tie rules, and the [development log](docs/development_log.md) records completed checks.
