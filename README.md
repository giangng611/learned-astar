# Learning Heuristics for A* Search with Obstacles

This CSCI/ARTI 4550/6550 capstone compares a learned distance heuristic with Manhattan A*, Weighted A*, BFS, and Dijkstra. It measures search effort, total query time including inference, and path quality on the same queries. The workflow covers dataset generation, training with three seeds, held-out evaluation, figures, and a terminal demonstration.

See the [original report](docs/report.md), [search-learning results](docs/search_learning_results.md), and [runtime results](docs/runtime_results.md) for findings. The [experiment architecture](docs/research_architecture.md) explains the design; the [submitted proposal](docs/proposal.md) retains the original plan. The updated [presentation](presentation/learned_astar_research.pptx) includes speaker notes, and the [recording guide](presentation/recording_guide.md) explains the demo and section handoffs.

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
15-slide deck remains at `presentation/learned_astar.pptx`; the updated deck covers
the later research as well.

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

## Reproduce the experiment

Run the whole pipeline in a new directory:

```sh
python scripts/run_experiment.py --run-dir outputs/reproduction
```

This copies the configurations, generates and verifies 2,300 maps, trains three models, evaluates 3,600 queries, and saves summaries and figures. Existing directories are not overwritten. Training can take several minutes per seed on this machine; evaluation runs after training finishes.

Individual stages are also available:

```sh
python -m learned_astar.data generate --config configs/dataset.toml --output datasets/main
python -m learned_astar.data verify --output datasets/main
python -m learned_astar.training.train --data datasets/main --config configs/training.toml --output checkpoints/main
python -m learned_astar.evaluation.run --data datasets/main --checkpoints checkpoints/main/seed_7.pt checkpoints/main/seed_17.pt checkpoints/main/seed_27.pt --config configs/evaluation.toml --output outputs/evaluation
python -m learned_astar.evaluation.summarize --input outputs/evaluation --output results/main
python -m learned_astar.evaluation.plot --results results/main --training checkpoints/main --data datasets/main --checkpoints checkpoints/main --output results/figures
```

Use fresh output paths when rerunning a completed stage. The repository includes compact query summaries, aggregate results, figures, training histories, and the selected checkpoints. Generated NPZ datasets and all timing repetitions remain local; the generator and saved settings reproduce them. The [dataset](docs/dataset.md), [training](docs/training.md), and [evaluation](docs/evaluation.md) notes explain the methods and measurement rules.

## Checks and code layout

```sh
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pip check
```

Tests cover every 3×3 obstacle layout and every pair of different free endpoints, larger maps, reopening, search limits, labels, split separation, checkpoints, inference timing, result summaries, and shell commands.

The package separates `environments`, `search`, `data`, `models`, `training`, and `evaluation`. `terminal.py` implements the shell, while `scripts/run_experiment.py` connects the experiment stages. The [baseline notes](docs/search_baselines.md) define counts and tie rules, and the [development log](docs/development_log.md) records completed checks.
