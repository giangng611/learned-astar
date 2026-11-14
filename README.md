# Learning Heuristics for A* Search with Obstacles

This CSCI/ARTI 4550/6550 capstone project compares a learned heuristic with Manhattan A* and Weighted A* on grids with obstacles. The main questions are whether the model reduces the number of cells searched, whether it saves time after including inference, and how it affects path length. BFS and Dijkstra provide shortest-path checks.

Phases 0 and 1 are complete: the environment, map generator, search baselines, tests, and demo are working. The dataset, model training, and full experiments are next. See the [proposal](docs/proposal.md) for the research question and references, and the [implementation plan](docs/research_architecture.md) for the planned methods.

## Setup

Use Python 3.11 or newer. The local setup was checked with Python 3.14.3 and PyTorch 2.15.0 on an Apple Silicon Mac. From the project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ml]'
python scripts/check_environment.py
```

The environment check runs a small CPU convolution and calculates its gradients. Baseline search only needs Python's standard library; install `'.[dev]'` to skip the model dependencies. PyTorch's [installation guide](https://pytorch.org/get-started/locally/) covers other platforms.

To reuse the recorded package versions on a matching setup, install with `python -m pip install -c requirements-lock.txt -e '.[dev,ml]'`. The map generator uses a separate seeded random generator. The model setup fixes the Python, NumPy, and PyTorch seeds and uses one CPU thread. Results and timings may still differ across machines.

## Run the baselines

```sh
python -m learned_astar demo --config configs/baseline.toml --show-map --output outputs/demo.json
```

The default configuration uses a 32×32 grid, 25% obstacles, seed 4550, and Weighted A* weights of 1.25, 1.5, and 2.0. All six baseline settings receive the same map and endpoints. The demo checks returned paths, compares their costs with BFS, and saves the map, settings, paths, counts, and timings as JSON. `learned-astar demo` is an equivalent command after installation.

The demo runs one query once, so its timings are only a quick check. Full experiments will use repeated measurements across many maps. Unreachable queries are kept, and searches that reach a limit are reported separately.

```sh
python -m learned_astar demo --seed 12
python -m learned_astar demo --max-expansions 10
python -m learned_astar demo --time-limit 0.01
```

## Checks and code layout

```sh
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

Tests cover every 3×3 obstacle layout and every pair of different free endpoints, plus larger maps, detours, invalid inputs, repeated runs, reopening, and search limits.

Grid code is in `src/learned_astar/environments/`, search code in `src/learned_astar/search/`, and the demo in `src/learned_astar/cli.py`. The [baseline notes](docs/search_baselines.md) define search behavior and counts. The [development log](docs/development_log.md) records progress. Generated data, checkpoints, local environments, and demo outputs stay out of Git.
