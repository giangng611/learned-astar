# Learning Heuristics for A* Search with Obstacles

This is my AI capstone project for CSCI/ARTI 4550/6550 at the University of Georgia. I want to find out whether a small neural network can help A* find paths more efficiently by using information about obstacles.

I will compare the learned heuristic with Manhattan distance and Weighted A*. I will also use Dijkstra and BFS to check the shortest paths. The main measurements will be how many cells each search expands, how long it takes, and the length of the path it returns.

Phases 0 and 1 are complete. The project has a working environment, repeatable map generation, and tested search baselines. There is no training dataset, trained model, or full experiment result yet.

The [proposal](docs/proposal.md) explains the research question and includes five related papers. The [implementation plan](docs/research_architecture.md) describes how I plan to build and test the project.

## Setup

Use Python 3.11 or newer. The local setup was checked with Python 3.14.3 and PyTorch 2.15.0 on an Apple Silicon Mac. From the project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ml]'
python scripts/check_environment.py
python -m pytest
```

The environment check runs a small convolution and calculates its gradients on the CPU. It checks that PyTorch works before I start building the model. PyTorch's [installation guide](https://pytorch.org/get-started/locally/) explains the options for other machines.

The ordinary search algorithms only need Python's standard library. To work on those without downloading PyTorch, install `'.[dev]'` instead. The model check needs the `ml` dependencies.

`requirements-lock.txt` records the package versions used for this local check. On a matching Python and operating system setup, use `python -m pip install -c requirements-lock.txt -e '.[dev,ml]'` to reuse those versions. Other platforms may need different PyTorch packages.

## Project settings

The starting settings are in `configs/baseline.toml`: a 32×32 grid, 25% blocked cells, seed 4550, and Weighted A* weights of 1.25, 1.5, and 2.0. This file is for a small demo. It is not the final experiment setup.

Map generation uses its own random generator so another part of the program does not change the maps by accident. The model setup fixes the Python, NumPy, and PyTorch seeds and uses one CPU thread. These settings help repeat runs on the same setup, but they do not promise identical timings or identical results on every machine.

Generated data, model files, local environments, and demo outputs are kept out of Git.

## Run the baselines

After setup, compare all six baseline settings on the same map, start, and goal:

```sh
python -m learned_astar demo --config configs/baseline.toml --show-map --output outputs/demo.json
```

The command prints path costs, the number of cells expanded, and search times. It also checks each returned path and compares its cost with BFS. The JSON file saves the map, endpoints, settings, paths, and counts so the query can be inspected later. `learned-astar demo` runs the same command after installation.

This demo runs one query once. Its timing is only a quick check that the program works; it is not enough to claim that one method is faster. The full experiment will repeat measurements across many maps.

To try another query or limit the work, run:

```sh
python -m learned_astar demo --seed 12
python -m learned_astar demo --max-expansions 10
python -m learned_astar demo --time-limit 0.01
```

A search that reaches a limit is reported separately from a map with no route. The program does not keep generating maps until it finds an easy or reachable query.

## Checks and project layout

```sh
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

The tests cover all 512 obstacle layouts on a 3×3 grid and every pair of different free endpoints in those layouts. They also cover larger random maps, invalid endpoints, detours, repeated runs, search limits, and reopening cells after a better route is found.

The grid code is in `src/learned_astar/environments/`, and the search algorithms are in `src/learned_astar/search/`. The command-line demo is in `src/learned_astar/cli.py`. The [baseline notes](docs/search_baselines.md) explain the counting and stopping rules. The [development log](docs/development_log.md) records the completed phases and the next step.
