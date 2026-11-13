# Learning Heuristics for A* Search with Obstacles

This is my AI capstone project for CSCI/ARTI 4550/6550 at the University of Georgia. I want to find out whether a small neural network can help A* find paths more efficiently by using information about obstacles.

I will compare the learned heuristic with Manhattan distance and Weighted A*. I will also use Dijkstra and BFS to check the shortest paths. The main measurements will be how many cells each search expands, how long it takes, and the length of the path it returns.

The project environment is ready. The search algorithms come next; there is no trained model or experimental result yet.

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

The ordinary search algorithms will only need Python's standard library. To work on those without downloading PyTorch, install `'.[dev]'` instead. The model check needs the `ml` dependencies.

`requirements-lock.txt` records the package versions used for this local check. On a matching Python and operating system setup, use `python -m pip install -c requirements-lock.txt -e '.[dev,ml]'` to reuse those versions. Other platforms may need different PyTorch packages.

## Project settings

The starting settings are in `configs/baseline.toml`: a 32×32 grid, 25% blocked cells, seed 4550, and Weighted A* weights of 1.25, 1.5, and 2.0. This file is for a small demo. It is not the final experiment setup.

Map generation uses its own random generator so another part of the program does not change the maps by accident. The model setup fixes the Python, NumPy, and PyTorch seeds and uses one CPU thread. These settings help repeat runs on the same setup, but they do not promise identical timings or identical results on every machine.

Generated data, model files, local environments, and demo outputs are kept out of Git.
