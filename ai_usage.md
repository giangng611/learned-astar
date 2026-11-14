# AI usage

## Implementation and troubleshooting

Codex was used during phases 0 and 1 for the following technical work:

- Setting up the Python package, configuration, dependencies, and random seeds.
- Implementing grid generation, path validation, BFS, Dijkstra, Manhattan A*, and Weighted A*.
- Implementing shared priority queue behavior, reopening, stale-entry handling, search limits, and measurement counts.
- Adding the command-line demo and JSON result output.
- Creating and running correctness tests, checking Python/PyTorch compatibility, and resolving setup and code-formatting issues.

The recorded checks include 146 passing tests, code style and formatting checks, a dependency check, and a CPU convolution with gradient calculation. The changes are saved in local commits for review before pushing. Dataset generation, model training, and full experiments have not been completed.
