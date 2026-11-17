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
