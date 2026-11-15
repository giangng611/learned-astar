# Evaluation method

Run the saved models on all held-out splits, then summarize the measurements:

```sh
python -m learned_astar.evaluation.run --data datasets/main --checkpoints checkpoints/main/seed_7.pt checkpoints/main/seed_17.pt checkpoints/main/seed_27.pt --config configs/evaluation.toml --output outputs/evaluation
python -m learned_astar.evaluation.summarize --input outputs/evaluation --output results/main
```

Every query is shared by BFS, Dijkstra, Manhattan A*, three Weighted A* settings, and the three trained models. Evaluation rejects checkpoints whose development layouts overlap the test maps or whose dataset checksum differs. Exact distances check the returned cost but are never supplied to the learned search.

Each method runs five times per query in a shuffled order after warm-up. The learned timer includes preparing inputs, a fresh model prediction, search, and path reconstruction. Model loading, reference BFS, validation, and file writing are measured separately or kept outside this timer. All query methods use one CPU thread. The limits are five seconds and one million expansions per query.

The raw file keeps every repetition and outcome. Summaries use each query's median repetition, then report means, medians, quartiles, and the 95th percentile. Completion times exclude searches that reach a limit; their attempted time and status remain available. Success rates use reachable queries, while unreachable queries have a separate correctness rate. Path ratios use successful queries with positive optimal distance.

Paired ratios compare each method with Manhattan A* on the same query. We average queries within a map, and average the three learned models within each map for the combined estimate. The 95% confidence interval uses 2,000 bootstrap samples of whole maps. It describes map variation for these fixed models, not uncertainty over all possible training seeds or machines. Each seed is also reported separately.

Prediction error is measured on reachable cells for each map. We report mean absolute error, root mean squared error, and the frequency and size of overestimates. Maps, queries, model runs, and repeated timing measurements are distinct units and should not be treated as independent copies of the same experiment.
