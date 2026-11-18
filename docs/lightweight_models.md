# Testing smaller models

This experiment changes only the CNN width. The current route model has 16 channels and 10,033 parameters. The new models use 8 channels with 2,713 parameters, or 4 channels with 781 parameters. They keep the same five convolution blocks, dilation pattern, inputs, distance scaling, and output field. Keeping the depth and receptive field fixed helps separate model size from how much of the map it can see.

## Training and search

Both smaller widths train with seeds 7, 17, and 27 on the same 1,200 training maps and 240 checkpoint-validation maps as the 16-channel model. The optimizer, batch size, 60-epoch budget, patience, and loss remain unchanged: distance loss plus 0.02 times the neighboring-move loss. Each seed retains its lowest combined validation loss. The three existing 16-channel checkpoints are fixed controls.

All models guide only ties in Manhattan A*. The primary priority remains `g + Manhattan`, so shrinking the model does not weaken the search rule's optimality guarantee. Manhattan without a model uses the same search implementation. The main workload is a single query with a fresh prediction; model loading is outside timing, while input encoding, inference, search, and path reconstruction are included. This round does not remeasure the 64-query reuse workload.

## Selection before testing

Generate 120 new policy-validation maps: 60 random and 60 wall maps, with four starts per goal. Exclude every layout group used in earlier experiments, including rotations. Measure all three widths and Manhattan on identical queries, with five shuffled repetitions and one CPU thread.

A smaller width is eligible only if all queries complete correctly, all returned paths are optimal, and its mean paired reachable-query expansion ratio against Manhattan is below one in both map families. Among eligible widths, choose the lowest mean paired total-time ratio against the 16-channel model. Pair the same query and seed, average within maps, and give random and wall groups equal weight. The ratio must also be below one; otherwise retain the 16-channel reference. An exact score tie favors the smaller width. This rule tests whether a faster model retains useful search ordering; it does not require matching every gain of the larger model.

Only after selection, generate 300 fresh test maps: 60 each for random 32×32, wall 32×32, dense 32×32, random 16×16, and random 64×64. Exclude all earlier maps and the new policy-validation maps. Report every predeclared width, including unsuccessful candidates. Test results do not change training, the eligibility rule, or the selected width.

## Measurements

Report total query time, prediction time, search time, expanded cells, path quality, and branching-move accuracy. Compare each width with both Manhattan and the 16-channel control on the same new queries. Use medians of five repetitions and 95% intervals from 2,000 whole-map bootstrap samples. Intervals describe variation across maps for three fixed training seeds. Fewer parameters do not automatically mean lower CPU latency, so actual times determine selection.

```sh
python scripts/run_lightweight.py prepare
python scripts/run_lightweight.py train
python scripts/run_lightweight.py select
python scripts/run_lightweight.py test
python scripts/report_lightweight.py
```

Use `--run` with a new directory to repeat the experiment. The runner freezes settings, source manifests, implementation files, and reference checkpoints. It checks trained checkpoints again before test generation. The final archive retains the six new checkpoints, histories, measurements, and dataset provenance.

The [completed results](lightweight_results.md) show the speed and search-quality tradeoff for every width. The validation-selected width remains recorded even where its search advantage does not carry over to test maps.
