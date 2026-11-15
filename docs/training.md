# Learning the distance heuristic

The model receives five maps: obstacles, the goal location, row and column offsets from the goal, and Manhattan distance. Five convolution layers combine nearby information at increasing distances. A final layer produces one nonnegative estimate for each cell. The network has 38,497 parameters and accepts different grid sizes.

Training compares predictions with reverse BFS distances on reachable free cells. Each map-goal example receives equal weight, even when its reachable region is small. Blocked and unreachable cells do not contribute to the loss. Targets are scaled by the grid dimensions but are not clipped, so long detours remain valid targets.

```sh
python -m learned_astar.training.train --data datasets/main --config configs/training.toml --output checkpoints/main
```

The fixed run uses seeds 7, 17, and 27, Adam at learning rate 0.001, batch size 16, and at most 40 epochs. Training stops early after ten epochs without validation improvement. The checkpoint with the lowest validation loss is saved separately for each seed. Checkpoints record the dataset checksum and all development layout groups, which evaluation checks for overlap with test maps.

The 40-epoch budget and four CPU threads were chosen after a small pilot and a CPU throughput check, before examining final test results. Query timing uses one CPU thread. Logs contain every training and validation loss, epoch duration, checkpoint choice, and total training time.

During search, predictions are converted back to distances and the goal estimate is set to zero. The model cannot block cells or remove neighbors. A* still searches the original map and can reopen a cell when it finds a shorter route. Since the learned estimate can overestimate distance, its returned path is not guaranteed to be optimal.
