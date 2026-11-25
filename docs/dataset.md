# Distance dataset

Generate the fixed map splits and verify them before training:

```sh
python -m learned_astar.data generate --config configs/dataset.toml --output datasets/main
python -m learned_astar.data verify --output datasets/main
```

The generator saves each obstacle layout with its goals, starts, and exact distance maps. Reverse BFS starts at the goal and gives the shortest distance to every reachable cell. Labels use `-2` for blocked cells and `-1` for free cells that cannot reach the goal. Training uses only labels greater than or equal to zero.

Maps are split before any goal examples are made. A layout and its rotations share one group ID, and the generator rejects repeated groups across all splits. Each record includes its seed, dimensions, obstacle fraction, reachable fraction, and file checksum. Queries are sampled without checking connectivity, so unreachable cases remain in the evaluation.

The manifest lists every map. `metadata.json` records the configuration and manifest checksum; it is written after generation finishes. Verification checks file integrity, group separation, valid endpoints, and local distance conditions: the goal has distance zero, neighboring reachable cells differ by at most one, and each positive distance has a neighbor one step closer. These conditions independently check the BFS labels.

The main sets contain 1,200 training, 200 validation, and 300 test maps. Four additional sets contain 150 maps each: higher obstacle density, smaller grids, larger grids, and wall layouts. See `configs/dataset.toml` for the exact settings. The pilot configuration uses separate seeds and a small training and validation set.
