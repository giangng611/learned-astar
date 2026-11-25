# On-demand heuristic evidence

The [protocol](../../docs/demand_heuristic.md) fixes the five methods, goal workloads, validation gate, and fresh-test design. The implementation changes when Manhattan and landmark scores are computed, while preserving the existing A* search loop. No model is trained in this experiment.

The `validation` directory uses the previous 40 policy-validation maps. The `test` directory contains the fixed 120-map test performed after the validation gate passed. Test layouts exclude all 6,800 earlier groups and include random and wall maps at sizes 32, 64, and 128. The [results](../../docs/demand_heuristic_results.md) distinguish these two stages.

Each completed archive contains the protocol, map manifest, map arrays, workload labels, frozen source hashes, raw timing repetitions, full query paths and counters, median sessions, summaries, numerical tables, and PNG/PDF figures. `score_entries` counts stored heuristic values; it is not a byte-level memory measurement. `policy_manifest.json` retains original input paths for provenance, including in the fresh-test archive. The arrays themselves are available under `maps/`.

Check the raw timing, query-file, and map checksums and independently recompute timing medians and paired means without local generated datasets:

```sh
python scripts/check_demand_archive.py results/demand_heuristic/validation
python scripts/check_demand_archive.py results/demand_heuristic/test
```

These commands verify arithmetic and evidence checksums. Full path checks, repeated-result signatures, and independent BFS-label checks are performed when the experiment report is generated. The complete reproduction commands in the protocol use the original local dataset manifests for validation selection and layout exclusions. Do not treat timing repetitions as independent maps or combine absolute timings from different experiment rounds.
