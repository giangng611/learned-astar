# Computing heuristic scores only when needed

The changing-goal study found that constructing a full score field for every new goal erased the landmark search benefit. This experiment asks whether computing the same scores only at requested cells reduces complete time. It changes evaluation cost, not the heuristic formula or priority order. The landmark method remains a classical control, based on the [landmark lower-bound approach](https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory-2/).

## Methods and correctness

Compare full-field Manhattan, on-demand Manhattan, full-field landmarks, on-demand landmarks, and reverse BFS. Both on-demand methods use the unchanged prepared A* loop and cache a score after its first request. Every cache belongs to one map and goal. The landmark index is built once per session; new goals retain the map index but receive separate score caches. All preparation, cache misses, dictionary lookups, and returned paths are timed.

The full-field and on-demand variants must return identical paths and all search counters for each query. Exact BFS checks optimal cost and reachability independently. Exhaustive 3×3 tests cover disconnected maps, equal endpoints, and cache reuse. Separate tests cover larger sparse caches and search limits. Computing the same consistent bound later cannot change its value; this experiment does not introduce learned primary scores.

## Fixed validation and test sequence

Use the same 40 validation maps and deterministic endpoints as the changing-goal study. Each session contains 128 queries and either 1, 8, or 128 distinct goals. Shuffle all methods and workloads within each of five repetitions after warm-up. Summarize the median complete session, including its timing components. Bootstrap whole maps for pointwise 95% intervals. Different goal counts change endpoints, so only within-workload comparisons are paired.

Before fresh testing, require correct paths and identical full-field/on-demand search results everywhere. At 128 goals, on-demand landmark time must be at most 95% of full-field landmark time, with the interval upper bound below one in both wall families. This gate tests the proposed implementation improvement; it does not require or imply a win over Manhattan or reverse BFS. Both Manhattan variants remain in all tables.

If the gate passes, freeze the implementation and evaluate 120 fresh maps: 20 random and 20 wall maps at each of 32×32, 64×64, and 128×128. Random families balance 10% and 25% obstacles. Exclude every earlier dataset layout group, including rotations, and all new groups from each other. Keep the same workload sizes, methods, and repetitions. Do not revise the design or select favorable families after seeing fresh-test outcomes. Larger grids test whether the same preparation tradeoff extends beyond the earlier sizes.

Retain settings, seeds, map occupancy, layout identifiers, all workload labels, raw repetitions, full query paths, counters, score-cache entries, summaries, and source hashes. Cache-entry counts measure stored score values, not peak memory bytes. Index fields and normal A* working arrays remain additional memory costs. Model training is outside this experiment; no neural improvement will be claimed from a classical implementation change.

## Reproduction

```sh
python scripts/run_demand_heuristic.py prepare --run outputs/demand_validation
python scripts/run_demand_heuristic.py evaluate --run outputs/demand_validation
python scripts/report_demand_heuristic.py --run outputs/demand_validation
python scripts/run_demand_heuristic.py prepare-test --run outputs/demand_test --validation outputs/demand_validation
python scripts/run_demand_heuristic.py evaluate --run outputs/demand_test
python scripts/report_demand_heuristic.py --run outputs/demand_test
```

The test preparation command checks the gate and frozen validation evidence first. An existing measurement directory is not overwritten. Previous studies remain unchanged so the new run can be assessed separately.
