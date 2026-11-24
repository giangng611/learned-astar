# Explaining the project

This guide supports group discussion of the [research synthesis](final_report.md). It is an explanation outline, not a replacement for the measured results or an updated slide deck.

## A short explanation

We study whether a small neural network can make shortest-path search more efficient. Manhattan A* estimates the remaining distance by counting row and column steps, which is cheap but ignores walls. Our model sees the map and goal and learns from exact BFS distances. It predicts how far each cell is from the goal.

Using that prediction directly can lose the shortest-path guarantee. Our safer version keeps Manhattan as the main priority and lets the model choose between cells with equal priority. This preserves shortest paths and reduces the number of cells explored on some fresh test maps. However, the time spent making predictions is larger than the search time saved.

We then test smaller models, reuse, additional training, and the best possible tie ordering. These explain why the gap is hard to close. A classical landmark method changes the main estimate and can beat Manhattan when many queries share a goal, but the CNN does not improve that result. Our contribution is showing what was learned, which costs prevent a speed win, and which changes are worth testing next.

## Walk through one query

Suppose Manhattan says the goal is four steps away, but a wall forces a long detour. During training, BFS supplies the actual distance, such as fourteen steps, and the network learns from its prediction error. This exact answer is a training label; the model does not receive the test answer when predicting.

At query time, the trained model reads the map and goal once and outputs one score per cell. A* receives the start and searches for a route. In the safe version, the network's scores only decide between equal Manhattan priorities. A* still checks legal moves and constructs the path. If the goal stays fixed, the same prediction can serve another start. A new goal needs another prediction.

## Questions the group should be able to answer

**If A* already finds the shortest path, what can improve?** The amount of work and time needed to find that path. We cannot find a path shorter than the optimum. A stronger estimate can guide A* more efficiently, so the comparison is between ways of guiding search, not optimal versus “more optimal” paths.

**What do four or eight channels mean?** They are internal feature maps in each convolution block. The model still receives five input planes. Reducing hidden width makes it smaller and usually cheaper, but can weaken its predictions.

**Does the model learn, or only copy Manhattan?** The route-trained model reduced reachable expansions by 7.2% on random 32×32 test maps and 4.7% on wall maps. That is evidence of useful ordering beyond the baseline on those conditions. A model without obstacle input also helps on random maps, so random-map savings alone do not prove obstacle reasoning. These findings do not show a runtime advantage or generalization to every map size.

**Why can fewer explored cells still take more time?** Total time includes preparation, prediction, search, and path output. A small saving in search may not repay a full prediction over the map. Queue operations also cost time, so expansion count alone cannot predict runtime.

**Why not keep training until it wins?** We tried a matched continuation with additional wall and larger-map examples. It lowered validation loss but did not establish a runtime gain. On the measured wall queries, even perfect tie ordering could save only about 4–5% of Manhattan's expansions. Training remains possible, but a new experiment needs a reason to improve a cost that matters.

**Which methods did our approach beat?** Smaller CNNs beat the larger CNN in total time within the width experiment. Learned ties beat Manhattan in expansion count on some test conditions, but did not establish a reliable complete-time win over equally prepared Manhattan. Classical landmarks beat Manhattan on wall validation workloads with sufficient goal reuse. Reverse BFS was faster still at 128 shared-goal starts. These are different comparisons, not one universal ranking.

**Why include reverse BFS if BFS supplies the training labels?** It is a practical competitor when many starts share a goal. It can compute exact next steps once and reuse them. We charge that computation and the returned paths. Labels used only for checking correctness remain outside timing for every method; BFS used to answer queries is inside its method's timer.

**Did the project fail?** The original learned speedup hypothesis is not supported by the current measurements. The research still produced useful findings: safe learned guidance reduces some search work, smaller models reduce its cost, and controlled diagnostics identify limits. We should report that result clearly rather than call expansion savings a runtime victory.

## Evidence to use in the presentation

Use the [architecture diagram](research_architecture.md) to explain inputs, learning, and search before showing results. Then use the [route results](search_learning_results.md) for learning versus runtime, the [width comparison](../results/research_v5/width_comparison.png) for the model-size tradeoff, and the [reuse comparison](../results/research_v6/width_reuse_comparison.png) for the remaining runtime gap.

The [ordering-limit figure](../results/research_v7/ordering_and_pilot.png) explains why another training round did not solve the problem. End with the [landmark results](landmark_timing_results.md) and [changing-goal results](changing_goals_results.md) as validation diagnostics. State that their benefit is classical and workload-dependent. Each figure must retain its baseline, workload, and test/validation label.

Avoid claiming that near-parity proves a win, that every direct learned heuristic preserves optimality, or that landmark speedups are neural-network gains. Do not compare absolute milliseconds from separate experiment rounds as if they were measured together.

The existing 18-slide deck covers research through the runtime/reuse round (v4). The next presentation task is to update it with the later findings, choose a short terminal demonstration, and rehearse the explanation. Recording duration and section allocation should follow the course requirements and the group's actual speaker count.
