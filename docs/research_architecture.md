# Implementation plan

November 11, 2026

This document describes how I plan to build the project. As of November 13, the Python environment and configuration checks are ready. The search code, dataset, and model are still to come. The [proposal](proposal.md) explains the research question and related papers.

## Starting point

The project now has an installable Python package, a small configuration file, and tests for its settings. The local environment uses Python 3.14.3 and PyTorch 2.15.0. A small CPU convolution passed both the forward calculation and the gradient check. The exact installed package versions are saved in `requirements-lock.txt`.

I will start with 32×32 grids and one small CNN. The first goal is to get a correct comparison working. Larger maps and extra model ideas can come after that.

## Code organization

I plan to put the Python code in `src/learned_astar/`. The `environments/` folder will describe the grids and generate maps. The `search/` folder will contain the search algorithms. Dijkstra, Manhattan A*, Weighted A*, and learned A* will share the same priority queue code so differences in implementation do not unfairly favor one method. BFS will use a separate ordinary queue.

The `data/` folder will create exact distance labels and keep track of the training, validation, and test maps. The network will go in `models/`, and the training loop will go in `training/`. The `evaluation/` folder will run comparisons and save measurements. The `visualization/` folder will make plots from those saved results.

I will keep settings in `configs/`, commands for running each step in `scripts/`, and checks in `tests/`. The project will run from the command line. It does not need a website or a graphical interface.

A search result will contain the path, its cost, the time taken, and the search counts. It will also say whether the search found a path, proved that no path exists, or reached a time or work limit. Invalid inputs and invalid model outputs will be reported as errors.

## Grid rules

A cell will use `(row, column)` coordinates starting at zero. A blocked cell cannot be entered. The search will check neighbors in the order north, east, south, and west. There will be no diagonal movement or movement across the map boundary. Every move costs one.

The start and goal must be free cells inside the map. If they are the same valid cell, the result will be a path containing that cell and a cost of zero. Every returned path will be checked for correct endpoints, legal moves, obstacles, and total cost.

## Search rules

Dijkstra will use a heuristic of zero. Standard A* will use Manhattan distance. Weighted A* will use `f = g + w * h`, where `g` is the distance traveled so far and `h` is Manhattan distance. I will test weights of 1.25, 1.5, and 2.0. Learned A* will use `f = g + h_model`.

The priority queue will choose the smallest `f`. If two entries have the same value, it will prefer the larger `g`, then the entry added first. Keeping this order fixed will make repeated runs easier to compare.

The search will store the best distance found for each cell. If a shorter route is found, it will update that distance and add a new queue entry. Older entries with the wrong distance will be skipped. A cell that was already expanded can be opened again if a shorter route reaches it. This matters because a learned heuristic can give inconsistent estimates for neighboring cells.

A search will finish when the goal is removed from the queue with its current best distance. Finding the goal as a neighbor is not enough to stop. If the queue becomes empty, there is no route. If a time limit is reached, the result will say so rather than claim that no path exists.

I will count a cell as expanded when its current queue entry is removed and its neighbors are checked. This count includes repeated expansions but excludes outdated entries and the final goal removal. I will also save the number of distinct cells expanded. The generated count will include every legal neighbor considered, including repeated visits. Queue insertions and outdated entries will have separate counts.

## What the algorithms guarantee

BFS and Dijkstra should find a shortest path because every move has the same positive cost. Manhattan distance never overestimates the remaining distance under these grid rules. With the search rules above, Manhattan A* should also find a shortest path.

Weighted A* can return a longer path. With Manhattan distance, a weight of at least one, reopening when needed, and stopping when the goal is removed from the queue, the returned cost is at most the weight times the optimal cost. I will check this property in tests.

A prediction from the network does not have the same guarantee. Making it nonnegative or mixing it with Manhattan distance does not prove that it is safe. Without a time limit, finite predictions and the reopening rule still allow the search to find a route if one exists on the finite grid. That route may be longer than necessary.

If there is time for another version, I can use the learned value only when two cells have the same Manhattan A* priority. This keeps the main priority unchanged and preserves optimal paths. I would still need to measure whether the benefit is worth running the network.

## First model

The network will take five input channels: the obstacle map, a map marking the goal, the row distance to the goal, the column distance to the goal, and Manhattan distance. Row and column distances will keep their signs and be divided by the corresponding grid dimension minus one. Manhattan distance will be divided by `S = max(H + W - 2, 1)`.

I plan to use five convolution layers with 3×3 filters and 32 output channels each. Their dilation values will be 1, 2, 4, 8, and 16. Dilation lets a layer look at cells farther apart without adding many more weights. Each layer will use matching padding and ReLU. A final 1×1 layer and softplus will produce nonnegative distances.

This model has about 39,000 parameters. Each output can use a region of up to 63×63 cells. That covers the main 32×32 maps, but it does not mean the model will understand every detour. Some distant obstacles on 64×64 maps will be outside that region.

The training target will be the exact distance divided by `S`. I will not clip the target at one because a detour can be longer than the grid's height plus width. During search, predictions will be multiplied by `S`, and the goal prediction will be set to zero.

I will use Smooth L1 loss on reachable free cells. The loss will first be averaged within each map and goal, then across the batch. This stops maps with more reachable cells from automatically getting more weight. Blocked and unreachable cells will be masked out of the loss. Their predictions will not be used to decide that a cell can be skipped during search.

My starting training settings are Adam with a learning rate of 0.001, a batch size of 16, and a limit of 100 epochs. I will stop early if validation loss does not improve for 10 epochs. These settings can change during the small trial run. I will save the best validation model for each of three chosen random seeds and report all three.

## Making the dataset

For a random map with obstacle fraction `p`, I will choose exactly `round(p * H * W)` blocked cells at random without replacement. Maps with walls and gaps will use a separate generator. I will record their actual obstacle fractions because similar fractions do not necessarily mean similar difficulty.

I will check the dimensions and obstacle layout for duplicates before dividing the maps into training, validation, and testing. All goals and examples from one layout will stay in the same group. If I later add rotated copies, they will stay with the original map too.

The starting plan is 1,200 training maps, 200 validation maps, and 300 test maps at 32×32. Each group will be split equally between 10% and 25% obstacles. I will choose two free goals per training and validation map. Reverse BFS will give the exact distances and mark cells that cannot reach the goal.

For each test map, I will choose one free goal and four different free starting cells, excluding the goal. I will not require a route to exist when choosing them. Maps without enough free cells will be recorded explicitly. I will keep unreachable searches instead of replacing them with easier examples.

The extra conditions will each start with 150 test maps. These will cover 32×32 maps with 40% obstacles, 16×16 and 64×64 maps with 25% obstacles, and 32×32 maps with walls and corridors. I will confirm the final counts after a small training and validation trial, before looking at test results. These counts are practical starting estimates, not a claim that they guarantee statistical significance.

I will record obstacle fraction, the fraction of cells that can reach the goal, and shortest path lengths. For valid paths with different endpoints, I will also compare the true distance with Manhattan distance to see how much of a detour is needed. Exact distances will stay in the evaluation code and will not be passed to the learned search.

## Fair timing

The main timing comparison will run on the CPU with fixed thread settings. Time will include preparing inputs, running the model, preparing its output for search, searching, and building the final path. Loading files, calculating evaluation labels, writing results, and checking completed paths will be outside the timer for every algorithm. Model loading and training time will be reported separately.

I will warm up the operations, vary the order of algorithms, and time each query five times. I will use the median of those repetitions for the main analysis. Every query will create a fresh prediction map. Repeated runs will not count as new independent test examples.

If I later test reuse of predictions for several starts with one goal, I will also compare against reverse BFS followed by exact distance lookups. Both methods must include the cost of preparing their distance maps. I will also time reverse BFS separately to understand how much a full neural prediction costs compared with computing exact distances.

## Reading the results

The main comparison will be against Manhattan A* for expansions and total runtime. Weighted A* will show whether a simple change in weight offers a better balance between speed and path length.

I will report means, medians, the middle half of the measurements, and unusually slow cases. For 95% bootstrap confidence intervals, I will resample whole maps and keep their starts and algorithm results together. For the main interval, I will average the three trained models within each map. I will also show each model's results separately. This interval describes variation across maps for those trained models; it does not capture every possible training outcome.

The path cost ratio will be the returned cost divided by the shortest cost. I will calculate it only for successful searches with a positive shortest cost and report the number of failures alongside it. Unreachable queries will have separate results. Searches that reach a limit will be marked incomplete, and their recorded time will not be treated as the time needed to finish.

Prediction error will be measured on reachable cells using MAE, RMSE, and the frequency and size of overestimates. I will summarize errors by map. Any relationship between prediction error and search performance will be treated as an observation, not proof that one caused the other.

## Schedule and checks

The immediate deadline is the proposal on November 16. I plan to review the question and scope from November 11 to 13, then revise the proposal and prepare it for submission from November 14 to 16. The final report and presentation dates still need to be confirmed.

After the proposal, I estimate two to three focused workdays for grid generation and ordinary search. Tests will cover empty grids, blocked grids, missing routes, equal start and goal, corridors, multiple shortest paths, boundaries, repeated runs, and reopening. BFS will provide an independent check on the priority queue algorithms.

The data and model stage may take three to four workdays. I will check that splits do not overlap, labels are correct, and a saved model can be loaded. Connecting the model to search may take another one to two workdays. A small complete run must work before larger experiments start.

I estimate three to four workdays for experiments and analysis, plus the actual computing time, and another three to four for the report and presentation. These are rough estimates. If the final deadline is too close, I will finish the main 32×32 comparison and reduce the extra conditions.

The main concerns are slow model inference, incorrect distance predictions, missing difficult cases, and testing on maps too similar to the training data. The timing rules, path checks, full result records, and separate map groups are meant to address these concerns. If the learned method performs poorly, I will include that result and try to explain it.

I will review each set of changes before committing and pushing. No large experiment should start until the smaller checks pass.
