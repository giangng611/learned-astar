# Learning Heuristics for A* Search with Obstacles

Course: CSCI/ARTI 4550/6550, Artificial Intelligence, University of Georgia

Instructor: Dr. Michael Zahorec

Track: Empirical research using code and data

Proposal deadline: November 16, 2026

## Research question

Can a neural network that uses obstacle information help A* search fewer cells and finish faster than Manhattan A* and Weighted A*? How does it affect path length, and does it work on maps different from the training maps?

## Research plan

### Project idea

A* combines distance traveled with an estimate of the distance remaining [1]. Manhattan distance is simple to calculate on a grid, but it ignores walls that may force long detours. I will train a small neural network to predict remaining distances using obstacle information, then use those predictions in ordinary A*. The comparison will test whether reduced searching makes up for inference time and prediction errors.

Related work includes weighted heuristic search [2], learned pathfinding heuristics [3], search decisions learned through imitation [4], and differentiable Neural A* [5]. My approach trains the distance predictor separately from search. The contribution will be an empirical comparison of when this simpler approach helps and where it fails.

### Maps and search algorithms

The main comparison uses fully known, static 32×32 grids with four movement directions and a cost of one per move. Every method receives the same map, start, and goal. Baselines are breadth first search (BFS), Dijkstra, Manhattan A*, and Weighted A* with weights of 1.25, 1.5, and 2.0.

BFS, Dijkstra, and Manhattan A* provide shortest-path checks. Weighted A* can trade path length for speed within its weight bound when the required search rules are followed. Learned A* has no optimality guarantee, so returned paths and costs must be checked.

### Training the model

Reverse BFS from each goal will give exact distances for all reachable cells. Blocked and unreachable cells will be marked separately and excluded from training loss. The starting dataset has 1,200 training maps, 200 validation maps, and 300 test maps. Each split is divided equally between 10% and 25% obstacles. Whole layouts stay in one split to avoid overlap between training and testing. A small pilot will check the planned counts.

A small convolutional neural network (CNN) will receive obstacles, a goal marker, signed row and column offsets from the goal, and Manhattan distance. It predicts the whole distance map once per query, allowing A* to look up values during search. Validation results will guide stopping and checkpoint selection. Training will use three random seeds, and settings will be fixed before final test evaluation.

### Measuring the results

The main measures are expanded cells and total query time, including input preparation, inference, search, and path reconstruction. Model loading is measured separately, and each timed query produces fresh predictions. I will report repeated timings using means and medians.

Path quality will be measured by returned cost divided by optimal cost, the fraction of optimal paths, and success rate on reachable queries. Unreachable cases are kept and reported separately. Start equal to goal is checked for correctness but excluded from cost ratios. Failed and limited searches remain in the results.

Prediction error and the frequency and size of overestimates will be compared with search behavior. Lower prediction error alone does not establish better search. Bootstrap confidence intervals will estimate uncertainty by resampling whole maps, keeping their queries together. Results from all three training seeds will be shown.

Extra tests will cover 40% obstacles, walls and corridors, and 16×16 and 64×64 grids, starting with 150 maps per condition. I will change one condition at a time where possible and report route availability, since higher obstacle density does not always mean a harder query.

### Work plan and expected outcome

The work proceeds from tested baselines to data generation, training, integration, and experiments. A small complete run must pass before the full comparison. The main 32×32 experiment takes priority if time is limited.

Deliverables include code, data generation steps, settings, checkpoints, results, plots, a presentation, and a report of 2,000 to 4,000 words. A slower or less accurate learned heuristic is still a useful result if the analysis explains its costs and limitations.

## Annotated bibliography

### [1] A* search

Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics, 4*(2), pages 100 to 107. [DOI](https://doi.org/10.1109/TSSC.1968.300136).

This paper introduces the foundation of A*. I will use it to explain how a heuristic guides the search and why the assumptions behind an optimal path matter. These assumptions are especially relevant when replacing Manhattan distance with a prediction that could be wrong. The citation can also be checked in [Nilsson's publication list](https://ai.stanford.edu/~nilsson/publications.html).

### [2] Weighted heuristic search

Pohl, I. (1970). Heuristic search viewed as path finding in a graph. *Artificial Intelligence, 1*(3 to 4), pages 193 to 204. [DOI](https://doi.org/10.1016/0004-3702(70)90007-X).

This paper studies heuristic search and the effect of changing the weight placed on a heuristic. It is useful for my Weighted A* comparison. I need to see whether a simple change to Manhattan A* can give similar benefits to the neural network. The [publisher record](https://www.sciencedirect.com/science/article/pii/000437027090007X) provides the citation and abstract.

### [3] Learning a heuristic

Takahashi, T., Sun, H., Tian, D., & Wang, Y. (2019). Learning heuristic functions for mobile robot path planning using deep neural networks. *Proceedings of the International Conference on Automated Planning and Scheduling, 29*(1), pages 764 to 772. [DOI](https://doi.org/10.1609/icaps.v29i1.3545).

This paper is closely related to my idea because it uses a neural network to learn a heuristic for path planning. It also discusses how to train the network. I will use it to guide my design, but I still need to test whether a smaller model helps on my grids. A record of the paper is available from [MERL](https://merl.com/publications/TR2019-072).

### [4] Learning which node to expand

Bhardwaj, M., Choudhury, S., & Scherer, S. (2017). Learning heuristic search via imitation. In *Proceedings of the 1st Annual Conference on Robot Learning*, *Proceedings of Machine Learning Research, 78*, pages 271 to 280. [Paper](https://proceedings.mlr.press/v78/bhardwaj17a.html).

The authors train a search policy by imitating an oracle that knows the environment. Their approach learns search decisions, while mine predicts distances. This difference is useful for my project because it shows why I need to measure actual search behavior as well as prediction accuracy.

### [5] Neural A*

Yonetani, R., Taniai, T., Barekatain, M., Nishimura, M., & Kanezaki, A. (2021). Path planning using Neural A* search. In *Proceedings of the 38th International Conference on Machine Learning*, *Proceedings of Machine Learning Research, 139*, pages 12029 to 12039. [Paper](https://proceedings.mlr.press/v139/yonetani21a.html).

This paper combines a convolutional network with a version of A* that can be used during training. My project will keep ordinary A* and train the network separately using exact distances. I will use the paper to understand a related approach and discuss how my simpler setup differs from it.
