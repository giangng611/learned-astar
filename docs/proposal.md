# Learning Heuristics for A* Search with Obstacles

Course: CSCI/ARTI 4550/6550, Artificial Intelligence, University of Georgia

Instructor: Dr. Michael Zahorec

Track: Empirical research using code and data

Proposal deadline: November 16, 2026

## Research question

Can a neural network that uses obstacle information help A* search fewer cells and finish faster than Manhattan A* and Weighted A*? How does it affect path length, and does it still work on maps that are different from the training maps?

## Research plan

### Project idea

I want to study how the choice of heuristic affects A* search. A* uses the distance traveled so far and an estimate of the remaining distance to decide where to search next [1]. On a grid, Manhattan distance is a simple estimate based on the horizontal and vertical distance to the goal. However, it does not account for walls. Two cells can look close even when a wall forces a long detour.

My idea is to train a small neural network to estimate the remaining distance using the map and the goal location. I expect this could help A* avoid some unnecessary searching. However, the network also takes time to run and can make incorrect predictions. I want to measure whether it helps overall, rather than assume that adding machine learning will make A* better.

Other researchers have already studied related ideas. Pohl examined the effect of giving more weight to a heuristic [2]. Takahashi and colleagues used neural networks to learn heuristics for path planning [3]. Bhardwaj and colleagues learned which nodes to expand through imitation [4]. Yonetani and colleagues combined a neural network with a differentiable version of A* [5]. My project will use a simpler approach: train a network on exact distances and use its predictions in ordinary A*. The main contribution will be a comparison of when this approach helps and where it has problems.

### Maps and search algorithms

I will begin with 32×32 grids. Each cell will be either free or blocked. Movement will be allowed up, down, left, and right, and every move will cost one. The whole map will be known before searching, and obstacles will not move.

I will compare Dijkstra, breadth first search (BFS), A* with Manhattan distance, Weighted A*, and A* with the learned heuristic. BFS is useful here because all moves have the same cost. For Weighted A*, I will try weights of 1.25, 1.5, and 2.0. Every algorithm will receive the same maps, starting cells, and goals.

Manhattan A* should return a shortest path under these movement rules. Weighted A* may return a longer path, but its weight gives a bound when the required search rules are followed. The learned heuristic has no such guarantee. I will check the paths it returns instead of assuming they are optimal.

### Training the model

I will generate the correct distances by running BFS backward from each goal. One search can find the distance from every cell that can reach that goal. Blocked cells and unreachable cells will be marked separately and left out of the training loss.

I plan to start with 1,200 training maps, 200 validation maps, and 300 test maps. Half of the maps in each group will have 10% obstacles, and half will have 25%. These are starting targets that I will check with a small trial run. I will split whole maps, so cells or goals from the same layout cannot appear in both training and testing.

The model will be a small convolutional neural network, or CNN. Its inputs will describe the obstacles, the goal, each cell's position relative to the goal, and Manhattan distance. It will predict distances for the whole map at once. A* can then look up a prediction when it needs one. This avoids running the network again for every cell.

I will use the validation maps to choose when to stop training and save the model. I will repeat training with three random seeds to see how much the results change. I will settle the model settings before looking at the final test results.

### Measuring the results

The two main measurements will be the number of cells expanded and the total time needed to return a path. Total time will include preparing the network input, running the network, searching, and reconstructing the path. Loading the model will be measured separately. Each timed search will generate its own predictions, even if another search used the same map before.

I will also compare the returned path length with the shortest possible length. I will report how often each algorithm finds an optimal path and how often it succeeds on maps with a valid route. Maps with no route will be kept and reported separately. Cases where the start equals the goal will be checked in tests, but they will not be used to calculate a path length ratio because the shortest distance is zero.

For the network, I will measure average prediction error and how often it predicts a distance that is too large. I will compare these errors with search performance. A model with smaller distance errors may still guide the search poorly, so prediction error alone will not decide whether it works.

After the main comparison, I plan to test maps with 40% obstacles, maps with walls and corridors, and grids of size 16×16 and 64×64. I will change one condition at a time where possible. Each extra condition will initially have 150 maps. I will report how many maps have a route, since a map with more obstacles is not always a harder search problem.

I will repeat timing measurements and report both means and medians. To estimate uncertainty, I will use bootstrap confidence intervals, which repeatedly sample the recorded maps to see how much the average difference changes. Searches from the same map will stay together during this process. Results from all three training seeds, failed searches, and time limits will remain in the report.

### Work plan and expected outcome

I will first build and test the ordinary search algorithms. Then I will generate training data, train a small model, and connect it to A*. A small experiment must work correctly before I run the full comparison. If time is limited, I will finish the 32×32 comparison before adding more map types or model variants.

The project will include the code, data generation steps, saved settings, model checkpoints, results, plots, a presentation, and a report of 2,000 to 4,000 words. I will record how AI tools helped with the work. Even if the learned heuristic is slower than Manhattan A*, that result would help explain when the extra model is useful and when it is unnecessary.

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
