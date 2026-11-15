# Terminal demonstration

The demo uses the trained checkpoints in `artifacts/checkpoints/`. Run it from the project folder after installing the model dependencies.

```sh
python -m learned_astar shell
```

Type `draw`, paste equal-width rows, and finish with `.end` or an empty line. Use `S` for the start, `G` for the goal, `#` for an obstacle, and `.` for a free cell. A map needs exactly one start and one goal.

```text
astar> draw
map> S...#.......
map> ....#.......
map> ....#.......
map> ....#....G..
map> ............
map> .end
astar> run
astar> path astar_manhattan
astar> path learned_seed_7
```

The table shows status, path cost, cost divided by the optimal cost, expanded cells, and query time. The path view marks the returned route with `*`. All methods use the same map and endpoints. Each learned run makes a new prediction. Its query time includes this prediction, while checkpoint loading is separate. Single-query timings illustrate the program; use the full experiment for performance claims.

Useful shell commands:

| Command | Action |
| --- | --- |
| `load examples/maps/detour.txt` | Load a map with obstacles and a detour. |
| `load examples/maps/unreachable.txt` | Show a case where no route exists. |
| `generate 16 16 0.25 4550` | Generate a repeatable map without filtering unreachable endpoints. |
| `show` | Display the current map. |
| `run` | Compare all loaded models and baseline methods. |
| `path learned_seed_17` | Display that model's returned route after a run. |
| `save outputs/demo.json` | Save the current comparison, including paths. |
| `quit` | End the shell session. |

For a single command that is easy to rehearse:

```sh
python -m learned_astar solve --map examples/maps/detour.txt --show-paths --output outputs/demo.json
```

For the recording, enlarge the terminal font and use a small map. Explain the map and the expected detour, run the comparison, then show Manhattan A* and one learned path. Finish with the unreachable example to explain why `no_path` is a valid outcome. Keep the demo to about two minutes and use the saved map files if typing a map live would interrupt the explanation.
