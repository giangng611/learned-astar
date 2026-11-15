# Presentation and recording guide

The deck is in English and is intended for a roughly 12–15 minute group presentation, including a short terminal demo. The PowerPoint file is editable and includes speaker notes. Add the group members' names if needed, and adjust the speaking time to the course requirement.

## Suggested section handoffs

| Slides | Section | Approximate time |
| --- | --- | --- |
| 1–3 | Research question and search methods | 2 minutes |
| 4–6 | Dataset, model, and fair timing | 3 minutes |
| 7–11 | Training, results, generalization, and one example | 4–5 minutes |
| 12 | Terminal demonstration | 2 minutes |
| 13 | Conclusions and limitations | 1 minute |
| 14–15 | References and seed details | Use when needed |

Assign these sections to the available group members. The person presenting results should explain the distinction between expanding fewer cells and finishing faster. When discussing path quality, state that the learned heuristic has no optimality guarantee, even when a particular returned path is optimal.

## Demo sequence

Before recording, activate the project environment and enlarge the terminal font. Keep the prepared map files available so the demo does not depend on typing a map correctly during the presentation.

```sh
python -m learned_astar shell
```

```text
load examples/maps/detour.txt
run
path astar_manhattan
path learned_seed_7
load examples/maps/unreachable.txt
run
quit
```

Explain `S`, `G`, `#`, and `.` before running the search. Show the cost, expanded cells, and time columns, then display the two routes. On the unreachable map, explain that `no_path` is the correct outcome. A single demo timing is an illustration; the recorded experiment supports the conclusions.

If the group wants to draw a map live, use `draw`, paste equal-width rows, and end with `.end`. The [terminal guide](../docs/terminal_demo.md) has a small example. The demo supports all three trained models; it does not use pre-recorded routes.

## Before the final recording

Rehearse the handoffs and make sure both the slide text and terminal map are readable at the shared-screen size. Check microphone levels, record a short sample, and watch it back. Use the final result figures and avoid changing model settings based on the test results. If further experiments are added, describe them separately and use a fresh evaluation split for any new tuning.

The group still needs to record the presentation, check the recording, and upload the required files through the course submission page.
