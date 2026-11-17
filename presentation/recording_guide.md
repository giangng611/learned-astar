# Presentation and recording guide

Use `learned_astar_research.pptx`, the updated 18-slide deck covering all four research rounds. It is in English, remains editable, and includes speaker notes. Plan roughly 14–17 minutes including the terminal demo; use the references and appendix only when needed. Add the group members' names and adjust the length to the course requirement. The earlier deck is retained separately.

## Suggested section handoffs

| Slides | Section | Approximate time |
| --- | --- | --- |
| 1–4 | Research question, methods, and progression | 3 minutes |
| 5–7 | Model, move objective, and data separation | 3 minutes |
| 8–11 | Learning quality, search effort, and fresh-query time | 3–4 minutes |
| 12–14 | Repeated queries and fair runtime comparisons | 2–3 minutes |
| 15 | Terminal demonstration | 2 minutes |
| 16 | Findings and limits | 1–2 minutes |
| 17–18 | References and detailed intervals | Use when needed |

Assign these sections to the available group members. Explain the distinction between expanding fewer cells and finishing faster. Direct learned heuristics have no optimality guarantee; learned tie ordering keeps Manhattan primary and preserves optimal cost when search completes. For runtime, compare against equally prepared Manhattan as well as the original implementation. BFS with stored next steps is a strong alternative when the map and goal stay fixed.

## Demo sequence

Before recording, activate the project environment and enlarge the terminal font. Keep the prepared map files available so the demo does not depend on typing a map correctly during the presentation.

```sh
python -m learned_astar shell --heuristic-mode tie \
  --checkpoints artifacts/checkpoints/research_v3/seed_7.pt
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

Explain `S`, `G`, `#`, and `.` before running the search. Show cost, expansions, and time, then display the two routes. On the unreachable map, explain that `no_path` is correct. The shell makes fresh predictions and illustrates round 3. It does not measure the reused sessions from round 4; use those benchmark slides for reuse claims.

If the group wants to draw a map live, use `draw`, paste equal-width rows, and end with `.end`. The [terminal guide](../docs/terminal_demo.md) has a small example. The demo supports all three trained models; it does not use pre-recorded routes.

## Before the final recording

Rehearse the handoffs and make sure both the slide text and terminal map are readable at the shared-screen size. Check microphone levels, record a short sample, and watch it back. Use the final result figures and avoid changing model settings based on the test results. If further experiments are added, describe them separately and use a fresh evaluation split for any new tuning.

The group still needs to record the presentation, check the recording, and upload the required files through the course submission page.
