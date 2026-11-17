"""Build the third experiment report from retained measurements."""

import json
from pathlib import Path

import numpy as np


def interval(value, *, percent=False):
    scale = 100 if percent else 1
    return f"{value['mean'] * scale:.3f} [{value['low'] * scale:.3f}, {value['high'] * scale:.3f}]"


def main():
    root = Path("results/research_v3")
    result = json.loads((root / "test/summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    effects = json.loads((root / "test/paired_objective_effects.json").read_text())["splits"]
    training = json.loads(Path("artifacts/checkpoints/research_v3/summary.json").read_text())
    selected = selection["family"]
    training_minutes = sum(r["training_seconds"] for r in training) / 60
    random_result = result["splits"]["test"]["methods"]["selected_mean"]
    wall_result = result["splits"]["walls"]["methods"]["selected_mean"]
    random_reduction = 100 * (1 - random_result["paired_95ci"]["reachable_expansion_ratio"]["mean"])
    wall_reduction = 100 * (1 - wall_result["paired_95ci"]["reachable_expansion_ratio"]["mean"])
    lines = [
        "# Learning objectives and search rules: results",
        "",
        "This round keeps Manhattan A* as the baseline and follows the "
        "[fixed comparison](search_learning.md). It separates two changes: adding a lesson "
        "about useful neighboring moves, and using predictions only to order equal Manhattan "
        "priorities. The [previous experiment](research_results.md) supplies the distance-only "
        "models; all methods are measured again on the same new queries.",
        "",
        f"The selected route-trained tie rule reduced paired reachable-query expansions by "
        f"**{random_reduction:.1f}% on random 32×32 maps and {wall_reduction:.1f}% on wall maps**, "
        "with optimal paths on every reachable query in all five test groups. "
        "It did not improve total runtime. Its mean query times were "
        f"{random_result['completion_ms']['mean']:.3f} ms and "
        f"{wall_result['completion_ms']['mean']:.3f} ms, respectively, versus "
        f"{result['splits']['test']['methods']['astar_manhattan']['completion_ms']['mean']:.3f} "
        "ms and "
        f"{result['splits']['walls']['methods']['astar_manhattan']['completion_ms']['mean']:.3f} "
        "ms for Manhattan. The 64×64 expansion interval includes one, so that condition "
        "does not show a clear reduction.",
        "",
        "## Training and selection",
        "",
        "Three new models use the same 1,200 training maps, 240 checkpoint-validation maps, "
        "16-channel architecture, optimizer, and seeds as the existing distance models. "
        "Each has 10,033 parameters. The added move objective accepts every neighboring step "
        "that reduces exact BFS distance by one. Its coefficient stays fixed at 0.02.",
        "",
        f"All three runs completed 60 epochs in {training_minutes:.2f} "
        "minutes in total. Checkpoint epochs for seeds 7, 17, and 27 are "
        + ", ".join(str(r["best_epoch"]) for r in training)
        + ". "
        "Route checkpoints minimize combined distance and move loss; the earlier distance "
        "checkpoints minimize distance loss alone. The training figure compares the shared "
        "distance component, so the two different total losses are not treated as equivalent.",
        "",
        "A separate 240-map policy-validation set chooses between the two learned tie families. "
        "The score gives random and wall groups equal weight and uses paired reachable-query "
        "expansion ratios. Both families must complete all queries correctly and keep optimal "
        "path cost. A score below one means fewer expansions than Manhattan.",
        "",
        "| Candidate | Validation expansion score | Quality gate |",
        "| --- | ---: | --- |",
    ]
    for candidate in selection["candidates"]:
        lines.append(
            f"| {candidate['family']} / ties | {candidate['expansion_score']:.3f} | "
            f"{'Pass' if candidate['quality_gate_passed'] else 'Fail'} |"
        )
    lines += [
        "",
        f"The fixed rule selected **{selected} / ties**. "
        + (
            "It reduced the validation expansion score below one."
            if selection["improves_expansions_on_validation"]
            else "Its score did not beat Manhattan; selection only identifies "
            "the better learned candidate."
        ),
        "",
        "Only then were 600 fresh test maps generated, giving 2,400 queries. These maps and "
        "the policy-validation maps are disjoint from all 4,340 earlier layout groups, including "
        "rotations. Test results did not change the models or selected rule.",
        "",
        "## Fresh test: learning and search are separate factors",
        "",
        "Times include one fresh prediction per query and use the median of five shuffled "
        "repetitions on one CPU thread. Each CNN row averages three seeds. Expansions and time "
        "in the tables include unreachable queries; path quality uses successful queries. "
        "The default Manhattan baseline breaks ties by larger distance already traveled. "
        "FIFO is an extra control, not the baseline to beat.",
        "",
    ]
    methods = {
        "astar_manhattan": "Manhattan A*",
        "astar_fifo": "Manhattan with FIFO ties",
        "weighted_astar_1.5": "Weighted A* (1.5)",
        "distance_raw_mean": "Distance / direct",
        "distance_tie_mean": "Distance / ties",
        "route_raw_mean": "Route / direct",
        "route_tie_mean": "Route / ties",
        "no_obstacles_seed_7_tie": "No-obstacle / ties, seed 7",
        "oracle_tie": "Exact distance / ties, lookup only",
        "oracle_lookup": "Exact primary heuristic, lookup only",
        "exact_bfs_fresh": "Fresh reverse BFS + search",
    }
    for split, title in (("test", "Random 32×32"), ("walls", "Walls 32×32")):
        lines += [
            f"### {title}",
            "",
            "| Method | Total ms | Expanded cells | Mean path ratio | Optimal paths |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, label in methods.items():
            row = result["splits"][split]["methods"][name]
            lines.append(
                f"| {label} | {row['completion_ms']['mean']:.3f} | "
                f"{row['expanded']['mean']:.1f} | {row['path_ratio']['mean']:.4f} | "
                f"{100 * row['optimal_fraction_of_found']:.1f}% |"
            )
        lines.append("")
    lines += [
        "The exact lookup rows receive BFS distances in advance. Their timings exclude building "
        "that information, so they diagnose search behavior rather than establish practical "
        "speed. Fresh reverse BFS includes its construction. Exact information only in ties "
        "still cannot move a cell ahead of one with a smaller Manhattan priority.",
        "",
        "## What did the added move objective teach?",
        "",
        "Move accuracy chooses the legal neighbor with the lowest predicted distance and "
        "accepts any shortest-path step. Equal predictions use north, east, south, west order. "
        "Branching accuracy excludes cells where every legal move is good. Values average "
        "cells within maps, then maps and seeds; they are diagnostics, not search success rates.",
        "",
        "| Predictor | Random: branching accuracy | Walls: branching accuracy | "
        "Random: MAE | Walls: MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family in ("manhattan", "distance", "route"):
        names = (
            ["manhattan"] if family == "manhattan" else [f"{family}_seed_{s}" for s in (7, 17, 27)]
        )
        values = []
        for metric in ("branch_move_fraction", "mae"):
            for split in ("test", "walls"):
                value = np.mean(
                    [result["splits"][split]["prediction_errors"][n][metric]["mean"] for n in names]
                )
                values.append(
                    f"{100 * value:.1f}%" if metric == "branch_move_fraction" else f"{value:.3f}"
                )
        lines.append(f"| {family} | " + " | ".join(values) + " |")
    lines += [
        "",
        "The following comparison pairs the same test query and seed. "
        "Ratios divide route training by distance-only training, so values below one "
        "favor route training. Accuracy differences are percentage points. Intervals "
        "resample maps 2,000 times, retaining paired queries and the three fixed seeds.",
        "",
        "| Condition | Branching accuracy difference | Direct expansion ratio | "
        "Tie expansion ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, label in (("test", "Random 32"), ("walls", "Walls 32")):
        row = effects[split]
        lines.append(
            f"| {label} | {interval(row['branch_move_fraction_difference'], percent=True)} | "
            f"{interval(row['raw_expansion_ratio'])} | {interval(row['tie_expansion_ratio'])} |"
        )
    lines += [
        "",
        "## Selected rule on every fresh condition",
        "",
        "The ratios below compare the validation-selected tie family with Manhattan A*. "
        "Each reachable query is paired with its baseline; queries and seeds are averaged "
        "within maps before the map average. These ratios need not equal division of the "
        "two overall means above. Brackets show 95% map-bootstrap intervals.",
        "",
        "| Condition | Reachable / queries | Expansion ratio | Total-time ratio | Optimal paths |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("test", "walls", "density_40", "size_16", "size_64"):
        group = result["splits"][split]["methods"]
        row, baseline = group["selected_mean"], group["astar_manhattan"]
        paired = row["paired_95ci"]
        lines.append(
            f"| {split} | {baseline['reachable_queries']} / {baseline['queries']} | "
            f"{interval(paired['reachable_expansion_ratio'])} | "
            f"{interval(paired['reachable_runtime_ratio'])} | "
            f"{100 * row['optimal_fraction_of_found']:.1f}% |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The added lesson improved branching-move accuracy from 86.0% to 94.4% on random "
        "maps and from 72.0% to 93.3% on wall maps. Wall distance error increased from "
        "7.25 to 8.31 steps at the same time. This is evidence that learning useful ordering "
        "differs from minimizing distance error. All three route-trained seeds reduced the "
        "paired expansion ratio below one on both 32×32 groups.",
        "",
        "Learning local moves and ranking the whole search frontier are different tasks. "
        "The added objective can improve neighboring preferences without producing a "
        "globally accurate distance field. The direct-heuristic controls reveal that "
        "tradeoff; using the same fields only for ties retains Manhattan's optimality "
        "guarantee whenever search completes.",
        "",
        "Fewer expansions do not establish a faster method. CNN prediction and queue "
        "ordering have costs that the total-time measurements include. The test covers "
        "new layouts from the two training families plus unseen sizes and density. "
        "Confidence intervals describe variation across maps for these three models, "
        "not uncertainty across all possible training runs. The one-seed no-obstacle "
        "control is a diagnostic rather than a complete architecture comparison.",
        "",
        "The no-obstacle model also reduces random-map expansions, so that result alone "
        "cannot establish learned obstacle reasoning. It does not show a clear reduction "
        "on wall maps. Even exact-distance tie lookup is slower than ordinary Manhattan "
        "in this implementation. Prediction reuse alone therefore may not remove the "
        "runtime gap; queue-ordering costs also need attention. A next experiment could "
        "profile those costs and compare shared map-goal predictions with equally shared "
        "BFS preprocessing, keeping setup time in the comparison.",
        "",
        "All methods in this round use the same updated search implementation. Absolute "
        "timings should be compared within this round, not against earlier runs on "
        "different maps and system loads. The first wall query below is fixed by dataset "
        "order; it illustrates the routes without representing average performance.",
        "",
        "## Figures and retained evidence",
        "",
        "![Search comparison](../results/research_v3_figures/search_rules.png)",
        "",
        "![Move and distance diagnostics](../results/research_v3_figures/local_moves.png)",
        "",
        "![Training objectives](../results/research_v3_figures/training_objectives.png)",
        "",
        "![First wall query](../results/research_v3_figures/example_paths.png)",
        "",
        f"Evidence checks verified {verification['validation']['raw_runs']:,} validation "
        f"and {verification['test']['raw_runs']:,} test runs, their paired method coverage, "
        "five repetitions, median summaries, checkpoint hashes, and layout separation. "
        "The [result archive](../results/research_v3/) retains summaries, compressed "
        "query and map records, manifests, selection, and settings. "
        "[New checkpoints and histories](../artifacts/checkpoints/research_v3/) and "
        "PNG/PDF figures are included. Full timing repetitions and generated map arrays "
        "remain local. See the [experiment plan](search_learning.md#reproduction) to rerun.",
        "",
    ]
    Path("docs/search_learning_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
