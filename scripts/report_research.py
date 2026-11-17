"""Write the measured follow-up report from retained experiment evidence."""

import json
from pathlib import Path


def main():
    root = Path("results/research_v2")
    result = json.loads((root / "test/summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    verified = json.loads((root / "verification.json").read_text())
    profile = json.loads((root / "data_profile.json").read_text())
    variant, mode = selection["variant"], selection["mode"]
    wall = result["splits"]["walls"]
    wall_new = wall["methods"]["selected_mean"]
    wall_ref = wall["methods"]["astar_manhattan"]
    direct_mae = wall["prediction_errors"]["direct_seed_7"]["mae"]["mean"]
    manhattan_mae = wall["prediction_errors"]["manhattan"]["mae"]["mean"]
    names = {
        "astar_manhattan": "Manhattan A*",
        "weighted_astar_1.5": "Weighted A* (1.5)",
        "original_mean": "Original CNN, 3 seeds",
        "direct_seed_7_raw": "Direct CNN, seed 7, raw",
        "direct_seed_7_parity": "Direct CNN, seed 7, rounded",
        "residual_seed_7_raw": "Detour CNN, seed 7, raw",
        "residual_seed_7_parity": "Detour CNN, seed 7, rounded",
        "no_obstacles_seed_7_raw": "No-obstacle control, seed 7",
        "selected_mean": "Selected CNN, 3 seeds",
        "oracle_lookup": "Exact-distance lookup",
        "exact_bfs_fresh": "Fresh reverse BFS + search",
    }
    lines = [
        "# Follow-up research results",
        "",
        "This experiment follows the [fixed plan](research_followup.md). "
        "It keeps the [first experiment](report.md) as the original baseline and evaluates "
        "new models on newly generated maps. Results below come from the saved measurements, "
        "rather than from the single-query terminal demo.",
        "",
        "The network learned useful information about wall detours, but this round did not "
        "produce a faster replacement for Manhattan A*. On fresh wall maps, the seed-7 direct "
        f"model has mean distance error {direct_mae:.2f} steps, "
        f"compared with {manhattan_mae:.2f} for Manhattan. "
        "The selected three-seed configuration takes "
        f"{wall_new['completion_ms']['mean']:.3f} ms per "
        f"query versus {wall_ref['completion_ms']['mean']:.3f} ms for Manhattan A*, and returns "
        f"an optimal path in {100 * wall_new['optimal_fraction_of_found']:.1f}% of wall queries. "
        "It had already failed the validation quality gate.",
        "",
        "## What changed",
        "",
        "The new training set has 800 random maps and 400 wall maps, with two goals per map. "
        "Validation has 240 separate maps. All new networks have 10,033 parameters, compared "
        "with 38,497 in the original model. Three controlled runs compare full distance "
        "prediction, a learned detour added to Manhattan, and a detour model without obstacle "
        "inputs. The optimizer, data, width, and screening seed are shared.",
        "",
        f"The original random training maps have a mean detour of "
        f"{profile['original']['random']['mean_detour']:.2f} steps per reachable cell. "
        f"The new wall maps have a mean detour of {profile['mixed']['walls']['mean_detour']:.2f} "
        "steps. This changes the examples the network learns from, rather than only training "
        "longer on the same distribution.",
        "",
        "## Validation choice",
        "",
        f"The predeclared rule selected the **{variant} model with {mode} predictions**. "
        + (
            "It passed the path-quality gate on both validation groups."
            if selection["quality_gate_passed"]
            else "No candidate passed the path-quality gate on both validation groups. "
            "The fallback rule chose the lowest worst-group mean path ratio, "
            "then the expansion score. "
            "This is a failed quality gate, not a successful validation claim."
        ),
        "",
        "| Candidate | Search form | Mean reachable expansion ratio | "
        "Worst-group mean path ratio | Passes gate |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for c in selection["candidates"]:
        lines.append(
            f"| {c['variant']} | {c['mode']} | {c['expansion_score']:.3f} | "
            f"{c['worst_mean_path_ratio']:.4f} | {'Yes' if c['eligible'] else 'No'} |"
        )
    lines += [
        "",
        "The expansion score gives random and wall maps equal weight. "
        "The selected architecture was then trained with seeds 17 and 27, without "
        "changing its search form. Only after that were 600 fresh test maps generated. "
        "The archive verifies that the 2,300 original, 1,440 new development, and 600 "
        "fresh test layout groups do not overlap, including rotations.",
        "",
        "## Fresh test comparisons",
        "",
        "Times include a fresh prediction and search for every query. Rows average "
        "per-query medians of five repetitions. The exact-distance lookup is an oracle "
        "reference: its distances are supplied in advance, so its time is not comparable "
        "as an end-to-end method. The separate fresh-BFS row includes constructing them.",
        "",
    ]
    for split, title in (("test", "Random 32×32 maps"), ("walls", "Wall 32×32 maps")):
        lines += [
            f"### {title}",
            "",
            "| Method | Total ms | Expanded cells | Mean path ratio | Optimal paths |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, label in names.items():
            m = result["splits"][split]["methods"][name]
            lines.append(
                f"| {label} | {m['completion_ms']['mean']:.3f} | "
                f"{m['expanded']['mean']:.1f} | {m['path_ratio']['mean']:.4f} | "
                f"{100 * m['optimal_fraction_of_found']:.1f}% |"
            )
        m = result["splits"][split]["methods"]["selected_mean"]
        o = result["splits"][split]["methods"]["oracle_lookup"]
        paired = m["paired_95ci"]["reachable_expansion_ratio"]
        headroom = o["paired_95ci"]["reachable_expansion_ratio"]
        lines += [
            "",
            f"On reachable queries, the selected model's mean paired expansion ratio "
            f"is **{paired['mean']:.3f}** (95% map-bootstrap interval "
            f"{paired['low']:.3f}–{paired['high']:.3f}). The exact-distance reference has "
            f"ratio {headroom['mean']:.3f}. Ratios below one indicate fewer expansions "
            "than Manhattan A*. A mean of paired ratios differs from dividing the "
            "two means in the table.",
            "",
            f"Selected models spend {m['prediction_ms']['mean']:.3f} ms on prediction "
            f"and {m['search_ms']['mean']:.3f} ms on search on average. These components "
            "are summarized separately, so their medians need not add exactly to total time. "
            f"The worst observed path ratio is {m['path_ratio']['max']:.3f}.",
            "",
        ]
    lines += [
        "## Did the models learn obstacle information?",
        "",
        "The table reports mean absolute error in steps, first averaging reachable cells "
        "within each map. These are raw distance predictions, before search rounding.",
        "",
        "| Predictor | Random maps MAE | Wall maps MAE |",
        "| --- | ---: | ---: |",
    ]
    error_names = {
        "manhattan": "Manhattan",
        "direct_seed_7": "Direct, seed 7",
        "direct_seed_7_hidden": "Direct with obstacles hidden at inference",
        "residual_seed_7": "Learned detour, seed 7",
        "residual_seed_7_hidden": "Learned detour with obstacles hidden at inference",
        "no_obstacles_seed_7": "Learned detour trained without obstacles",
    }
    for name, label in error_names.items():
        values = [
            result["splits"][s]["prediction_errors"][name]["mae"]["mean"] for s in ("test", "walls")
        ]
        lines.append(f"| {label} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += [
        "",
        "A secondary diagnostic checks neighboring reachable cells. "
        "A consistent heuristic cannot differ by more than one across a unit-cost edge. "
        "This diagnostic was added after the validation failure and did not affect selection.",
        "",
        "| Predictor | Wall edges exceeding one step | Mean excess per wall edge |",
        "| --- | ---: | ---: |",
    ]
    for name in ("manhattan", "direct_seed_7", "residual_seed_7", "no_obstacles_seed_7"):
        errors = result["splits"]["walls"]["prediction_errors"][name]
        lines.append(
            f"| {error_names[name]} | {100 * errors['neighbor_violation_fraction']['mean']:.1f}% | "
            f"{errors['mean_neighbor_violation']['mean']:.3f} steps |"
        )
    lines += [
        "",
        "Violation frequency is not a direct measure of search quality. For example, "
        "scaling Manhattan upward can create many violations while behaving like Weighted A*. "
        "The table diagnoses a missing consistency guarantee; it does not establish the cause "
        "of the observed expansion counts.",
        "",
        "The separately trained no-obstacle model controls for input information. "
        "Hiding obstacles only at inference provides a second check, but also shifts the "
        "model's input distribution. These controls use one screening seed, so they "
        "should not be presented as a precise estimate over all training seeds. "
        "Lower prediction error does not by itself establish faster or more accurate search.",
        "",
        "## Generalization and seed variation",
        "",
        "Random and wall 32×32 maps are new layouts from the training families. "
        "The different sizes and 40% density were not used in training.",
        "",
        "| Fresh condition | Reachable / queries | Expansion ratio | "
        "Total-time ratio | Mean path ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("test", "walls", "density_40", "size_16", "size_64"):
        m = result["splits"][split]["methods"]["selected_mean"]
        ref = result["splits"][split]["methods"]["astar_manhattan"]
        paired = m["paired_95ci"]
        lines.append(
            f"| {split} | {ref['reachable_queries']} / {ref['queries']} | "
            f"{paired['reachable_expansion_ratio']['mean']:.3f} | "
            f"{paired['reachable_runtime_ratio']['mean']:.3f} | {m['path_ratio']['mean']:.4f} |"
        )
    lines += [
        "",
        "Expansion and time ratios in this table use reachable queries. "
        "The saved summaries also include every unreachable query and separate correctness rates.",
        "",
        "| Seed | Random: expanded | Random: ms | Walls: expanded | Walls: ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in (7, 17, 27):
        name = f"{variant}_seed_{seed}_{mode}"
        a, b = [result["splits"][s]["methods"][name] for s in ("test", "walls")]
        lines.append(
            f"| {seed} | {a['expanded']['mean']:.1f} | {a['completion_ms']['mean']:.3f} | "
            f"{b['expanded']['mean']:.1f} | {b['completion_ms']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "This round separates three questions: whether exact distances leave room to reduce "
        "search, whether the network uses obstacles, and whether that information improves "
        "the full query. A model may succeed at one and still fail at another. The fresh "
        "test results were not used to change its architecture, checkpoint, or rounding rule.",
        "",
        "The comparison with the original model changes training data, network width, and "
        "epoch budget together. It measures the new configuration as a whole; it cannot "
        "assign the improvement or decline to one of those changes. The direct-versus-detour "
        "and obstacle controls are the more focused comparisons. All runtime findings "
        "apply to this CPU and workload, with a fresh prediction per query. Rounded or "
        "residual predictions still have no optimality guarantee.",
        "",
        "The wall-query mean expansion count falls slightly, but the mean paired expansion "
        "ratio rises above one. These weight queries differently: a small number of expensive "
        "queries can dominate the unpaired mean. The paired result therefore does not support "
        "a broad reduction in search effort. On random maps, even the no-obstacle control "
        "reduces expansions despite its larger distance error. That finding cautions against "
        "attributing every search improvement to learned obstacle reasoning.",
        "",
        "A useful next experiment would change how predictions affect search, rather than "
        "only minimizing distance error. One option is to keep Manhattan as the main A* "
        "priority and use learned estimates only to break ties. This preserves the primary "
        "admissible priority, but whether it reduces work needs a new validation comparison. "
        "The current fresh test is now observed and must not become a tuning set for a claim "
        "of fresh-test performance.",
        "",
        "## Evidence and figures",
        "",
        f"Verification covers {verified['test']['queries']:,} fresh test queries, "
        f"{verified['test']['raw_runs']:,} timed runs, paired methods, median summaries, "
        "checkpoint hashes, and separation of layout groups. Raw repetitions stay in "
        "the local run directory; compact evidence and model histories are retained in "
        "`results/research_v2/` and `artifacts/checkpoints/research_v2/`.",
        "",
        "![Controlled training](../results/research_v2_figures/training_controls.png)",
        "",
        "![Fresh search comparison](../results/research_v2_figures/search_comparison.png)",
        "",
        "![Obstacle controls](../results/research_v2_figures/obstacle_controls.png)",
        "",
        "![First wall map](../results/research_v2_figures/distance_fields.png)",
        "",
        "The example is the first fresh wall map and first query in manifest order. "
        "It was not chosen because the model performed well on it.",
        "",
    ]
    Path("docs/research_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
