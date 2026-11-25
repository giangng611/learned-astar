"""Verify, archive, and summarize the fixed model-width experiment."""

import argparse
import gzip
import json
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from archive_research import verify_evidence
from run_lightweight import fingerprints

from learned_astar.data import read_manifest
from learned_astar.research.experiment import digest, save_json
from learned_astar.research.lightweight import choose_width, paired_widths
from learned_astar.research.plot import save

SPLITS = {
    "test": "Random 32",
    "walls": "Walls 32",
    "density_40": "Dense 32",
    "size_16": "Random 16",
    "size_64": "Random 64",
}


def load(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def verify(root, config):
    if fingerprints(config) != load(root / "fingerprints.json"):
        raise ValueError("Frozen sources changed")
    selection = load(root / "selection.json")
    for key, name in (
        ("protocol_sha256", "protocol.toml"),
        ("validation_summary_sha256", "validation/summary.json"),
        ("validation_pairs_sha256", "validation/paired_widths.json"),
    ):
        if selection[key] != digest(root / name):
            raise ValueError("Selection evidence changed")
    if any(
        selection[k] != v
        for k, v in choose_width(
            load(root / "validation/summary.json"),
            load(root / "validation/paired_widths.json"),
            config,
        ).items()
    ):
        raise ValueError("Selected width changed")
    result = {}
    for stage, data_name in (("validation", "policy_validation"), ("test", "fresh_test")):
        directory = root / stage
        checked = verify_evidence(directory)
        summary = load(directory / "summary.json")
        if summary["settings"]["dataset_manifest_sha256"] != digest(
            root / data_name / "manifest.jsonl"
        ):
            raise ValueError("Measured dataset changed")
        rows = read_rows(directory / "queries.jsonl.gz")
        expected = config["evaluation"]["repetitions"]
        groups = defaultdict(list)
        endpoints = {}
        for row in read_rows(directory / "runs.jsonl.gz"):
            key = row["layout_id"], row["query"]
            endpoint = row["start"], row["goal"], row["optimal_cost"]
            if endpoints.setdefault(key, endpoint) != endpoint:
                raise ValueError("Methods measured different endpoints")
            optimal = row["optimal_cost"]
            if (row["status"], row["cost"]) != (
                ("found", optimal) if optimal >= 0 else ("no_path", None)
            ):
                raise ValueError("Incorrect path cost or reachability")
            signature = tuple(
                row[k]
                for k in (
                    "status",
                    "cost",
                    "expanded",
                    "unique_expanded",
                    "generated",
                    "queue_insertions",
                    "stale_pops",
                    "reexpanded",
                    "max_frontier",
                )
            )
            groups[*key, row["method"]].append(signature)
        if any(len(g) != expected or len(set(g)) != 1 for g in groups.values()):
            raise ValueError("Search behavior changed across repetitions")
        checked.update(
            all_optimal_and_correct=True, all_counters_repeat=True, all_methods_share_endpoints=True
        )
        if paired_widths(directory, config) != load(directory / "paired_widths.json"):
            raise ValueError("Paired comparisons do not reproduce")
        # Independent scalar arithmetic checks the reported paired runtime means.
        indexed = {(r["layout_id"], r["query"], r["method"]): r for r in rows}
        for split in summary["splits"]:
            for width in config["widths"]:
                values = defaultdict(list)
                for row in rows:
                    if row["split"] != split or not row["method"].startswith(
                        f"width_{width}_seed_"
                    ):
                        continue
                    seed = row["method"].rsplit("_", 1)[1]
                    ref = indexed[
                        row["layout_id"],
                        row["query"],
                        f"width_{config['reference_width']}_seed_{seed}",
                    ]
                    values[row["layout_id"]].append(row["elapsed_seconds"] / ref["elapsed_seconds"])
                means = [sum(v) / len(v) for v in values.values()]
                saved = load(directory / "paired_widths.json")[split][str(width)][
                    "elapsed_seconds_ratio"
                ]["mean"]
                if abs(sum(means) / len(means) - saved) > 1e-12:
                    raise ValueError("Independent runtime mean differs")
        checked["independent_paired_runtime_means"] = True
        result[stage] = checked
    earlier = {r["group_id"] for p in config["sources"] for r in read_manifest(p)}
    validation = {r["group_id"] for r in read_manifest(root / "policy_validation")}
    test = {r["group_id"] for r in read_manifest(root / "fresh_test")}
    if earlier & validation or earlier & test or validation & test:
        raise ValueError("Layout groups overlap")
    result["disjoint_layout_groups"] = {
        "earlier": len(earlier),
        "validation": len(validation),
        "test": len(test),
    }
    result["frozen_sources_and_checkpoints"] = True
    return result


def branch_accuracy(result, split, width, seeds):
    errors = result["splits"][split]["prediction_errors"]
    if width is None:
        return errors["manhattan"]["branch_move_fraction"]["mean"]
    return float(
        np.mean([errors[f"width_{width}_seed_{s}"]["branch_move_fraction"]["mean"] for s in seeds])
    )


def interval(value):
    return f"{value['mean']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"


def expansion_evidence(value):
    if value["high"] < 1:
        return "fewer expansions"
    if value["low"] > 1:
        return "more expansions"
    return "no clear expansion difference"


def figures(result, config, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    widths = [None, *config["widths"]]
    colors = ["#5c6874", "#215487", "#009581", "#de7c24"]
    for i, (width, color) in enumerate(zip(widths, colors, strict=True)):
        name = "astar_manhattan" if width is None else f"width_{width}"
        label = "Manhattan" if width is None else f"CNN {width} channels"
        x = np.arange(2) + (i - 1.5) * 0.19
        for ax, field, title in (
            (axes[0, 0], "completion_ms", "Complete query time (ms)"),
            (axes[0, 1], "prediction_ms", "Fresh prediction time (ms)"),
        ):
            ax.bar(
                x,
                [result["splits"][s]["methods"][name][field]["mean"] for s in ("test", "walls")],
                0.18,
                color=color,
                label=label,
            )
            ax.set_title(title)
        axes[1, 0].bar(
            x,
            [
                100 * branch_accuracy(result, s, width, config["training"]["seeds"])
                for s in ("test", "walls")
            ],
            0.18,
            color=color,
        )
        ratios = [
            result["splits"][s]["methods"][name]["paired_95ci"]["reachable_expansion_ratio"]
            for s in ("test", "walls")
        ]
        axes[1, 1].errorbar(
            x,
            [r["mean"] for r in ratios],
            yerr=[[r["mean"] - r["low"] for r in ratios], [r["high"] - r["mean"] for r in ratios]],
            fmt="o",
            color=color,
            capsize=3,
        )
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].set(title="Correct moves at branches (%)", ylim=(0, 100))
    axes[1, 1].set_title("Paired expansions / Manhattan (95% intervals)")
    axes[1, 1].axhline(1, color="#666666", linestyle="--")
    for ax in axes.flat:
        ax.set_xticks([0, 1], ["Random 32×32", "Walls 32×32"])
    fig.suptitle("Smaller route models · fresh predictions · three seeds per width")
    save(fig, output, "width_comparison")


def report(result, paired, selection, config, training, verification, path):
    width = selection["width"]
    total_runs = sum(verification[s]["raw_runs"] for s in ("validation", "test"))
    selected_main = [result["splits"][s]["methods"][f"width_{width}"] for s in ("test", "walls")]
    reference_main = [
        result["splits"][s]["methods"][f"width_{config['reference_width']}"]
        for s in ("test", "walls")
    ]
    manhattan_main = [result["splits"][s]["methods"]["astar_manhattan"] for s in ("test", "walls")]
    lines = [
        "# Smaller-model results",
        "",
        "This experiment follows the [fixed width comparison](lightweight_models.md). "
        "It changes the CNN from 16 channels to 8 or 4 while keeping the training data, "
        "five convolution blocks, receptive field, loss, and search rule unchanged. "
        "The existing 16-channel models are remeasured on every new query.",
        "",
        f"Validation selected **{width} channels**. "
        + (
            "The smaller model passed the fixed speed, search-effort, and path-quality gates."
            if selection["smaller_model_selected"]
            else "Neither smaller model passed every fixed gate, so the reference was retained."
        ),
        "",
        "On random and wall 32×32 test maps, the selected model averaged "
        + " and ".join(f"{r['completion_ms']['mean']:.4f} ms" for r in selected_main)
        + " per query. The 16-channel control averaged "
        + " and ".join(f"{r['completion_ms']['mean']:.4f} ms" for r in reference_main)
        + "; Manhattan averaged "
        + " and ".join(f"{r['completion_ms']['mean']:.4f} ms" for r in manhattan_main)
        + ". These values include a fresh prediction for every query.",
        "",
        "The selected model's test intervals show "
        + expansion_evidence(selected_main[0]["paired_95ci"]["reachable_expansion_ratio"])
        + " on random maps and "
        + expansion_evidence(selected_main[1]["paired_95ci"]["reachable_expansion_ratio"])
        + " on walls, compared with Manhattan. The validation gate used mean ratios below "
        "one; it did not require a statistically clear reduction. Selection remains fixed "
        "even when a validation advantage does not carry over to test maps.",
        "",
        "## Training and validation",
        "",
        "Three seeds were used for each width. The six new training runs reused 1,200 "
        "training maps and 240 checkpoint-validation maps. Another 120 new maps supplied "
        "policy validation; only after selection were 300 fresh test maps generated. "
        "Every new layout is separate from all earlier layout groups, including rotations.",
        "",
        "| Width | Parameters | New training epochs (seeds 7, 17, 27) | Selected epochs |",
        "| --- | ---: | --- | --- |",
    ]
    for w in config["widths"]:
        rows = training[str(w)]
        epochs = (
            ", ".join(str(r["epochs"]) for r in rows)
            if w != config["reference_width"]
            else "Existing controls"
        )
        lines.append(
            f"| {w} | {rows[0]['parameters']:,} | {epochs} | "
            + ", ".join(str(r["best_epoch"]) for r in rows)
            + " |"
        )
    lines += [
        "",
        "| Candidate | Total-time ratio to 16 channels | "
        "Expansion ratio: random / walls | Eligible |",
        "| --- | ---: | --- | --- |",
    ]
    for row in selection["candidates"]:
        ratios = row["expansion_ratios_to_manhattan"]
        lines.append(
            f"| {row['width']} | {row['runtime_ratio_to_reference']:.3f} | "
            f"{ratios['random']:.3f} / {ratios['walls']:.3f} | "
            f"{'Yes' if row['eligible'] else 'No'} |"
        )
    lines += [
        "",
        "The selection score pairs the same query and training seed, averages within maps, "
        "and gives random and wall groups equal weight. Eligibility requires correct optimal "
        "paths, "
        "fewer expansions than Manhattan in both groups, and a total-time ratio below one. "
        "All three widths remain in the test tables; the test never changes the choice.",
        "",
        "## Fresh test",
        "",
        "Times include input preparation, a fresh prediction, search, and path reconstruction. "
        "Model loading is excluded. Each query uses the median of five shuffled repetitions "
        "on one CPU thread; CNN rows average three seeds. Times and raw expansions include "
        "unreachable queries. Branch accuracy measures whether the preferred legal neighbor "
        "belongs to a shortest route, rather than the search success rate.",
    ]
    for split in ("test", "walls"):
        lines += [
            "",
            f"### {SPLITS[split]}",
            "",
            "| Method | Total ms | Prediction ms | Search ms | Expanded cells | Branch accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for w in [None, *config["widths"]]:
            name = "astar_manhattan" if w is None else f"width_{w}"
            row = result["splits"][split]["methods"][name]
            label = "Manhattan" if w is None else f"CNN {w} channels"
            accuracy = branch_accuracy(result, split, w, config["training"]["seeds"])
            lines.append(
                f"| {label} | {row['completion_ms']['mean']:.4f} | "
                f"{row['prediction_ms']['mean']:.4f} | {row['search_ms']['mean']:.4f} | "
                f"{row['expanded']['mean']:.1f} | {100 * accuracy:.1f}% |"
            )
    lines += [
        "",
        "Component medians are computed separately and need not sum exactly to the total.",
        "",
        "## All held-out conditions",
        "",
        "Each row compares the same queries. Ratios average queries and seeds within maps. "
        "Brackets are 95% intervals from 2,000 whole-map bootstrap samples. Expansion ratios "
        "use reachable queries; runtime ratios include all queries. A ratio below one favors "
        "the row's model. Parameter count alone is not a latency measurement.",
        "",
        "| Condition | Width | Total time / 16 channels | Expansions / Manhattan | Optimal paths |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for split, label in SPLITS.items():
        for w in config["widths"]:
            row = result["splits"][split]["methods"][f"width_{w}"]
            lines.append(
                f"| {label} | {w} | {interval(paired[split][str(w)]['elapsed_seconds_ratio'])} | "
                f"{interval(row['paired_95ci']['reachable_expansion_ratio'])} | "
                f"{100 * row['optimal_fraction_of_found']:.1f}% |"
            )
    lines += ["", "## What changed", ""]
    for w in config["widths"]:
        if w == config["reference_width"]:
            continue
        main = [result["splits"][s]["methods"][f"width_{w}"] for s in ("test", "walls")]
        changes = [
            100 * (1 - row["completion_ms"]["mean"] / ref["completion_ms"]["mean"])
            for row, ref in zip(main, reference_main, strict=True)
        ]
        evidence = [
            expansion_evidence(row["paired_95ci"]["reachable_expansion_ratio"]) for row in main
        ]
        large = result["splits"]["size_64"]["methods"][f"width_{w}"]
        lines += [
            f"The {w}-channel model reduced mean total time by {changes[0]:.1f}% on random "
            f"32×32 maps and {changes[1]:.1f}% on walls, relative to the 16-channel control. "
            f"Its intervals show {evidence[0]} on random maps and {evidence[1]} on walls "
            "against Manhattan. On 64×64 maps, they show "
            + expansion_evidence(large["paired_95ci"]["reachable_expansion_ratio"])
            + ".",
            "",
        ]
    if all(
        result["splits"][s]["methods"][f"width_{w}"]["completion_ms"]["mean"]
        > result["splits"][s]["methods"]["astar_manhattan"]["completion_ms"]["mean"]
        for s in SPLITS
        for w in config["widths"]
    ):
        lines += [
            "Every CNN width remained slower than Manhattan on all five test groups. "
            "Smaller prediction cost improves the learned method without establishing "
            "an overall speed advantage over the classical baseline.",
            "",
        ]
    lines += [
        "## Scope and retained evidence",
        "",
        "This is a width-only experiment with the route objective and Manhattan tie ordering. "
        "It does not test fewer convolution blocks, quantization, different training targets, "
        "or repeated-query reuse. Absolute times should be compared within this round. "
        "The intervals describe maps for three fixed models per width; they do not cover "
        "all possible training runs or hardware. All timed methods returned optimal paths "
        "and correct reachability, and their counters agreed across repetitions.",
        "",
        f"Evidence verification checked {total_runs:,} "
        "timed runs, all repetitions, shared endpoints, checkpoint hashes, and separated layouts. "
        "Independent arithmetic reproduced the paired runtime means. The "
        "[archive](../results/research_v5/) retains raw repetitions, summaries, selection, "
        "and dataset manifests. The [new models](../artifacts/checkpoints/research_v5/) "
        "retain all six checkpoints and training histories.",
        "",
        "![Width comparison](../results/research_v5/width_comparison.png)",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/research_v5"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v5"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/checkpoints/research_v5"))
    parser.add_argument("--report", type=Path, default=Path("docs/lightweight_results.md"))
    args = parser.parse_args()
    config = tomllib.loads((args.run / "protocol.toml").read_text())
    checked = verify(args.run, config)
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=False)
    training = {}
    for w in config["widths"]:
        source = (
            Path(config["reference_models"])
            if w == config["reference_width"]
            else args.run / f"width_{w}"
        )
        training[str(w)] = load(source / "summary.json")
        if w != config["reference_width"]:
            shutil.copytree(source, args.models / f"width_{w}")
            for seed in config["training"]["seeds"]:
                if digest(source / f"seed_{seed}.pt") != digest(
                    args.models / f"width_{w}/seed_{seed}.pt"
                ):
                    raise ValueError("Archived checkpoint differs")
    for stage in ("validation", "test"):
        shutil.copytree(args.run / stage, args.output / stage)
    for name in (
        "protocol.toml",
        "fingerprints.json",
        "selection.json",
        "policy_validation_verified.json",
        "fresh_test_verified.json",
    ):
        shutil.copy2(args.run / name, args.output / name)
    for name in ("policy_validation", "fresh_test"):
        shutil.copy2(args.run / name / "metadata.json", args.output / f"{name}_metadata.json")
        with gzip.open(args.output / f"{name}_manifest.jsonl.gz", "wb") as stream:
            stream.write((args.run / name / "manifest.jsonl").read_bytes())
    save_json(args.output / "verification.json", checked)
    save_json(args.output / "training_summary.json", training)
    result = load(args.output / "test/summary.json")
    figures(result, config, args.output)
    report(
        result,
        load(args.output / "test/paired_widths.json"),
        load(args.output / "selection.json"),
        config,
        training,
        checked,
        args.report,
    )
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
