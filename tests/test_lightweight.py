import copy
import gzip
import json

import pytest
import torch

from learned_astar.models import DistanceCNN
from learned_astar.models.network import RESEARCH_ARCHITECTURE
from learned_astar.research.experiment import digest
from learned_astar.research.lightweight import choose_width, make_spec, paired_widths


def test_width_selection_requires_speed_search_gain_and_correct_validation_paths():
    group = {
        "queries": 10,
        "reachable_success_rate": 1,
        "optimal_fraction_of_found": 1,
        "unreachable_correct_rate": 1,
        "statuses": {"found": 8, "no_path": 2},
        "paired_95ci": {"reachable_expansion_ratio": {"mean": 0.95}},
    }
    summary = {
        "settings": {"validation": True},
        "splits": {
            s: {"methods": {f"width_{w}": copy.deepcopy(group) for w in (4, 8)}}
            for s in ("random", "walls")
        },
    }
    pairs = {
        s: {str(w): {"elapsed_seconds_ratio": {"mean": score}} for w, score in ((4, 0.5), (8, 0.7))}
        for s in ("random", "walls")
    }
    config = {"widths": [16, 8, 4], "reference_width": 16}
    assert choose_width(summary, pairs, config)["width"] == 4
    summary["splits"]["walls"]["methods"]["width_4"]["paired_95ci"]["reachable_expansion_ratio"][
        "mean"
    ] = 1.01
    assert choose_width(summary, pairs, config)["width"] == 8
    summary["splits"]["random"]["methods"]["width_8"]["optimal_fraction_of_found"] = 0.99
    assert choose_width(summary, pairs, config)["width"] == 16
    summary["settings"]["validation"] = False
    with pytest.raises(ValueError, match="not test"):
        choose_width(summary, pairs, config)


def test_width_comparison_pairs_seed_and_query_then_weights_maps_equally(tmp_path):
    rows = []
    for layout, queries, ratio in (("a", 1, 0.5), ("b", 3, 0.9)):
        for query in range(queries):
            for seed in (7, 17, 27):
                for width in (16, 8):
                    duration = (seed + 1) * (1 if width == 16 else ratio)
                    rows.append(
                        {
                            "layout_id": layout,
                            "query": query,
                            "split": "random",
                            "method": f"width_{width}_seed_{seed}",
                            "completed": True,
                            "elapsed_seconds": duration,
                            "prediction_seconds": duration * 0.8,
                            "search_seconds": duration * 0.2,
                            "optimal_cost": 5,
                            "expanded": 10,
                        }
                    )
    with gzip.open(tmp_path / "queries.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    config = {"reference_width": 16, "evaluation": {"bootstrap_seed": 7, "bootstrap_samples": 100}}
    result = paired_widths(tmp_path, config)["random"]
    assert result["8"]["elapsed_seconds_ratio"]["mean"] == pytest.approx(0.7)
    assert result["8"]["elapsed_seconds_ratio"]["maps"] == 2
    assert result["16"]["elapsed_seconds_ratio"]["mean"] == 1


def test_width_spec_preserves_training_controls_and_rejects_wrong_checkpoint(tmp_path):
    development = tmp_path / "data"
    development.mkdir()
    (development / "manifest.jsonl").write_text("{}\n")
    training = {"seeds": [7], "route_loss_weight": 0.02}
    config = {
        "development": str(development),
        "reference_models": str(tmp_path / "width_16"),
        "widths": [16, 8, 4],
        "reference_width": 16,
        "training": training,
    }
    for width, count in ((16, 10033), (8, 2713), (4, 781)):
        model = DistanceCNN(channels=width)
        assert sum(p.numel() for p in model.parameters()) == count
        folder = tmp_path / f"width_{width}"
        folder.mkdir()
        torch.save(
            {
                "architecture": RESEARCH_ARCHITECTURE,
                "model_config": model.model_config,
                "state_dict": model.state_dict(),
                "seed": 7,
                "dataset_manifest_sha256": digest(development / "manifest.jsonl"),
                "training_config": {**training, "seeds": (7,)},
            },
            folder / "seed_7.pt",
        )
    spec = make_spec(tmp_path, config)
    assert len(spec["search_plan"]) == 4
    assert all(
        item["mode"] == "tie"
        for name, item in spec["search_plan"].items()
        if name != "astar_manhattan"
    )
    path = tmp_path / "width_4/seed_7.pt"
    bad = torch.load(path, weights_only=True)
    bad["training_config"]["route_loss_weight"] = 0.0
    torch.save(bad, path)
    with pytest.raises(ValueError, match="training settings"):
        make_spec(tmp_path, config)
