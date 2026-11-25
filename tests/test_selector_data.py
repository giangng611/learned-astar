import gzip
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from learned_astar.data.dataset import layout_id, occupancy
from learned_astar.environments import Grid, random_grid


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("prepare_selector_data")
    archive = tmp_path / "archive"
    archive.mkdir()
    first_seed = random.Random(119501).randrange(2**32)
    first_grid = random_grid(8, 8, 0.1, random.Random(first_seed))
    excluded = layout_id(np.rot90(occupancy(first_grid)), rotations=True)
    (archive / "separation.json").write_text(json.dumps({"excluded_groups": [excluded]}))
    (archive / "policy_manifest.json").write_text("[]")
    config = tmp_path / "pilot.toml"
    config.write_text(
        "seed = 119501\nquery_seed = 2\norder_seed = 3\nsizes = [8]\n"
        "train_per_family = 2\nvalidation_per_family = 2\nqueries_per_session = 4\n"
        "goal_counts = [1, 4]\nrepetitions = 3\ncpu_threads = 1\n"
        f"exclusion_archive = {json.dumps(str(archive))}\n"
    )
    root = tmp_path / "run"
    module.prepare(root, config)
    return module, root, config, excluded


def test_split_separation_rotated_exclusions_and_reproducible_queries(pilot, tmp_path):
    module, root, config, excluded = pilot
    records = json.loads((root / "manifest.json").read_text())
    assert len(records) == 8
    groups = {r["group_id"] for r in records}
    assert len(groups) == len(records) and excluded not in groups
    assert {r["split"] for r in records} == {"train", "validation"}
    second = tmp_path / "repeat"
    module.prepare(second, config)
    assert (root / "workloads.json").read_bytes() == (second / "workloads.json").read_bytes()
    for w in json.loads((root / "workloads.json").read_text()):
        starts = {tuple(q["start"]) for q in w["queries"]}
        goals = {tuple(q["goal"]) for q in w["queries"]}
        assert len(starts) == 4 and len(goals) == w["goal_count"] and starts.isdisjoint(goals)
    with pytest.raises(FileExistsError):
        module.prepare(root, config)
    (root / "workloads.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen inputs"):
        module.verify_inputs(root)


def test_end_to_end_medians_resume_and_corrupt_evidence(pilot, monkeypatch):
    module, root, _, _ = pilot
    # Actual pathfinding and timing on eight tiny maps; no synthetic runtime labels.
    module.collect(root)
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "dataset.json").read_text())
    assert summary["maps"] == 8 and summary["rows"] == 16
    assert summary["sessions"] == 144 and summary["timed_queries"] == 576
    assert summary["split_maps"] == {"train": 4, "validation": 4}
    for row in rows:
        for method, samples in row["samples_seconds"].items():
            assert row["median_seconds"][method] == sorted(samples)[1]
        assert row["winner"] == min(row["median_seconds"], key=row["median_seconds"].get)
        assert not {"family", "optimal_cost", "winner", "split"} & row["features"].keys()
    before = (root / "dataset.json").read_bytes()
    # Only warm-ups may execute on resume; all completed maps must be retained.
    original = module.measure
    calls = []

    def tracked(name, grid, pairs):
        calls.append(len(pairs))
        return original(name, grid, pairs)

    monkeypatch.setattr(module, "measure", tracked)
    module.collect(root)
    assert calls == [1] * 9
    assert (root / "dataset.json").read_bytes() == before
    path = next((root / "measurements").glob("*.gz"))
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    data["sessions"][0]["result_sha256"] = "wrong"
    with gzip.open(path, "wt") as handle:
        json.dump(data, handle)
    with pytest.raises(ValueError, match="Changed measurement evidence"):
        module.summarize(root)


def test_features_and_invalid_path(pilot):
    module, _, _, _ = pilot
    grid = Grid.from_ascii(["....", "###.", "....", "...."])
    values = module.features(grid, [((0, 0), (2, 0)), ((3, 0), (2, 0))])
    assert values["obstacle_fraction"] == 3 / 16
    assert values["max_line_obstacles"] == 3 / 4
    assert values["queries"] == 2 and values["unique_goals"] == 1
    workload = {"queries": [{"start": (0, 0), "goal": (2, 0), "optimal_cost": 8}]}
    with pytest.raises(ValueError):
        module.check_results(
            grid, workload, [{"status": "found", "cost": 2, "path": [(0, 0), (1, 0), (2, 0)]}]
        )


def test_invalid_config_rejected_before_creation(pilot, tmp_path):
    module, _, config, _ = pilot
    config.write_text(config.read_text().replace("goal_counts = [1, 4]", "goal_counts = [1, 3]"))
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="goal counts"):
        module.prepare(root, config)
    assert not root.exists()


def test_training_checkpoint_scaler_and_validation_only_outputs(pilot, tmp_path):
    import torch

    module, root, _, _ = pilot
    training = importlib.import_module("train_selector")
    module.collect(root)
    config = tmp_path / "training.toml"
    config.write_text(
        "seed = 7\nhidden_channels = 8\nepochs = 3\npatience = 2\n"
        "learning_rate = 0.01\ncpu_threads = 1\ninference_repetitions = 3\n"
    )
    output = tmp_path / "model"
    training.train(root, output, config)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["parameters"] == 99
    assert summary["train_maps"] == summary["validation_maps"] == 4
    assert summary["epochs_run"] <= 3
    assert summary["stage"] == "validation_pilot_only"
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    rows = json.loads((root / "dataset.json").read_text())
    train_values = np.asarray(
        [[r["features"][k] for k in training.FEATURES] for r in rows if r["split"] == "train"],
        dtype=np.float32,
    )
    expected_mean, expected_scale = training.fit_scaler(train_values)
    np.testing.assert_allclose(checkpoint["mean"], expected_mean)
    np.testing.assert_allclose(checkpoint["scale"], expected_scale)
    predictions = json.loads((output / "validation_predictions.json").read_text())
    val_groups = {r["group_id"] for r in rows if r["split"] == "validation"}
    assert {r["group_id"] for r in predictions} == val_groups
    model = training.RuntimeSelector(checkpoint["hidden_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    for predicted in predictions:
        row = next(
            r
            for r in rows
            if r["layout_id"] == predicted["layout_id"]
            and r["goal_count"] == predicted["goal_count"]
        )
        with torch.inference_mode():
            assert (
                training.predict_one(model, row["features"], expected_mean, expected_scale)
                == predicted["chosen_method"]
            )
        chosen_cost = row["median_seconds"][predicted["chosen_method"]]
        assert predicted["costs_seconds"]["learned_selector"] == pytest.approx(
            chosen_cost + row["feature_seconds"] + predicted["inference_seconds"]
        )
    with pytest.raises(FileExistsError):
        training.train(root, output, config)


def test_hand_rule_and_scaler_constant_features(pilot):
    training = importlib.import_module("train_selector")
    assert training.hand_rule({"unique_goals": 8, "max_line_obstacles": 0.95}) == "bfs_policy"
    assert (
        training.hand_rule({"unique_goals": 128, "max_line_obstacles": 0.95}) == "landmark_demand"
    )
    assert (
        training.hand_rule({"unique_goals": 128, "max_line_obstacles": 0.25}) == "manhattan_demand"
    )
    mean, scale = training.fit_scaler(np.asarray([[1, 0], [1, 4]], dtype=np.float32))
    np.testing.assert_equal(mean, [1, 2])
    np.testing.assert_equal(scale, [1, 2])
