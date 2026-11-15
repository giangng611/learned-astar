import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from learned_astar.data import DatasetConfig, generate_dataset, read_manifest
from learned_astar.environments import Grid
from learned_astar.evaluation.run import read_checkpoints
from learned_astar.evaluation.summarize import bootstrap_mean, summarize
from learned_astar.models import DistanceCNN
from learned_astar.models.network import ARCHITECTURE
from learned_astar.search import SearchLimits, SearchStatus, bfs
from learned_astar.search.learned import learned_astar


def test_learned_search_uses_one_fresh_prediction_and_counts_inference_time(monkeypatch):
    import learned_astar.search.learned as module

    calls = []
    clock = iter([10.0, 10.2])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        module, "predict_distances", lambda *args: calls.append(1) or np.zeros((3, 3))
    )
    grid = Grid(3, 3)
    result = learned_astar(grid, (0, 0), (2, 2), None)
    assert result.cost == bfs(grid, (0, 0), (2, 2)).cost
    assert calls == [1]
    assert result.elapsed_seconds == pytest.approx(0.2)


def test_inference_consumes_search_time_budget(monkeypatch):
    import learned_astar.search.learned as module

    clock = iter([10.0, 10.2, 10.21])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(module, "predict_distances", lambda *args: np.zeros((2, 2)))
    result = learned_astar(
        Grid(2, 2), (0, 0), (1, 1), None, limits=SearchLimits(time_limit_seconds=0.1)
    )
    assert result.status == SearchStatus.TIME_LIMIT
    assert result.stats.expanded == 0
    assert result.elapsed_seconds == pytest.approx(0.21)


def test_complete_evaluation_and_summary_rejects_leakage_and_tampering(tmp_path):
    data = tmp_path / "data"
    generate_dataset(
        data,
        DatasetConfig(
            train_maps=2,
            validation_maps=2,
            test_maps=2,
            extra_maps=0,
            size=8,
            goals_per_map=1,
            starts_per_map=2,
        ),
    )
    records = read_manifest(data)
    manifest_hash = hashlib.sha256((data / "manifest.jsonl").read_bytes()).hexdigest()
    checkpoint = tmp_path / "model.pt"
    torch.manual_seed(1)
    payload = {
        "architecture": ARCHITECTURE,
        "state_dict": DistanceCNN().state_dict(),
        "seed": 1,
        "epoch": 1,
        "validation_loss": 0.1,
        "dataset_manifest_sha256": manifest_hash,
        "development_layout_groups": [r["group_id"] for r in records if r["split"] != "test"],
    }
    torch.save(payload, checkpoint)
    output = tmp_path / "run"
    config = tmp_path / "evaluation.toml"
    config.write_text("repetitions = 2\nbootstrap_samples = 50\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "learned_astar.evaluation.run",
            "--data",
            str(data),
            "--checkpoints",
            str(checkpoint),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = summarize(output, tmp_path / "summary")
    methods = result["splits"]["test"]["methods"]
    assert methods["astar_manhattan"]["queries"] == 4
    assert methods["astar_manhattan"]["optimal_fraction_of_found"] == 1
    assert result["splits"]["test"]["paired_95ci"]["learned_mean"]["runtime_ratio"]["maps"] == 2
    raw = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert len(raw) == 4 * 7 * 2
    payload["development_layout_groups"].append(
        next(r["group_id"] for r in records if r["split"] == "test")
    )
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="overlap"):
        read_checkpoints([checkpoint], [r for r in records if r["split"] == "test"], manifest_hash)
    with (output / "runs.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        summarize(output, tmp_path / "bad")


def test_bootstrap_resamples_map_values():
    result = bootstrap_mean([2.0, 2.0, 2.0], np.random.default_rng(1), 100)
    assert result == {"mean": 2.0, "low": 2.0, "high": 2.0, "maps": 3}
