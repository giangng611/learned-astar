import json
import subprocess
import sys

import pytest

from learned_astar.cli import main


def test_demo_exports_a_replayable_map_and_valid_results(tmp_path, capsys):
    output = tmp_path / "results" / "demo.json"
    assert main(["demo", "--output", str(output), "--show-map"]) == 0
    report = json.loads(output.read_text())
    assert report["config"]["seed"] == 4550
    assert len(report["grid"]["blocked"]) == 256
    assert len(report["results"]) == 6
    assert report["results"]["bfs"]["status"] == "found"
    optimal = report["optimal_cost"]
    for name in ("bfs", "dijkstra", "astar_manhattan"):
        result = report["results"][name]
        assert result["cost"] == optimal
        assert len(result["path"]) - 1 == optimal
        assert result["path"][0] == report["query"]["start"]
        assert result["path"][-1] == report["query"]["goal"]
    assert "S: start, G: goal" in capsys.readouterr().out


def test_demo_retains_unreachable_queries(tmp_path):
    config, output = tmp_path / "map.toml", tmp_path / "demo.json"
    config.write_text("height = 1\nwidth = 3\nobstacle_fraction = 0.333333\nseed = 0\n")
    main(["demo", "--config", str(config), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["optimal_cost"] is None
    assert report["grid"]["blocked"] == [[0, 1]]
    for result in report["results"].values():
        assert result["status"] == "no_path"
        assert result["path_cost_ratio"] is None


def test_demo_reports_limits_and_seed_override(tmp_path):
    output = tmp_path / "limited.json"
    main(["demo", "--seed", "10", "--max-expansions", "0", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["config"]["seed"] == 10
    for result in report["results"].values():
        assert result["status"] == "expansion_limit"
        assert result["path_cost_ratio"] is None


@pytest.mark.parametrize(
    "contents",
    [
        "height = 1\nwidth = 1\nobstacle_fraction = 0\n",
        "height = -1\n",
        "obstacle_fracton = 0.25\n",
        "height = [\n",
    ],
)
def test_demo_explains_invalid_settings(tmp_path, capsys, contents):
    config = tmp_path / "invalid.toml"
    config.write_text(contents)
    with pytest.raises(SystemExit) as error:
        main(["demo", "--config", str(config)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_module_entry_point_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "learned_astar", "demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "astar_manhattan" in result.stdout
