import json
import subprocess
import sys
from pathlib import Path

import pytest

from learned_astar.terminal import parse_map, render_map


@pytest.mark.parametrize("text", ["", "S.G\n..", "S..", "SSG", "S?G", "S G"])
def test_invalid_ascii_map(text):
    with pytest.raises(ValueError):
        parse_map(text)


def test_map_roundtrip_and_path_overlay():
    text = "S.#\n..G"
    grid, start, goal = parse_map(text)
    assert render_map(grid, start, goal) == text
    assert render_map(grid, start, goal, [(0, 0), (1, 0), (1, 1), (1, 2)]) == "S.#\n**G"


def test_example_maps_and_solve_cli(tmp_path):
    for path in Path("examples/maps").glob("*.txt"):
        parse_map(path.read_text())
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "learned_astar",
            "solve",
            "--map",
            "examples/maps/unreachable.txt",
            "--checkpoints",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "no_path" in result.stdout
    saved = json.loads(output.read_text())
    assert saved["optimal_cost"] is None
    assert all(row["status"] == "no_path" for row in saved["results"].values())


def test_shell_can_draw_run_show_path_and_recover_from_invalid_input():
    result = subprocess.run(
        [sys.executable, "-m", "learned_astar", "shell", "--checkpoints"],
        input="draw\nS?G\n.end\ndraw\nS.G\n.end\nrun\npath astar_manhattan\nquit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Error:" in result.stdout
    assert "astar_manhattan" in result.stdout
    assert "S*G" in result.stdout
