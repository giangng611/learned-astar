import copy
import importlib.util
from pathlib import Path

import numpy as np

from learned_astar.data import reverse_bfs
from learned_astar.environments import Grid
from learned_astar.search.prepared import PreparedAStar, PreparedGrid

SPEC = importlib.util.spec_from_file_location(
    "order_limit", Path(__file__).parents[1] / "scripts/order_limit.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_expansion_floor_is_attained_on_every_three_by_three_grid_and_endpoint_pair():
    cells = tuple((r, c) for r in range(3) for c in range(3))
    for mask in range(512):
        grid = Grid(3, 3, frozenset(c for i, c in enumerate(cells) if mask & (1 << i)))
        prepared = PreparedGrid.build(grid)
        for goal in grid.free_cells():
            dg = reverse_bfs(grid, goal)
            baseline = PreparedAStar(prepared, goal)
            for start in grid.free_cells():
                floor, mandatory, scores = module.expansion_floor(grid, start, goal, dg)
                result = PreparedAStar(prepared, goal, ties=scores.ravel()).search(start)
                ordinary = baseline.search(start)
                assert result.cost == ordinary.cost
                assert result.stats.expanded == floor <= ordinary.stats.expanded
                assert result.stats.reexpanded == 0
                assert 0 <= mandatory <= floor
                rows, columns = np.indices(dg.shape)
                h = abs(rows - goal[0]) + abs(columns - goal[1])
                detour = np.maximum(dg - h, 0)
                exact_detour = PreparedAStar(prepared, goal, ties=detour.ravel()).search(start)
                assert exact_detour.stats.expanded == floor


def test_expansion_floor_counts_forced_detours_and_unreachable_components():
    grid = Grid.from_ascii([".....", "####.", ".....", ".####", "....."])
    floor, mandatory, scores = module.expansion_floor(grid, (0, 0), (4, 4))
    result = PreparedAStar(PreparedGrid.build(grid), (4, 4), ties=scores.ravel()).search((0, 0))
    assert result.cost == 16
    assert result.stats.expanded == floor >= 16
    assert mandatory > 0
    isolated = Grid.from_ascii(["...", "###", "..."])
    floor, mandatory, _ = module.expansion_floor(isolated, (0, 0), (2, 2))
    assert floor == mandatory == 3


def test_learned_detour_preserves_score_order_and_distance_adapter_matches_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import torch
    from run_order_pilot import measure_learned, shifted_detour

    from learned_astar.models import DistanceCNN
    from learned_astar.research.runtime import measure_session, result_signature
    from learned_astar.search import SearchLimits

    torch.set_num_threads(1)
    model = DistanceCNN(channels=4).eval()
    grid = Grid.from_ascii([".....", "####.", ".....", ".####", "....."])
    starts, goal, limits = [(0, 0), (2, 0)], (4, 4), SearchLimits()
    _, actual = measure_learned(model, "distance", grid, goal, starts, limits)
    _, expected = measure_session("prepared_7", grid, goal, starts, {7: model}, limits)
    assert result_signature(actual) == result_signature(expected)
    field = np.array([[1.2, 1.8], [2.7, 0.0]])
    transformed = shifted_detour(field, (1, 1))
    raw = field - np.array([[2, 1], [1, 0]])
    assert transformed.min() == 0
    assert np.array_equal(np.argsort(transformed.ravel()), np.argsort(raw.ravel()))


def test_pilot_gate_requires_manhattan_and_matched_control_gains(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from report_order_pilot import apply_gate
    from run_order_pilot import FAMILIES

    estimate = {"mean": 0.97, "low": 0.94, "high": 0.99}
    config = {
        "primary_batch_size": 128,
        "gate": {"maximum_mean_time_ratio": 0.98, "maximum_group_regression": 1.03},
    }
    summary = {
        "measurement": {"all_correct": True},
        "overall": {},
        "families": {f: {"128": {}} for f in FAMILIES},
    }
    for rule in ("distance", "detour"):
        name = f"candidate_{rule}"
        summary["overall"][name] = {
            "ratio_to_frozen_distance": dict(estimate),
            f"ratio_to_control_{rule}": dict(estimate),
        }
        for family in FAMILIES:
            summary["families"][family]["128"][name] = {
                key: dict(estimate)
                for key in (
                    "ratio_to_frozen_distance",
                    "ratio_to_prepared_manhattan",
                    "expansion_ratio_to_frozen",
                )
            }
    assert apply_gate(summary, config)["expand_to_more_seeds"]
    bad = copy.deepcopy(summary)
    for rule in ("distance", "detour"):
        bad["overall"][f"candidate_{rule}"][f"ratio_to_control_{rule}"]["high"] = 1.01
    assert not apply_gate(bad, config)["expand_to_more_seeds"]
    bad = copy.deepcopy(summary)
    for rule in ("distance", "detour"):
        bad["families"]["walls_32"]["128"][f"candidate_{rule}"]["ratio_to_prepared_manhattan"][
            "high"
        ] = 1.01
    assert not apply_gate(bad, config)["expand_to_more_seeds"]
