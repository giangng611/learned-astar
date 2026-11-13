import random

import pytest

from learned_astar.config import ExperimentConfig, load_config
from learned_astar.reproducibility import make_rng


def test_config_loads_and_rejects_misspelled_settings(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("height = 16\nweights = [1.25, 2.0]\n")
    assert load_config(config) == ExperimentConfig(height=16, weights=(1.25, 2.0))
    config.write_text("hieght = 16\n")
    with pytest.raises(ValueError, match="Unknown settings"):
        load_config(config)


@pytest.mark.parametrize(
    "settings",
    [
        {"height": 0},
        {"width": True},
        {"seed": -1},
        {"seed": 2**32},
        {"cpu_threads": 0},
        {"obstacle_fraction": 1.1},
        {"obstacle_fraction": float("nan")},
        {"weights": []},
        {"weights": [0.5]},
        {"weights": [float("inf")]},
        {"weights": [True]},
        {"weights": [1.5, 1.5]},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_map_rng_is_repeatable_without_changing_global_random_state():
    before = random.getstate()
    first, second = make_rng(4550), make_rng(4550)
    assert [first.random() for _ in range(10)] == [second.random() for _ in range(10)]
    assert random.getstate() == before
