import numpy as np
import pytest
import torch

from learned_astar.data import DatasetConfig, generate_dataset, read_manifest, verify_dataset
from learned_astar.models import DistanceCNN, encode_inputs, load_model
from learned_astar.models.network import RESEARCH_ARCHITECTURE


def test_residual_preserves_geometry_and_obstacle_ablation_is_exact(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    blocked = np.zeros((8, 8), bool)
    blocked[3, :6] = True
    inputs = torch.from_numpy(encode_inputs(blocked, (7, 7))).unsqueeze(0)
    hidden = inputs.clone()
    hidden[:, 0] = 0
    model = DistanceCNN(channels=16, residual=True, obstacles=False).eval()
    with torch.inference_mode():
        prediction = model(inputs)
        assert torch.all(prediction >= inputs[:, 4])
        torch.testing.assert_close(prediction, model(hidden), rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": RESEARCH_ARCHITECTURE,
            "model_config": model.model_config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = load_model(path)
    torch.testing.assert_close(restored(inputs), prediction)


def test_mixed_dataset_excludes_previous_layouts_and_preserves_validation_split(tmp_path):
    original = tmp_path / "original"
    config = DatasetConfig(
        train_maps=2,
        validation_maps=2,
        test_maps=0,
        extra_maps=0,
        train_wall_maps=2,
        validation_wall_maps=2,
        size=16,
        goals_per_map=2,
    )
    generate_dataset(original, config)
    excluded = {r["group_id"] for r in read_manifest(original)}
    fresh = tmp_path / "fresh"
    metadata = generate_dataset(fresh, config, exclude_groups=excluded)
    records = read_manifest(fresh)
    assert not excluded & {r["group_id"] for r in records}
    assert metadata["excluded_layout_groups"] == 8
    assert verify_dataset(fresh)["verified_goal_maps"] == 16
    assert {r["kind"] for r in records if r["split"] == "validation"} == {"random", "walls"}


def test_invalid_model_width_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        DistanceCNN(channels=0)
