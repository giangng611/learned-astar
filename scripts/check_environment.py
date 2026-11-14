"""Check that a small convolution can run and backpropagate on the CPU."""

import json
import platform

import numpy as np
import torch

from learned_astar.config import ExperimentConfig
from learned_astar.reproducibility import configure_torch


def main() -> None:
    config = ExperimentConfig()
    configure_torch(config.seed, config.cpu_threads)
    layer = torch.nn.Conv2d(5, 8, kernel_size=3, padding=1)
    inputs = torch.randn(2, 5, config.height, config.width)
    output = layer(inputs)
    loss = output.square().mean()
    loss.backward()
    if not torch.isfinite(output).all() or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("The CPU model check produced nonfinite values")
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(output.device),
                "cpu_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "output_shape": list(output.shape),
                "forward_and_backward": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
