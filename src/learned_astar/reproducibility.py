"""Keep map generation and small CPU model checks repeatable."""

import random

from learned_astar.config import require_int


def make_rng(seed: int) -> random.Random:
    require_int(seed, "seed")
    return random.Random(seed)


def configure_torch(seed: int, cpu_threads: int = 1) -> None:
    """Configure CPU work; repeated calls with the same thread count are safe."""
    import numpy as np
    import torch

    require_int(seed, "seed")
    if seed >= 2**32:
        raise ValueError("seed must be less than 2**32")
    require_int(cpu_threads, "cpu_threads", minimum=1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != cpu_threads:
        torch.set_num_interop_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
