"""
Optuna example that optimizes a simple function with Hydra.

In this example, we optimize a simple function with hydra's sweeper.

You can run this example code as follows:

    python simple.py --multirun

The optimization results in `multirun/` directory,
for example `multirun/2021-01-31/19-26-39/optimization_results.yaml`.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def objective(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    z: int = cfg.z
    return x ** 2 + y ** 2 + z


if __name__ == "__main__":
    objective()
