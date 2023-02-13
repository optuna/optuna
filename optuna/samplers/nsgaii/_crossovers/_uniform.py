import numpy as np

from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.study import Study


class UniformCrossover(BaseCrossover):
    """Uniform Crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Select each parameter with equal probability from the two parent individuals.
    For further information about uniform crossover, please refer to the following paper:

    - `Gilbert Syswerda. 1989. Uniform Crossover in Genetic Algorithms.
      In Proceedings of the 3rd International Conference on Genetic Algorithms.
      Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2-9.
      <https://www.researchgate.net/publication/201976488_Uniform_Crossover_in_Genetic_Algorithms>`_

    Args:
        swapping_prob:
            Probability of swapping each parameter of the parents during crossover.
    """

    n_parents = 2

    def __init__(self, swapping_prob: float = 0.5) -> None:
        self._swapping_prob = swapping_prob

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://www.researchgate.net/publication/201976488_Uniform_Crossover_in_Genetic_Algorithms
        # Section 1 Introduction

        n_params = len(search_space_bounds)
        masks = (rng.rand(n_params) >= self._swapping_prob).astype(int)
        child_params = parents_params[masks, range(n_params)]

        return child_params
