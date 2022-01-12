from typing import Dict

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.samplers._nsga2._crossovers._base import BaseCrossover
from optuna.study import Study


class BLXAlphaCrossover(BaseCrossover):
    """A crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Uniformly samples child individuals from the hyper-rectangles created
    by the two parent individuals. For further information about BLX-alpha crossover,
    please refer to the following paper:

    - `Eshelman, L. and J. D. Schaffer.
      Real-Coded Genetic Algorithms and Interval-Schemata. FOGA (1992).
      <https://www.sciencedirect.com/science/article/abs/pii/B9780080948324500180>`_
    """

    n_parents = 2

    def __init__(self, alpha: float = 0.5) -> None:

        self._alpha = alpha

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> np.ndarray:

        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.465.6900&rep=rep1&type=pdf
        # Section 2 Crossover Operators for RCGA 2.1 Blend Crossover

        parents_min = parents_params.min(axis=0)
        parents_max = parents_params.max(axis=1)
        diff = self._alpha * (parents_max - parents_min)  # Equation (1).
        low = parents_min - diff  # Equation (1).
        high = parents_max + diff  # Equation (1).
        r = rng.rand(len(search_space))
        child_params = (high - low) * r + low

        return child_params
