from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class VSBXCrossover(BaseCrossover):
    """Modified Simulated Binary Crossover operation used by
    :class:`~optuna.samplers.NSGAIISampler`.

    vSBX generates child individuals without excluding any region of the parameter space,
    while maintaining the excellent properties of SBX.

    In the paper, vSBX has only one argument, ``eta``,
    and generate two child individuals.
    However, Optuna can only return one child individual in one crossover operation,
    so it uses the ``uniform_crossover_prob`` and ``use_child_gene_prob`` arguments
    to make two individuals into one.

    - `Pedro J. Ballester, Jonathan N. Carter.
      Real-Parameter Genetic Algorithms for Finding Multiple Optimal Solutions
      in Multi-modal Optimization. GECCO 2003: 706-717
      <https://doi.org/10.1007/3-540-45105-6_86>`__

    Args:
        eta:
            Distribution index. A small value of ``eta`` allows distant solutions
            to be selected as children solutions. If not specified, takes default
            value of ``2`` for single objective functions and ``20`` for multi objective.
        uniform_crossover_prob:
            ``uniform_crossover_prob`` is the probability of uniform crossover
            between two individuals selected as candidate child individuals.
            This argument is whether or not two individuals are
            crossover to make one child individual.
            If the ``uniform_crossover_prob`` exceeds 0.5,
            the result is equivalent to ``1-uniform_crossover_prob``,
            because it returns one of the two individuals of the crossover result.
            If not specified, takes default value of ``0.5``.
            The range of values is ``[0.0, 1.0]``.
        use_child_gene_prob:
            ``use_child_gene_prob`` is the probability of using the value of the generated
            child variable rather than the value of the parent.
            This probability is applied to each variable individually.
            where ``1-use_chile_gene_prob`` is the probability of
            using the parent's values as it is.
            If not specified, takes default value of ``0.5``.
            The range of values is ``(0.0, 1.0]``.
    """

    n_parents = 2

    def __init__(
        self,
        eta: float | None = None,
        uniform_crossover_prob: float = 0.5,
        use_child_gene_prob: float = 0.5,
    ) -> None:
        if (eta is not None) and (eta < 0.0):
            raise ValueError("The value of `eta` must be greater than or equal to 0.0.")
        self._eta = eta

        if uniform_crossover_prob < 0.0 or uniform_crossover_prob > 1.0:
            raise ValueError(
                "The value of `uniform_crossover_prob` must be in the range [0.0, 1.0]."
            )
        if use_child_gene_prob <= 0.0 or use_child_gene_prob > 1.0:
            raise ValueError("The value of `use_child_gene_prob` must be in the range (0.0, 1.0].")
        self._uniform_crossover_prob = uniform_crossover_prob
        self._use_child_gene_prob = use_child_gene_prob

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://doi.org/10.1007/3-540-45105-6_86
        # Section 3.2 Crossover Schemes (vSBX)
        if self._eta is None:
            eta = 20.0 if study._is_multi_objective() else 2.0
        else:
            eta = self._eta

        eps = 1e-10
        us = rng.rand(len(search_space_bounds))
        beta_1 = np.power(1 / np.maximum((2 * us), eps), 1 / (eta + 1))
        beta_2 = np.power(1 / np.maximum((2 * (1 - us)), eps), 1 / (eta + 1))

        u_1 = rng.rand()
        if u_1 <= 0.5:
            c1 = 0.5 * ((1 + beta_1) * parents_params[0] + (1 - beta_2) * parents_params[1])
        else:
            c1 = 0.5 * ((1 - beta_1) * parents_params[0] + (1 + beta_2) * parents_params[1])
        u_2 = rng.rand()
        if u_2 <= 0.5:
            c2 = 0.5 * ((3 - beta_1) * parents_params[0] - (1 - beta_2) * parents_params[1])
        else:
            c2 = 0.5 * (-(1 - beta_1) * parents_params[0] + (3 - beta_2) * parents_params[1])

        # vSBX applies crossover with use_child_gene_prob and uniform_crossover_prob.
        # the gene of the parent individual is the gene of the child individual.
        # The original vSBX creates two child individuals,
        # but optuna's implementation creates only one child individual.
        # Therefore, when there is no crossover,
        # the gene is selected with equal probability from the parent individuals x1 and x2.

        child1_params_list = []
        child2_params_list = []

        for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, parents_params[0], parents_params[1]):
            if rng.rand() < self._use_child_gene_prob:
                if rng.rand() >= self._uniform_crossover_prob:
                    child1_params_list.append(c1_i)
                    child2_params_list.append(c2_i)
                else:
                    child1_params_list.append(c2_i)
                    child2_params_list.append(c1_i)
            else:
                if rng.rand() >= self._uniform_crossover_prob:
                    child1_params_list.append(x1_i)
                    child2_params_list.append(x2_i)
                else:
                    child1_params_list.append(x2_i)
                    child2_params_list.append(x1_i)

        child_params_list = child1_params_list if rng.rand() < 0.5 else child2_params_list
        child_params = np.array(child_params_list)

        return child_params
