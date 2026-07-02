from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.samplers.nsgaii._mutations._base import BaseMutation


if TYPE_CHECKING:
    import numpy as np

    from optuna.study import Study


class PolynomialMutation(BaseMutation):
    """Polynomial mutation operation used by :class:`~optuna.samplers.NSGAIISampler`.

    This operator mutates a real-valued parameter according to the polynomial probability
    distribution.

    This implementation follows the polynomial mutation procedure used in the
    revision 1.1.6 of the original NSGA-II C implementation released as
    ``Multi-objective NSGA-II code in C``.

    - `Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T.
      A fast and elitist multiobjective genetic algorithm: NSGA-II.
      IEEE Transactions on Evolutionary Computation, 6(2), 182-197 (2002).
      <https://doi.org/10.1109/4235.996017>`__
    - `Multi-objective NSGA-II code in C, Revision 1.1.6
      <https://www.egr.msu.edu/~kdeb/codes.shtml>`__

    Args:
        eta:
            Distribution index. Larger values make mutated parameter values closer to
            the original value.
    """

    def __init__(self, eta: float = 20.0) -> None:
        if eta < 0:
            raise ValueError("`eta` must be a non-negative float value.")

        self._eta = eta

    def mutation(
        self,
        param: float,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> float:
        u = rng.rand()
        lb, ub = search_space_bounds
        width = ub - lb

        if width <= 0.0:
            return param

        delta1 = (param - lb) / width
        delta2 = (ub - param) / width
        mutation_power = 1.0 / (self._eta + 1.0)

        if u <= 0.5:
            xy = 1.0 - delta1
            value = 2.0 * u + (1.0 - 2.0 * u) * xy ** (self._eta + 1.0)
            delta_q = value**mutation_power - 1.0
        else:
            xy = 1.0 - delta2
            value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (self._eta + 1.0)
            delta_q = 1.0 - value**mutation_power

        return param + delta_q * width
