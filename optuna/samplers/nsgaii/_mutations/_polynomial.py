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

    - `Deb, K. and Deb, D.
      Analysing mutation schemes for real-parameter genetic algorithms.
      International Journal of Artificial Intelligence and Soft Computing, 4(1), 1 (2014).
      <https://doi.org/10.1504/IJAISC.2014.059280>`__

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

        if u <= 0.5:
            delta_l = (2.0 * u) ** (1.0 / (self._eta + 1.0)) - 1.0
            child_param = param + delta_l * (param - lb)
        else:
            delta_r = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self._eta + 1.0))
            child_param = param + delta_r * (ub - param)

        return child_param
