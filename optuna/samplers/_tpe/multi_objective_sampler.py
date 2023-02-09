from typing import Callable
from typing import Optional

import numpy as np

from optuna._deprecated import deprecated_class
from optuna.samplers._tpe.sampler import TPESampler


EPS = 1e-12


def default_gamma(x: int) -> int:
    return int(np.floor(0.1 * x))


def _default_weights_above(x: int) -> np.ndarray:
    return np.ones(x)


@deprecated_class("2.9.0", "4.0.0")
class MOTPESampler(TPESampler):
    """Multi-objective sampler using the MOTPE algorithm.

    This sampler is a multiobjective version of :class:`~optuna.samplers.TPESampler`.

    For further information about MOTPE algorithm, please refer to the following paper:

    - `Multiobjective tree-structured parzen estimator for computationally expensive optimization
      problems <https://dl.acm.org/doi/abs/10.1145/3377930.3389817>`_
    - `Multiobjective Tree-Structured Parzen Estimator <https://doi.org/10.1613/jair.1.13188>`_

    Args:
        consider_prior:
            Enhance the stability of Parzen estimator by imposing a Gaussian prior when
            :obj:`True`. The prior is only effective if the sampling distribution is
            either :class:`~optuna.distributions.FloatDistribution`,
            or :class:`~optuna.distributions.IntDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.FloatDistribution`,
            :class:`~optuna.distributions.IntDistribution`, and
            :class:`~optuna.distributions.CategoricalDistribution`.
        consider_magic_clip:
            Enable a heuristic to limit the smallest variances of Gaussians used in
            the Parzen estimator.
        consider_endpoints:
            Take endpoints of domains into account when calculating variances of Gaussians
            in Parzen estimator. See the original paper for details on the heuristics
            to calculate the variances.
        n_startup_trials:
            The random sampling is used instead of the MOTPE algorithm until the given number
            of trials finish in the same study. 11 * number of variables - 1 is recommended in the
            original paper.
        n_ehvi_candidates:
            Number of candidate samples used to calculate the expected hypervolume improvement.
        gamma:
            A function that takes the number of finished trials and returns the number of trials to
            form a density function for samples with low grains. See the original paper for more
            details.
        weights_above:
            A function that takes the number of finished trials and returns a weight for them. As
            default, weights are automatically calculated by the MOTPE's default strategy.
        seed:
            Seed for random number generator.

    .. note::
        Initialization with Latin hypercube sampling may improve optimization performance.
        However, the current implementation only supports initialization with random sampling.

    Example:

        .. testcode::

            import optuna

            seed = 128
            num_variables = 2
            n_startup_trials = 11 * num_variables - 1


            def objective(trial):
                x = []
                for i in range(1, num_variables + 1):
                    x.append(trial.suggest_float(f"x{i}", 0.0, 2.0 * i))
                return x


            sampler = optuna.samplers.MOTPESampler(
                n_startup_trials=n_startup_trials, n_ehvi_candidates=24, seed=seed
            )
            study = optuna.create_study(directions=["minimize"] * num_variables, sampler=sampler)
            study.optimize(objective, n_trials=n_startup_trials + 10)
    """

    def __init__(
        self,
        *,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = True,
        n_startup_trials: int = 10,
        n_ehvi_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights_above: Callable[[int], np.ndarray] = _default_weights_above,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ehvi_candidates,
            gamma=gamma,
            weights=weights_above,
            seed=seed,
        )
