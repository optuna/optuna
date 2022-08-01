from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
import warnings

import numpy as np

import optuna
from optuna import multi_objective
from optuna._deprecated import deprecated_class
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.multi_objective.samplers import _MultiObjectiveSamplerAdapter
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.pruners import NopPruner
from optuna.samplers import MOTPESampler
from optuna.samplers._tpe.multi_objective_sampler import _default_weights_above
from optuna.samplers._tpe.multi_objective_sampler import default_gamma
from optuna.study import create_study
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


@deprecated_class("2.4.0", "4.0.0")
class MOTPEMultiObjectiveSampler(BaseMultiObjectiveSampler):
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
            num_variables = 9
            n_startup_trials = 11 * num_variables - 1


            def objective(trial):
                x = []
                for i in range(1, num_variables + 1):
                    x.append(trial.suggest_float(f"x{i}", 0.0, 2.0 * i))
                return x


            sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(
                n_startup_trials=n_startup_trials, n_ehvi_candidates=24, seed=seed
            )
            study = optuna.multi_objective.create_study(
                ["minimize"] * num_variables, sampler=sampler
            )
            study.optimize(objective, n_trials=250)
    """

    def __init__(
        self,
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            self._motpe_sampler = MOTPESampler(
                consider_prior=consider_prior,
                prior_weight=prior_weight,
                consider_magic_clip=consider_magic_clip,
                consider_endpoints=consider_endpoints,
                n_startup_trials=n_startup_trials,
                n_ehvi_candidates=n_ehvi_candidates,
                gamma=gamma,
                weights_above=weights_above,
                seed=seed,
            )

    def reseed_rng(self) -> None:
        self._motpe_sampler.reseed_rng()

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        return self._motpe_sampler.sample_independent(
            _create_study(study), _create_trial(trial), param_name, param_distribution
        )


def _create_study(mo_study: "multi_objective.study.MultiObjectiveStudy") -> "optuna.Study":
    study = create_study(
        storage=mo_study._storage,
        sampler=_MultiObjectiveSamplerAdapter(mo_study.sampler),
        pruner=NopPruner(),
        study_name="_motpe-" + mo_study._storage.get_study_name_from_id(mo_study._study_id),
        directions=mo_study.directions,
        load_if_exists=True,
    )
    for mo_trial in mo_study.trials:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            study.add_trial(_create_trial(mo_trial))
    return study


def _create_trial(mo_trial: "multi_objective.trial.FrozenMultiObjectiveTrial") -> FrozenTrial:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        trial = create_trial(
            state=mo_trial.state,
            values=mo_trial.values,  # type: ignore
            params=mo_trial.params,
            distributions=mo_trial.distributions,
            user_attrs=mo_trial.user_attrs,
            system_attrs=mo_trial.system_attrs,
        )
    return trial
