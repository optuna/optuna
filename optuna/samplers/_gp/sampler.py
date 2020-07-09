import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna._experimental import experimental
from optuna import distributions
from optuna import logging
from optuna.samplers._gp.controller import _BayesianOptimizationController
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_logger = logging.get_logger(__name__)


@experimental("2.1.0")
class GPSampler(BaseSampler):
    """Sampler based on Bayesian optimization (BO) algorithm with Gaussian process (GP) model.

    Args:
        kernel:
            The kernel used in the constructed GP model.
        model:
            The type of GP model for the Bayesian optimization.
        acquisition:
            The acquisition function for the Bayesian optimization.
        optimizer:
            The optimizer to maximize the acquisition function in each step for the Bayesian
            optimization.
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.GPSampler` is determined by
            :func:`~optuna.samplers.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.
        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.
        n_startup_trials:
            The random sampling is used instead of the BO algorithm until the given number
            of trials finish in the same study.
        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.
        seed:
            Seed for random number generator.
    """

    def __init__(
        self,
        model: str = "SVGP",
        acquisition: str = "EI",
        optimizer: str = "LBFGS",
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        n_startup_trials: int = 1,
        consider_pruned_trials: bool = False,
        seed: Optional[int] = None,
    ) -> None:

        self._boc_kwargs = {"model": model, "acquisition": acquisition, "optimizer": optimizer}
        self._independent_sampler = independent_sampler or RandomSampler()
        self._warn_independent_sampling = warn_independent_sampling
        self._n_startup_trials = n_startup_trials
        self._consider_pruned_trials = consider_pruned_trials
        self._rng = np.random.RandomState(seed)

        self._search_space = IntersectionSearchSpace()

    def reseed_rng(self) -> None:

        self._rng = np.random.RandomState()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:

        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `GPSampler` cannot handle distributions that contain just a single value,
                # so we skip them.
                # Note that the parameter values for such distributions are sampled in `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    distributions.UniformDistribution,
                    distributions.LogUniformDistribution,
                    distributions.DiscreteUniformDistribution,
                    distributions.IntUniformDistribution,
                    distributions.IntLogUniformDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:

        if len(search_space) == 0:
            return {}

        trials = self._get_trials(study)
        if len(trials) < self._n_startup_trials:
            return {}

        controller = _BayesianOptimizationController(search_space=search_space, **self._boc_kwargs)
        controller.tell(study, trials)
        return controller.ask()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        if self._warn_independent_sampling:
            trials = self._get_trials(study)
            if len(trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:

        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `GPSampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `GPSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: Study) -> List[FrozenTrial]:
        complete_trials = []
        for t in study.get_trials(deepcopy=False):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials
