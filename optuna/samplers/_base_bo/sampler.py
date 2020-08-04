import abc
import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna import distributions
from optuna import samplers
from optuna.samplers._base_bo.controller import BaseBoController
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


@experimental("2.1.0")
class BaseBoSampler(BaseSampler, metaclass=abc.ABCMeta):
    """Base Sampler for the surrogate model based Bayesian optimization algorithms.

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space is determined by :func:`~optuna.samplers.intersection_search_space()`.

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
            The independent sampling is used until the given number of trials finish in the
            same study.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.
    """

    def __init__(
        self,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        n_startup_trials: int = 1,
        *,
        consider_pruned_trials: bool = False
    ) -> None:

        self._independent_sampler = independent_sampler or samplers.RandomSampler()
        self._warn_independent_sampling = warn_independent_sampling
        self._n_startup_trials = n_startup_trials
        self._search_space = samplers.IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:

        if len(search_space) == 0:
            return {}

        complete_trials = self._get_trials(study)
        if len(complete_trials) < self._n_startup_trials:
            return {}

        controller = self._create_controller(search_space)
        controller.tell(study, complete_trials)
        return controller.ask()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:

        logger = optuna.logging.get_logger(__name__)
        logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of the Bayesian optimization sampler "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of the Bayesian optimization sampler, "
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

    @abc.abstractmethod
    def _create_controller(
        self, search_space: Dict[str, distributions.BaseDistribution]
    ) -> BaseBoController:
        raise NotImplementedError
