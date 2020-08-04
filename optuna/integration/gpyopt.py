from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from optuna._experimental import experimental
from optuna._imports import try_import
from optuna import distributions
from optuna.samplers import BaseBoController
from optuna.samplers import BaseBoSampler
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial

with try_import() as _imports:
    import GPyOpt


@experimental("2.1.0")
class GpyoptSampler(BaseBoSampler):
    """Sampler using GPyOpt as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.GpyoptSampler`.

        .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform('x', -10, 10)
                    y = trial.suggest_int('y', 0, 10)
                    return x**2 + y

                sampler = optuna.integration.GpyoptSampler()
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.GpyoptSampler` is determined by
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

        gpyopt_kwargs:
            Keyword arguments passed to the `GPyOpt`.

        n_startup_trials:
            The independent sampling is used until the given number of trials finish in the
            same study.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                As the number of trials :math:`n` increases, each sampling takes longer and longer
                on a scale of :math:`O(n^3)`. And, if this is :obj:`True`, the number of trials
                will increase. So, it is suggested to set this flag :obj:`False` when each
                evaluation of the objective function is relatively faster than each sampling. On
                the other hand, it is suggested to set this flag :obj:`True` when each evaluation
                of the objective function is relatively slower than each sampling.
    """

    def __init__(
        self,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        gpyopt_kwargs: Optional[Dict[str, Any]] = None,
        n_startup_trials: int = 1,
        *,
        consider_pruned_trials: bool = False
    ) -> None:

        super().__init__(
            independent_sampler=independent_sampler,
            warn_independent_sampling=warn_independent_sampling,
            n_startup_trials=n_startup_trials,
            consider_pruned_trials=consider_pruned_trials,
        )

        _imports.check()

        self._gpyopt_kwargs = gpyopt_kwargs or {}
        if "domain" in self._gpyopt_kwargs:
            del self._gpyopt_kwargs["domain"]

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:

        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
            search_space[name] = distribution

        return search_space

    def _create_controller(
        self, search_space: Dict[str, distributions.BaseDistribution]
    ) -> BaseBoController:
        return _GpyoptController(search_space, self._gpyopt_kwargs)


class _GpyoptController(BaseBoController):
    def __init__(
        self,
        search_space: Dict[str, distributions.BaseDistribution],
        gpyopt_kwargs: Dict[str, Any],
    ) -> None:

        self._search_space = search_space
        self._gpyopt_kwargs = gpyopt_kwargs
        self._bo = None

        self._domain = []
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (gpyopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = {
                    "name": name,
                    "type": "continuous",
                    "domain": (distribution.low, high),
                }
            elif isinstance(distribution, distributions.LogUniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (gpyopt).
                high = np.log(np.nextafter(distribution.high, float("-inf")))
                dimension = {
                    "name": name,
                    "type": "continuous",
                    "domain": (np.log(distribution.low), high),
                }
            elif isinstance(distribution, distributions.IntUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.step
                dimension = {"name": name, "type": "discrete", "domain": (0, count)}
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                low = np.log(distribution.low - 0.5)
                high = np.log(distribution.high + 0.5)
                dimension = {"name": name, "type": "continuous", "domain": (low, high)}
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = int((distribution.high - distribution.low) // distribution.q)
                dimension = {"name": name, "type": "discrete", "domain": (0, count)}
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = {"name": name, "type": "categorical", "domain": distribution.choices}
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )

            self._domain.append(dimension)

    def tell(self, study: Study, complete_trials: List[FrozenTrial]) -> None:

        xs = []
        ys = []

        for trial in complete_trials:
            if not self._is_compatible(trial):
                continue

            x, y = self._complete_trial_to_observation_pair(study, trial)
            xs.append(x)
            ys.append(y)

        xs = np.asarray(xs)
        ys = np.asarray(ys).reshape((len(ys), 1))
        self._bo = GPyOpt.methods.BayesianOptimization(
            f=None, domain=self._domain, X=xs, Y=ys, **self._gpyopt_kwargs
        )

    def ask(self) -> Dict[str, Any]:
        assert self._bo is not None

        params = {}
        param_values = self._bo.suggest_next_locations()[0]
        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.LogUniformDistribution):
                value = np.exp(value)
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low
            if isinstance(distribution, distributions.IntUniformDistribution):
                value = value * distribution.step + distribution.low
            if isinstance(distribution, distributions.IntLogUniformDistribution):
                value = int(np.round(np.exp(value)))
                value = min(max(value, distribution.low), distribution.high)

            params[name] = value

        return params

    def _is_compatible(self, trial: FrozenTrial) -> bool:

        # Thanks to `intersection_search_space()` function, in sequential optimization,
        # the parameters of complete trials are always compatible with the search space.
        #
        # However, in distributed optimization, incompatible trials may complete on a worker
        # just after an intersection search space is calculated on another worker.

        for name, distribution in self._search_space.items():
            if name not in trial.params:
                return False

            distributions.check_distribution_compatibility(distribution, trial.distributions[name])
            param_value = trial.params[name]
            param_internal_value = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_internal_value):
                return False

        return True

    def _complete_trial_to_observation_pair(
        self, study: Study, trial: FrozenTrial
    ) -> Tuple[List[Any], float]:

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

            if isinstance(
                distribution,
                (distributions.LogUniformDistribution, distributions.IntLogUniformDistribution,),
            ):
                param_value = np.log(param_value)
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.q
            if isinstance(distribution, distributions.IntUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.step

            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value
