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
    import skopt
    from skopt.space import space


class SkoptSampler(BaseBoSampler):
    """Sampler using Scikit-Optimize as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.SkoptSampler`.

        .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform('x', -10, 10)
                    y = trial.suggest_int('y', 0, 10)
                    return x**2 + y

                sampler = optuna.integration.SkoptSampler()
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.SkoptSampler` is determined by
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

        skopt_kwargs:
            Keyword arguments passed to the constructor of
            `skopt.Optimizer <https://scikit-optimize.github.io/#skopt.Optimizer>`_
            class.

            Note that ``dimensions`` argument in ``skopt_kwargs`` will be ignored
            because it is added by :class:`~optuna.integration.SkoptSampler` automatically.

        n_startup_trials:
            The independent sampling is used until the given number of trials finish in the
            same study.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

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
        skopt_kwargs: Optional[Dict[str, Any]] = None,
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

        self._skopt_kwargs = skopt_kwargs or {}
        if "dimensions" in self._skopt_kwargs:
            del self._skopt_kwargs["dimensions"]

        if self._consider_pruned_trials:
            self._raise_experimental_warning_for_consider_pruned_trials()

    @experimental("2.0.0", name="`consider_pruned_trials = True` in SkoptSampler")
    def _raise_experimental_warning_for_consider_pruned_trials(self) -> None:
        pass

    def _create_controller(
        self, search_space: Dict[str, distributions.BaseDistribution]
    ) -> BaseBoController:
        return _SkoptController(search_space, self._skopt_kwargs)


class _SkoptController(BaseBoController):
    def __init__(
        self, search_space: Dict[str, distributions.BaseDistribution], skopt_kwargs: Dict[str, Any]
    ) -> None:

        self._search_space = search_space

        dimensions = []
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high, prior="log-uniform")
            elif isinstance(distribution, distributions.IntUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.step
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                low = distribution.low - 0.5
                high = distribution.high + 0.5
                dimension = space.Real(low, high, prior="log-uniform")
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = int((distribution.high - distribution.low) // distribution.q)
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )

            dimensions.append(dimension)

        self._optimizer = skopt.Optimizer(dimensions, **skopt_kwargs)

    def tell(self, study: Study, complete_trials: List[FrozenTrial]) -> None:

        xs = []
        ys = []

        for trial in complete_trials:
            if not self._is_compatible(trial):
                continue

            x, y = self._complete_trial_to_skopt_observation(study, trial)
            xs.append(x)
            ys.append(y)

        self._optimizer.tell(xs, ys)

    def ask(self) -> Dict[str, Any]:

        params = {}
        param_values = self._optimizer.ask()
        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low
            if isinstance(distribution, distributions.IntUniformDistribution):
                value = value * distribution.step + distribution.low
            if isinstance(distribution, distributions.IntLogUniformDistribution):
                value = int(np.round(value))
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

    def _complete_trial_to_skopt_observation(
        self, study: Study, trial: FrozenTrial
    ) -> Tuple[List[Any], float]:

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

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
