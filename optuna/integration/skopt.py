import copy
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy as np

import optuna
from optuna import distributions
from optuna import samplers
from optuna._deprecated import deprecated_class
from optuna._imports import try_import
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    import skopt
    from skopt.space import space


@deprecated_class("3.4.0", "4.0.0")
class SkoptSampler(BaseSampler):
    """Sampler using Scikit-Optimize as the backend.

    The use of :class:`~optuna.integration.SkoptSampler` is highly not recommended, as the
    development of Scikit-Optimize has been inactive and we have identified compatibility
    issues with newer NumPy versions.

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.SkoptSampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

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

        seed:
            Seed for random number generator.
    """

    def __init__(
        self,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        skopt_kwargs: Optional[Dict[str, Any]] = None,
        n_startup_trials: int = 1,
        *,
        consider_pruned_trials: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        _imports.check()

        self._skopt_kwargs = skopt_kwargs or {}
        if "dimensions" in self._skopt_kwargs:
            del self._skopt_kwargs["dimensions"]

        self._independent_sampler = independent_sampler or samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling
        self._n_startup_trials = n_startup_trials
        self._search_space = IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials

        if self._consider_pruned_trials:
            warnings.warn(
                "`consider_pruned_trials` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if seed is not None and "random_state" not in self._skopt_kwargs:
            self._skopt_kwargs["random_state"] = seed
        self._rng: Optional[np.random.RandomState] = None

    def reseed_rng(self) -> None:
        self._skopt_kwargs["random_state"] = random.randint(1, np.iinfo(np.int32).max)
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:
        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                if not isinstance(distribution, distributions.CategoricalDistribution):
                    # `skopt` cannot handle non-categorical distributions that contain just
                    # a single value, so we skip this distribution.
                    #
                    # Note that `Trial` takes care of this distribution during suggestion.
                    continue

            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        complete_trials = self._get_trials(study)
        if len(complete_trials) < self._n_startup_trials:
            return {}

        optimizer = _Optimizer(search_space, self._skopt_kwargs)
        if self._rng is not None:
            optimizer._optimizer.rng = self._rng
        optimizer.tell(study, complete_trials)
        params = optimizer.ask()
        self._rng = optimizer._optimizer.rng
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

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
            "by using `{}` instead of `SkoptSampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `SkoptSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: Study) -> List[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
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

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


class _Optimizer:
    def __init__(
        self, search_space: Dict[str, distributions.BaseDistribution], skopt_kwargs: Dict[str, Any]
    ) -> None:
        self._search_space = search_space

        dimensions = []
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.IntDistribution):
                if distribution.log:
                    low = distribution.low - 0.5
                    high = distribution.high + 0.5
                    dimension = space.Real(low, high, prior="log-uniform")
                else:
                    count = (distribution.high - distribution.low) // distribution.step
                    dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices)
            elif isinstance(distribution, distributions.FloatDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                if distribution.log:
                    high = np.nextafter(distribution.high, float("-inf"))
                    dimension = space.Real(distribution.low, high, prior="log-uniform")
                elif distribution.step is not None:
                    count = int((distribution.high - distribution.low) // distribution.step)
                    dimension = space.Integer(0, count)
                else:
                    high = np.nextafter(distribution.high, float("-inf"))
                    dimension = space.Real(distribution.low, high)
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
            if isinstance(distribution, distributions.FloatDistribution):
                # Type of value is np.floating, so cast it to Python's built-in float.
                value = float(value)
                if distribution.step is not None:
                    value = value * distribution.step + distribution.low
            elif isinstance(distribution, distributions.IntDistribution):
                if distribution.log:
                    value = int(np.round(value))
                    value = min(max(value, distribution.low), distribution.high)
                else:
                    value = int(value * distribution.step + distribution.low)

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

            if isinstance(distribution, distributions.FloatDistribution):
                if distribution.step is not None:
                    param_value = (param_value - distribution.low) // distribution.step
            elif isinstance(distribution, distributions.IntDistribution):
                if not distribution.log:
                    param_value = (param_value - distribution.low) // distribution.step

            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value
