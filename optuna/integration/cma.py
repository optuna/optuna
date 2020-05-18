import math
import random

import numpy

import optuna
from optuna import distributions
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna import type_checking

try:
    import cma

    _available = True
except ImportError as e:
    _import_error = e
    # CmaEsSampler is disabled because cma is not available.
    _available = False

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Set  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.trial import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA

# Minimum value of sigma0 to avoid ZeroDivisionError in cma.CMAEvolutionStrategy.
_MIN_SIGMA0 = 1e-10


class CmaEsSampler(BaseSampler):
    """A Sampler using cma library as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.CmaEsSampler`.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -1, 1)
                y = trial.suggest_int('y', -1, 1)
                return x**2 + y

            sampler = optuna.integration.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Note that parallel execution of trials may affect the optimization performance of CMA-ES,
    especially if the number of trials running in parallel exceeds the population size.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used.
            Please refer to cma.CMAEvolutionStrategy_ for further details of ``x0``.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space. If distribution is categorical, ``min_range`` is
            ``len(choices) - 1``.
            Please refer to cma.CMAEvolutionStrategy_ for further details of ``sigma0``.

        cma_stds:
            A dictionary of multipliers of sigma0 for each parameters. The default value is 1.0.
            Please refer to cma.CMAEvolutionStrategy_ for further details of ``cma_stds``.

        seed:
            A random seed for CMA-ES.

        cma_opts:
            Options passed to the constructor of cma.CMAEvolutionStrategy_ class.

            Note that ``BoundaryHandler``, ``bounds``, ``CMA_stds`` and ``seed`` arguments in
            ``cma_opts`` will be ignored because it is added by
            :class:`~optuna.integration.CmaEsSampler` automatically.

        n_startup_trials:
            The independent sampling is used instead of the CMA-ES algorithm until the given number
            of trials finish in the same study.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.CmaEsSampler` is determined by
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

    .. _cma.CMAEvolutionStrategy: http://cma.gforge.inria.fr/apidocs-pycma/\
    cma.evolution_strategy.CMAEvolutionStrategy.html
    """

    def __init__(
        self,
        x0=None,  # type: Optional[Dict[str, Any]]
        sigma0=None,  # type: Optional[float]
        cma_stds=None,  # type: Optional[Dict[str, float]]
        seed=None,  # type: Optional[int]
        cma_opts=None,  # type: Optional[Dict[str, Any]]
        n_startup_trials=1,  # type: int
        independent_sampler=None,  # type: Optional[BaseSampler]
        warn_independent_sampling=True,  # type: bool
    ):
        # type: (...) -> None

        _check_cma_availability()

        self._x0 = x0
        self._sigma0 = sigma0
        self._cma_stds = cma_stds
        if seed is None:
            seed = random.randint(1, 2 ** 32)
        self._cma_opts = cma_opts or {}
        self._cma_opts["seed"] = seed
        self._cma_opts.setdefault("verbose", -2)
        self._n_startup_trials = n_startup_trials
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling
        self._logger = optuna.logging.get_logger(__name__)
        self._search_space = optuna.samplers.IntersectionSearchSpace()

    def reseed_rng(self) -> None:

        self._cma_opts["seed"] = random.randint(1, 2 ** 32)
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            search_space[name] = distribution

        return search_space

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> float

        if self._warn_independent_sampling:
            complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            return {}

        if len(search_space) == 1:
            self._logger.info(
                "`CmaEsSampler` does not support optimization of 1-D search space. "
                "`{}` is used instead of `CmaEsSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            self._warn_independent_sampling = False
            return {}

        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(complete_trials) < self._n_startup_trials:
            return {}

        if self._x0 is None:
            self._x0 = self._initialize_x0(search_space)

        if self._sigma0 is None:
            sigma0 = self._initialize_sigma0(search_space)
        else:
            sigma0 = self._sigma0
        sigma0 = max(sigma0, _MIN_SIGMA0)

        optimizer = _Optimizer(search_space, self._x0, sigma0, self._cma_stds, self._cma_opts)
        trials = study.trials
        last_told_trial_number = optimizer.tell(trials, study.direction)
        return optimizer.ask(trials, last_told_trial_number)

    @staticmethod
    def _initialize_x0(search_space):
        # type: (Dict[str, BaseDistribution]) -> Dict[str, Any]

        x0 = {}
        for name, distribution in search_space.items():
            # TODO(nzw0301) support IntLogUniform
            if isinstance(distribution, UniformDistribution):
                x0[name] = numpy.mean([distribution.high, distribution.low])
            elif isinstance(distribution, DiscreteUniformDistribution):
                x0[name] = numpy.mean([distribution.high, distribution.low])
            elif isinstance(distribution, IntUniformDistribution):
                x0[name] = int(numpy.mean([distribution.high, distribution.low]))
            elif isinstance(distribution, LogUniformDistribution):
                log_high = math.log(distribution.high)
                log_low = math.log(distribution.low)
                x0[name] = math.exp(numpy.mean([log_high, log_low]))
            elif isinstance(distribution, CategoricalDistribution):
                index = (len(distribution.choices) - 1) // 2
                x0[name] = distribution.choices[index]
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )
        return x0

    @staticmethod
    def _initialize_sigma0(search_space):
        # type: (Dict[str, BaseDistribution]) -> float

        sigma0s = []
        for name, distribution in search_space.items():
            # TODO(nzw0301) support IntLogUniform
            if isinstance(distribution, UniformDistribution):
                sigma0s.append((distribution.high - distribution.low) / 6)
            elif isinstance(distribution, DiscreteUniformDistribution):
                sigma0s.append((distribution.high - distribution.low) / 6)
            elif isinstance(distribution, IntUniformDistribution):
                sigma0s.append((distribution.high - distribution.low) / 6)
            elif isinstance(distribution, LogUniformDistribution):
                log_high = math.log(distribution.high)
                log_low = math.log(distribution.low)
                sigma0s.append((log_high - log_low) / 6)
            elif isinstance(distribution, CategoricalDistribution):
                sigma0s.append((len(distribution.choices) - 1) / 6)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )
        return min(sigma0s)

    def _log_independent_sampling(self, trial, param_name):
        # type: (FrozenTrial, str) -> None

        self._logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `CmaEsSampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `CmaEsSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )


class _Optimizer(object):
    def __init__(
        self,
        search_space,  # type: Dict[str, BaseDistribution]
        x0,  # type: Dict[str, Any]
        sigma0,  # type: float
        cma_stds,  # type: Optional[Dict[str, float]]
        cma_opts,  # type: Dict[str, Any]
    ):
        # type: (...) -> None

        self._search_space = search_space
        self._param_names = list(sorted(self._search_space.keys()))

        lows = []
        highs = []
        for param_name in self._param_names:
            dist = self._search_space[param_name]
            if isinstance(dist, CategoricalDistribution):
                # Handle categorical values by ordinal representation.
                # TODO(Yanase): Support one-hot representation.
                lows.append(-0.5)
                highs.append(len(dist.choices) - 0.5)
            elif isinstance(dist, UniformDistribution) or isinstance(dist, LogUniformDistribution):
                lows.append(self._to_cma_params(search_space, param_name, dist.low))
                highs.append(self._to_cma_params(search_space, param_name, dist.high))
            elif isinstance(dist, DiscreteUniformDistribution):
                r = dist.high - dist.low
                lows.append(0 - 0.5 * dist.q)
                highs.append(r + 0.5 * dist.q)
            elif isinstance(dist, IntUniformDistribution):
                lows.append(dist.low - 0.5)
                highs.append(dist.high + 0.5)
            else:
                raise NotImplementedError("The distribution {} is not implemented.".format(dist))

        # Set initial params.
        initial_cma_params = []
        for param_name in self._param_names:
            initial_cma_params.append(
                self._to_cma_params(self._search_space, param_name, x0[param_name])
            )
        cma_option = {
            "BoundaryHandler": cma.BoundTransform,
            "bounds": [lows, highs],
        }

        if cma_stds:
            cma_option["CMA_stds"] = [cma_stds.get(name, 1.0) for name in self._param_names]

        cma_opts.update(cma_option)

        self._es = cma.CMAEvolutionStrategy(initial_cma_params, sigma0, cma_opts)

    def tell(self, trials, study_direction):
        # type: (List[FrozenTrial], StudyDirection) -> int

        complete_trials = self._collect_target_trials(trials, target_states={TrialState.COMPLETE})

        popsize = self._es.popsize
        generation = len(complete_trials) // popsize
        last_told_trial_number = -1
        for i in range(generation):
            xs = []
            ys = []
            for t in complete_trials[i * popsize : (i + 1) * popsize]:
                x = [
                    self._to_cma_params(self._search_space, name, t.params[name])
                    for name in self._param_names
                ]
                xs.append(x)
                ys.append(t.value)
                last_told_trial_number = t.number
            if study_direction == StudyDirection.MAXIMIZE:
                ys = [-1 * y if y is not None else y for y in ys]

            # Calling `ask` is required to avoid RuntimeError which claims that `tell` should only
            # be called once per iteration.
            self._es.ask()
            self._es.tell(xs, ys)
        return last_told_trial_number

    def ask(self, trials, last_told_trial_number):
        # type: (List[FrozenTrial], int) -> Dict[str, Any]

        individual_index = len(self._collect_target_trials(trials, last_told_trial_number))
        popsize = self._es.popsize

        # individual_index may exceed the population size due to the parallel execution of multiple
        # trials. In such cases, `cma.cma.CMAEvolutionStrategy.ask` is called multiple times in an
        # iteration, and that may affect the optimization performance of CMA-ES.
        # In addition, please note that some trials may suggest the same parameters when multiple
        # samplers invoke this method simultaneously.
        while individual_index >= popsize:
            individual_index -= popsize
            self._es.ask()
        cma_params = self._es.ask()[individual_index]

        ret_val = {}
        for param_name, value in zip(self._param_names, cma_params):
            ret_val[param_name] = self._to_optuna_params(self._search_space, param_name, value)
        return ret_val

    def _is_compatible(self, trial):
        # type: (FrozenTrial) -> bool

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

    def _collect_target_trials(self, trials, last_told=-1, target_states=None):
        # type: (List[FrozenTrial], int, Optional[Set[TrialState]]) -> List[FrozenTrial]

        target_trials = [t for t in trials if t.number > last_told]
        target_trials = [t for t in target_trials if self._is_compatible(t)]
        if target_states is not None:
            target_trials = [t for t in target_trials if t.state in target_states]

        return target_trials

    @staticmethod
    def _to_cma_params(search_space, param_name, optuna_param_value):
        # type: (Dict[str, BaseDistribution], str, Any) -> float

        dist = search_space[param_name]
        if isinstance(dist, LogUniformDistribution):
            return math.log(optuna_param_value)
        elif isinstance(dist, DiscreteUniformDistribution):
            return optuna_param_value - dist.low
        elif isinstance(dist, CategoricalDistribution):
            return dist.choices.index(optuna_param_value)
        return optuna_param_value

    @staticmethod
    def _to_optuna_params(search_space, param_name, cma_param_value):
        # type: (Dict[str, BaseDistribution], str, float) -> Any

        dist = search_space[param_name]
        if isinstance(dist, LogUniformDistribution):
            return math.exp(cma_param_value)
        if isinstance(dist, DiscreteUniformDistribution):
            v = numpy.round(cma_param_value / dist.q) * dist.q + dist.low
            # v may slightly exceed range due to round-off errors.
            return float(min(max(v, dist.low), dist.high))
        if isinstance(dist, IntUniformDistribution):
            r = numpy.round((cma_param_value - dist.low) / dist.step)
            v = r * dist.step + dist.low
            return v
        if isinstance(dist, CategoricalDistribution):
            v = int(numpy.round(cma_param_value))
            return dist.choices[v]
        return cma_param_value


def _check_cma_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "cma library is not available. Please install cma to use this feature. "
            "cma can be installed by executing `$ pip install cma`. "
            "For further information, please refer to the installation guide of cma. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
