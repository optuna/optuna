import copy
import math
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from cmaes import CMA
import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import logging
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_logger = logging.get_logger(__name__)

_EPS = 1e-10


class CmaEsSampler(BaseSampler):
    """A Sampler using CMA-ES algorithm.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.samplers.CmaEsSampler`.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -1, 1)
                y = trial.suggest_int('y', -1, 1)
                return x ** 2 + y

            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=20)

    Please note that this sampler does not support CategoricalDistribution.
    If your search space contains categorical parameters, I recommend you
    to use :class:`~optuna.samplers.TPESampler` instead.
    Furthermore, there is room for performance improvements in parallel
    optimization settings. This sampler cannot use some trials for updating
    the parameters of multivariate normal distribution.

    For further information about CMA-ES algorithm and its restarting strategy
    algorithm, please refer to the following papers:

    - `N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
      <https://arxiv.org/abs/1604.00772>`_
    - `A. Auger and N. Hansen. A restart CMA evolution strategy with increasing population
      size. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005),
      pages 1769–1776. IEEE Press, 2005.
      <http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf>`_

    .. seealso::
        You can also use :class:`optuna.integration.CmaEsSampler` which is a sampler using cma
        library as the backend.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used. Note that ``x0`` is sampled uniformly
            within the search space domain for each restart if you specify ``restart_strategy``
            argument.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for CMA-ES.

        n_startup_trials:
            The independent sampling is used instead of the CMA-ES algorithm until the given number
            of trials finish in the same study.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CmaEsSampler` is determined by
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

        restart_strategy:
            Strategy for restarting CMA-ES optimization when converges to a local minimum.
            If given :obj:`None`, CMA-ES will not restart (default).
            If given 'ipop', CMA-ES will restart with increasing population size.
            Please see also ``inc_popsize`` parameter.

            .. note::
                Added in v2.1.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.1.0.

        inc_popsize:
            Multiplier for increasing population size before each restart.
            This argument will be used when setting ``restart_strategy = 'ipop'``.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

            .. note::
                It is suggested to set this flag :obj:`False` when the
                :class:`~optuna.pruners.MedianPruner` is used. On the other hand, it is suggested
                to set this flag :obj:`True` when the :class:`~optuna.pruners.HyperbandPruner` is
                used. Please see `the benchmark result
                <https://github.com/optuna/optuna/pull/1229>`_ for the details.
    """

    def __init__(
        self,
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: Optional[str] = None,
        inc_popsize: int = 2
    ) -> None:
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._logger = optuna.logging.get_logger(__name__)
        self._cma_rng = np.random.RandomState(seed)
        self._search_space = optuna.samplers.IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._restart_strategy = restart_strategy
        self._inc_popsize = inc_popsize

        if self._restart_strategy:
            self._raise_experimental_warning_for_restart_strategy()

        if self._consider_pruned_trials:
            self._raise_experimental_warning_for_consider_pruned_trials()

        # TODO(c-bata): Support BIPOP-CMA-ES.
        if restart_strategy not in (
            "ipop",
            None,
        ):
            raise ValueError(
                "restart_strategy={} is unsupported. Please specify: 'ipop' or None.".format(
                    restart_strategy
                )
            )

    @experimental("2.1.0", name="`restart_strategy is not None` in CmaEsSampler")
    def _raise_experimental_warning_for_restart_strategy(self) -> None:
        pass

    @experimental("2.0.0", name="`consider_pruned_trials = True` in CmaEsSampler")
    def _raise_experimental_warning_for_consider_pruned_trials(self) -> None:
        pass

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        search_space = {}  # type: Dict[str, BaseDistribution]
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    optuna.distributions.UniformDistribution,
                    optuna.distributions.LogUniformDistribution,
                    optuna.distributions.DiscreteUniformDistribution,
                    optuna.distributions.IntUniformDistribution,
                    optuna.distributions.IntLogUniformDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            self._logger.info(
                "`CmaEsSampler` only supports two or more dimensional continuous "
                "search space. `{}` is used instead of `CmaEsSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            self._warn_independent_sampling = False
            return {}

        # TODO(c-bata): Remove `ordered_keys` by passing `ordered_dict=True`
        # to `intersection_search_space`.
        ordered_keys = [key for key in search_space]
        ordered_keys.sort()

        optimizer, n_restarts = self._restore_optimizer(completed_trials)
        if optimizer is None:
            n_restarts = 0
            optimizer = self._init_optimizer(search_space, ordered_keys)

        if self._restart_strategy is None:
            generation_attr_key = "cma:generation"  # for backward compatibility
        else:
            generation_attr_key = "cma:restart_{}:generation".format(n_restarts)

        if optimizer.dim != len(ordered_keys):
            self._logger.info(
                "`CmaEsSampler` does not support dynamic search space. "
                "`{}` is used instead of `CmaEsSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            self._warn_independent_sampling = False
            return {}

        # TODO(c-bata): Reduce the number of wasted trials during parallel optimization.
        # See https://github.com/optuna/optuna/pull/920#discussion_r385114002 for details.
        solution_trials = [
            t
            for t in completed_trials
            if optimizer.generation == t.system_attrs.get(generation_attr_key, -1)
        ]
        if len(solution_trials) >= optimizer.population_size:
            solutions = []  # type: List[Tuple[np.ndarray, float]]
            for t in solution_trials[: optimizer.population_size]:
                assert t.value is not None, "completed trials must have a value"
                x = np.array(
                    [_to_cma_param(search_space[k], t.params[k]) for k in ordered_keys],
                    dtype=float,
                )
                solutions.append((x, t.value))

            optimizer.tell(solutions)

            if self._restart_strategy == "ipop" and optimizer.should_stop():
                n_restarts += 1
                generation_attr_key = "cma:restart_{}:generation".format(n_restarts)
                popsize = optimizer.population_size * self._inc_popsize
                optimizer = self._init_optimizer(
                    search_space, ordered_keys, population_size=popsize, randomize_start_point=True
                )

            optimizer_str = pickle.dumps(optimizer).hex()
            study._storage.set_trial_system_attr(trial._trial_id, "cma:optimizer", optimizer_str)

        # Caution: optimizer should update its seed value
        seed = self._cma_rng.randint(1, 2 ** 16) + trial.number
        optimizer._rng = np.random.RandomState(seed)
        params = optimizer.ask()

        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        study._storage.set_trial_system_attr(trial._trial_id, "cma:n_restarts", n_restarts)
        external_values = {
            k: _to_optuna_param(search_space[k], p) for k, p in zip(ordered_keys, params)
        }
        return external_values

    def _restore_optimizer(
        self,
        completed_trials: "List[optuna.trial.FrozenTrial]",
    ) -> Tuple[Optional[CMA], int]:
        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            serialized_optimizer = trial.system_attrs.get(
                "cma:optimizer", None
            )  # type: Optional[str]
            if serialized_optimizer is None:
                continue
            n_restarts = trial.system_attrs.get("cma:n_restarts", 0)  # type: int
            return pickle.loads(bytes.fromhex(serialized_optimizer)), n_restarts
        return None, 0

    def _init_optimizer(
        self,
        search_space: Dict[str, BaseDistribution],
        ordered_keys: List[str],
        population_size: Optional[int] = None,
        randomize_start_point: bool = False,
    ) -> CMA:
        if randomize_start_point:
            # `_initialize_x0_randomly ` returns internal representations.
            x0 = _initialize_x0_randomly(self._cma_rng, search_space)
            mean = np.array([x0[k] for k in ordered_keys], dtype=float)
        elif self._x0 is None:
            # `_initialize_x0` returns internal representations.
            x0 = _initialize_x0(search_space)
            mean = np.array([x0[k] for k in ordered_keys], dtype=float)
        else:
            # `self._x0` is external representations.
            mean = np.array(
                [_to_cma_param(search_space[k], self._x0[k]) for k in ordered_keys], dtype=float
            )

        if self._sigma0 is None:
            sigma0 = _initialize_sigma0(search_space)
        else:
            sigma0 = self._sigma0

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, _EPS)
        bounds = _get_search_space_bound(ordered_keys, search_space)
        n_dimension = len(ordered_keys)
        return CMA(
            mean=mean,
            sigma=sigma0,
            bounds=bounds,
            seed=self._cma_rng.randint(1, 2 ** 31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
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

    def _get_trials(self, study: "optuna.Study") -> List[FrozenTrial]:
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


def _to_cma_param(distribution: BaseDistribution, optuna_param: Any) -> float:
    if isinstance(distribution, optuna.distributions.LogUniformDistribution):
        return math.log(optuna_param)
    if isinstance(distribution, optuna.distributions.IntUniformDistribution):
        return float(optuna_param)
    if isinstance(distribution, optuna.distributions.IntLogUniformDistribution):
        return math.log(optuna_param)
    return optuna_param


def _to_optuna_param(distribution: BaseDistribution, cma_param: float) -> Any:
    if isinstance(distribution, optuna.distributions.LogUniformDistribution):
        return math.exp(cma_param)
    if isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
        v = np.round(cma_param / distribution.q) * distribution.q + distribution.low
        # v may slightly exceed range due to round-off errors.
        return float(min(max(v, distribution.low), distribution.high))
    if isinstance(distribution, optuna.distributions.IntUniformDistribution):
        r = np.round((cma_param - distribution.low) / distribution.step)
        v = r * distribution.step + distribution.low
        return int(v)
    if isinstance(distribution, optuna.distributions.IntLogUniformDistribution):
        r = np.round(cma_param - math.log(distribution.low))
        v = r + math.log(distribution.low)
        return int(math.exp(v))
    return cma_param


def _initialize_x0(search_space: Dict[str, BaseDistribution]) -> Dict[str, float]:
    x0 = {}
    for name, distribution in search_space.items():
        if isinstance(
            distribution,
            (
                optuna.distributions.UniformDistribution,
                optuna.distributions.DiscreteUniformDistribution,
                optuna.distributions.IntUniformDistribution,
            ),
        ):
            x0[name] = distribution.low + (distribution.high - distribution.low) / 2
        elif isinstance(
            distribution,
            (
                optuna.distributions.LogUniformDistribution,
                optuna.distributions.IntLogUniformDistribution,
            ),
        ):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            x0[name] = log_low + (log_high - log_low) / 2
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(distribution)
            )
    return x0


def _initialize_x0_randomly(
    rng: np.random.RandomState, search_space: Dict[str, BaseDistribution]
) -> Dict[str, float]:
    x0 = {}
    for name, distribution in search_space.items():
        if isinstance(
            distribution,
            (
                optuna.distributions.UniformDistribution,
                optuna.distributions.DiscreteUniformDistribution,
                optuna.distributions.IntUniformDistribution,
            ),
        ):
            x0[name] = distribution.low + rng.rand() * (distribution.high - distribution.low)
        elif isinstance(
            distribution,
            (
                optuna.distributions.IntLogUniformDistribution,
                optuna.distributions.LogUniformDistribution,
            ),
        ):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            x0[name] = log_low + rng.rand() * (log_high - log_low)
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(distribution)
            )
    return x0


def _initialize_sigma0(search_space: Dict[str, BaseDistribution]) -> float:
    sigma0 = []
    for name, distribution in search_space.items():
        if isinstance(distribution, optuna.distributions.UniformDistribution):
            sigma0.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
            sigma0.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
            sigma0.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.IntLogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            sigma0.append((log_high - log_low) / 6)
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            sigma0.append((log_high - log_low) / 6)
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(distribution)
            )
    return min(sigma0)


def _get_search_space_bound(
    keys: List[str], search_space: Dict[str, BaseDistribution]
) -> np.ndarray:
    bounds = []
    for param_name in keys:
        dist = search_space[param_name]
        if isinstance(
            dist,
            (
                optuna.distributions.UniformDistribution,
                optuna.distributions.LogUniformDistribution,
            ),
        ):
            # These distributions cannot accept the value which equals to the upper bound.
            bounds.append([_to_cma_param(dist, dist.low), _to_cma_param(dist, dist.high) - _EPS])
        elif isinstance(
            dist,
            (
                optuna.distributions.DiscreteUniformDistribution,
                optuna.distributions.IntUniformDistribution,
                optuna.distributions.IntLogUniformDistribution,
            ),
        ):
            bounds.append([_to_cma_param(dist, dist.low), _to_cma_param(dist, dist.high)])
        else:
            raise NotImplementedError("The distribution {} is not implemented.".format(dist))
    return np.array(bounds, dtype=float)
