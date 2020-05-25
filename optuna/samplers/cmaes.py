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
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

# Minimum value of sigma0 to avoid ZeroDivisionError.
_MIN_SIGMA0 = 1e-10


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

    .. seealso::
        You can also use :class:`optuna.integration.CmaEsSampler` which is a sampler using cma
        library as the backend.

    Args:

        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used.

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
    """

    def __init__(
        self,
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
    ) -> None:

        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._logger = optuna.logging.get_logger(__name__)
        self._cma_rng = np.random.RandomState(seed)
        self._search_space = optuna.samplers.IntersectionSearchSpace()

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial",
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

        completed_trials = [
            t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE
        ]
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

        optimizer = self._restore_or_init_optimizer(completed_trials, search_space, ordered_keys)

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
            if optimizer.generation == t.system_attrs.get("cma:generation", -1)
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

            optimizer_str = pickle.dumps(optimizer).hex()
            study._storage.set_trial_system_attr(trial._trial_id, "cma:optimizer", optimizer_str)

        # Caution: optimizer should update its seed value
        seed = self._cma_rng.randint(1, 2 ** 16) + trial.number
        optimizer._rng = np.random.RandomState(seed)
        params = optimizer.ask()

        study._storage.set_trial_system_attr(
            trial._trial_id, "cma:generation", optimizer.generation
        )
        external_values = {
            k: _to_optuna_param(search_space[k], p) for k, p in zip(ordered_keys, params)
        }
        return external_values

    def _restore_or_init_optimizer(
        self,
        completed_trials: "List[optuna.trial.FrozenTrial]",
        search_space: Dict[str, BaseDistribution],
        ordered_keys: List[str],
    ) -> CMA:

        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            serialized_optimizer = trial.system_attrs.get(
                "cma:optimizer", None
            )  # type: Optional[str]
            if serialized_optimizer is None:
                continue
            return pickle.loads(bytes.fromhex(serialized_optimizer))

        # Init a CMA object.
        if self._x0 is None:
            self._x0 = _initialize_x0(search_space)

        if self._sigma0 is None:
            sigma0 = _initialize_sigma0(search_space)
        else:
            sigma0 = self._sigma0
        sigma0 = max(sigma0, _MIN_SIGMA0)
        mean = np.array([self._x0[k] for k in ordered_keys], dtype=float)
        bounds = _get_search_space_bound(ordered_keys, search_space)
        n_dimension = len(ordered_keys)
        return CMA(
            mean=mean,
            sigma=sigma0,
            bounds=bounds,
            seed=self._cma_rng.randint(1, 2 ** 31 - 2),
            n_max_resampling=10 * n_dimension,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        if self._warn_independent_sampling:
            complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
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
        r = np.round((cma_param - math.log(distribution.low)) / math.log(distribution.step))
        v = r * math.log(distribution.step) + math.log(distribution.low)
        return int(math.exp(v))
    return cma_param


def _initialize_x0(search_space: Dict[str, BaseDistribution]) -> Dict[str, np.ndarray]:
    x0 = {}
    for name, distribution in search_space.items():
        if isinstance(distribution, optuna.distributions.UniformDistribution):
            x0[name] = np.mean([distribution.high, distribution.low])
        elif isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
            x0[name] = np.mean([distribution.high, distribution.low])
        elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
            x0[name] = int(np.mean([distribution.high, distribution.low]))
        elif isinstance(distribution, optuna.distributions.IntLogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            x0[name] = np.mean([log_high, log_low])
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            x0[name] = math.exp(np.mean([log_high, log_low]))
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
    keys: List[str], search_space: Dict[str, BaseDistribution],
) -> np.ndarray:

    bounds = []
    for param_name in keys:
        dist = search_space[param_name]
        if isinstance(
            dist,
            (
                optuna.distributions.UniformDistribution,
                optuna.distributions.LogUniformDistribution,
                optuna.distributions.DiscreteUniformDistribution,
                optuna.distributions.IntUniformDistribution,
                optuna.distributions.IntLogUniformDistribution,
            ),
        ):
            bounds.append([_to_cma_param(dist, dist.low), _to_cma_param(dist, dist.high)])
        else:
            raise NotImplementedError("The distribution {} is not implemented.".format(dist))
    return np.array(bounds, dtype=float)
