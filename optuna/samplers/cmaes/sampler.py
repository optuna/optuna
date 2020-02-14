import math
import pickle
import numpy as np
import optuna

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.structs import FrozenTrial
from optuna.structs import TrialState

from .cma import _CMA

# Minimum value of sigma0 to avoid ZeroDivisionError.
_MIN_SIGMA0 = 1e-10


class CmaEsSampler(BaseSampler):
    def __init__(
            self,
            x0: Optional[Dict[str, Any]] = None,
            sigma0: Optional[float] = None,
            n_startup_trials: int = 1,
            independent_sampler: Optional[BaseSampler] = None,
            warn_independent_sampling: bool = True,
            seed: Optional[int] = None,
    ):
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = (
            independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        )
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._logger = optuna.logging.get_logger(__name__)
        self._cma_rng = np.random.RandomState(seed)

    def infer_relative_search_space(
            self, study: 'optuna.Study', trial: 'optuna.structs.FrozenTrial',
    ) -> 'Dict[str, BaseDistribution]':
        # Import here to avoid circular imports without breaking a backward compatibility.
        from optuna.samplers import intersection_search_space

        search_space = {}
        if trial.number < self._n_startup_trials:
            return {}

        for name, distribution in intersection_search_space(study).items():
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
                    ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
            self,
            study: 'optuna.Study',
            trial: 'optuna.structs.FrozenTrial',
            search_space: 'Dict[str, BaseDistribution]',
    ) -> Dict[str, Any]:
        if len(search_space) == 0:
            return {}

        completed_trials = [
            t
            for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE
        ]
        if len(completed_trials) < self._n_startup_trials:
            return {}

        ordered_keys = [key for key in search_space]
        ordered_keys.sort()

        optimizer = self._restore_or_init_optimizer(
            completed_trials, search_space, ordered_keys
        )

        solution_trials = [
            t
            for t in completed_trials
            if optimizer.generation == t.system_attrs.get("cma:generation", -1)
        ]
        if len(solution_trials) >= optimizer.population_size:
            solutions = []
            for t in solution_trials[: optimizer.population_size]:
                x = np.array([t.params[k] for k in ordered_keys])
                solutions.append((x, t.value))
            optimizer.tell(solutions)
            pickled_optimizer = pickle.dumps(optimizer)
            if isinstance(study._storage, optuna.storages.InMemoryStorage):
                study._storage.set_trial_system_attr(
                    trial._trial_id, "cma:optimizer", pickled_optimizer
                )
            else:
                # RDB storage does not accept bytes object.
                study._storage.set_trial_system_attr(
                    trial._trial_id, "cma:optimizer", pickled_optimizer.hex()
                )

        # Caution: optimizer should update its seed value
        seed = self._cma_rng.randint(1, 2 ** 16) + trial.number
        optimizer._rng = np.random.RandomState(seed)
        params = optimizer.ask()

        study._storage.set_trial_system_attr(
            trial._trial_id, "cma:generation", optimizer.generation
        )
        external_values = {
            k: _to_external_repr(search_space, k, p)
            for k, p in zip(ordered_keys, params)
        }
        return external_values

    def _restore_or_init_optimizer(
            self,
            completed_trials: 'List[optuna.structs.FrozenTrial]',
            search_space: 'Dict[str, BaseDistribution]',
            ordered_keys: List[str],
    ) -> _CMA:
        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            serialized_optimizer = trial.system_attrs.get(
                "cma:optimizer", None
            )  # type: Union[str, bytes, None]
            if serialized_optimizer is None:
                continue
            if isinstance(serialized_optimizer, bytes):
                return pickle.loads(serialized_optimizer)
            else:
                return pickle.loads(bytes.fromhex(serialized_optimizer))

        # Init a CMA object.
        if self._x0 is None:
            self._x0 = _initialize_x0(search_space)

        if self._sigma0 is None:
            sigma0 = _initialize_sigma0(search_space)
        else:
            sigma0 = self._sigma0
        sigma0 = max(sigma0, _MIN_SIGMA0)
        mean = np.array([self._x0[k] for k in ordered_keys])
        bounds = _get_search_space_bound(ordered_keys, search_space)
        n_dimension = len(ordered_keys)
        return _CMA(
            mean=mean,
            sigma=sigma0,
            bounds=bounds,
            seed=self._cma_rng.randint(1, 2 ** 32),
            n_max_resampling=10 * n_dimension,
        )

    def sample_independent(
            self,
            study: 'optuna.Study',
            trial: 'optuna.structs.FrozenTrial',
            param_name: str,
            param_distribution: 'BaseDistribution',
    ) -> Any:
        if self._warn_independent_sampling:
            complete_trials = [
                t for t in study.trials if t.state == TrialState.COMPLETE
            ]
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        self._logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `CMASampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `CMASampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )


def _to_external_repr(
        search_space: 'Dict[str, BaseDistribution]', param_name: str, internal_repr: float,
) -> Any:
    dist = search_space[param_name]
    if isinstance(dist, optuna.distributions.LogUniformDistribution):
        return math.exp(internal_repr)
    if isinstance(dist, optuna.distributions.DiscreteUniformDistribution):
        v = np.round(internal_repr / dist.q) * dist.q + dist.low
        # v may slightly exceed range due to round-off errors.
        return float(min(max(v, dist.low), dist.high))
    if isinstance(dist, optuna.distributions.IntUniformDistribution):
        return int(np.round(internal_repr))
    return internal_repr


def _initialize_x0(search_space: 'Dict[str, BaseDistribution]') -> Dict[str, np.ndarray]:
    x0 = {}
    for name, distribution in search_space.items():
        if isinstance(distribution, optuna.distributions.UniformDistribution):
            x0[name] = np.mean([distribution.high, distribution.low])
        elif isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
            x0[name] = np.mean([distribution.high, distribution.low])
        elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
            x0[name] = int(np.mean([distribution.high, distribution.low]))
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            x0[name] = math.exp(np.mean([log_high, log_low]))
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(distribution)
            )
    return x0


def _initialize_sigma0(search_space: 'Dict[str, BaseDistribution]') -> float:
    sigma0s = []
    for name, distribution in search_space.items():
        if isinstance(distribution, optuna.distributions.UniformDistribution):
            sigma0s.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.DiscreteUniformDistribution):
            sigma0s.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
            sigma0s.append((distribution.high - distribution.low) / 6)
        elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
            log_high = math.log(distribution.high)
            log_low = math.log(distribution.low)
            sigma0s.append((log_high - log_low) / 6)
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(distribution)
            )
    return min(sigma0s)


def _get_search_space_bound(
        keys: List[str], search_space: 'Dict[str, BaseDistribution]',
) -> np.ndarray:
    bounds = []
    for param_name in keys:
        dist = search_space[param_name]
        if isinstance(dist, optuna.distributions.UniformDistribution):
            bounds.append([dist.low, dist.high])
        else:
            raise NotImplementedError(
                "The distribution {} is not implemented.".format(dist)
            )
    return np.array(bounds)
