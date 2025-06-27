from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


if TYPE_CHECKING:
    import scipy.stats.qmc as qmc

    from optuna.trial import FrozenTrial
else:
    from optuna._imports import _LazyImport

    qmc = _LazyImport("scipy.stats.qmc")


_threading_lock = threading.Lock()


class ScaleType(IntEnum):
    LINEAR = 0
    LOG = 1
    CATEGORICAL = 2


@dataclass(frozen=True)
class SearchSpace:
    scale_types: np.ndarray
    bounds: np.ndarray
    steps: np.ndarray

    @property
    def is_categorical(self) -> np.ndarray:
        return self.scale_types == ScaleType.CATEGORICAL

    def _is_log(self, param_idx: int) -> bool:
        return self.scale_types[param_idx] == ScaleType.LOG

    def choices(self, param_idx: int) -> np.ndarray:
        low, high = self.bounds[param_idx]
        if self.scale_types[param_idx] == ScaleType.CATEGORICAL:
            return np.arange(high)

        _, modified_high = self._get_modified_bounds(param_idx)
        return self.normalize_params(np.arange(low, modified_high, steps[i]), param_idx)

    def _get_modified_bounds(self, param_idx: int) -> tuple[float, float]:
        is_log = self.scale_types[param_idx] == ScaleType.LOG
        low = self.bounds[param_idx, 0] - 0.5 * self.steps[param_idx]
        high = self.bounds[param_idx, 1] + 0.5 * self.steps[param_idx]
        if is_log:
            return math.log(low), math.log(high)

        return low, high

    def unnormalize_params(self, params: np.ndarray, param_idx: int) -> np.ndarray:
        if self.scale_types[param_idx] == ScaleType.CATEGORICAL:
            return params

        low, high = self._get_modified_bounds(param_idx)
        scaled_params = params * (high - low) + low
        is_log = self.scale_types[param_idx] == ScaleType.LOG
        return np.exp(scaled_params) if is_log else scaled_params

    def normalize_params(self, params: np.ndarray, param_idx: int) -> np.ndarray:
        if self.scale_types[param_idx] == ScaleType.CATEGORICAL:
            return params

        low, high = self._get_modified_bounds(param_idx)
        if high == low:
            return np.full_like(params, 0.5)

        is_log = self.scale_types[param_idx] == ScaleType.LOG
        scaled_params = np.log(params) if is_log else params
        return (scaled_params - low) / (high - low)

    def round_normalized_param(self, params: np.ndarray, param_idx: int) -> np.ndarray:
        assert self.scale_types[param_idx] != ScaleType.CATEGORICAL
        step = self.steps[param_idx]
        if step == 0.0:
            return params
        unnormalized_params = self.unnormalize_params(params, param_idx)
        low, high = self.bounds[param_idx]
        modified_low, _ = self._get_modified_bounds(param_idx)
        return self.normalize_params(
            params=np.clip((unnormalized_params - modified_low) // step * step + low, low, high),
            param_idx=param_idx,
        )


def sample_normalized_params(
    n: int, search_space: SearchSpace, rng: np.random.RandomState | None
) -> np.ndarray:
    rng = rng or np.random.RandomState()
    dim = search_space.scale_types.shape[0]
    is_categorical = search_space.is_categorical
    bounds = search_space.bounds
    steps = search_space.steps

    # Sobol engine likely shares its internal state among threads.
    # Without threading.Lock, ValueError exceptions are raised in Sobol engine as discussed in
    # https://github.com/optuna/optunahub-registry/pull/168#pullrequestreview-2404054969
    with _threading_lock:
        qmc_engine = qmc.Sobol(dim, scramble=True, seed=rng.randint(np.iinfo(np.int32).max))
    param_values = qmc_engine.random(n)

    for i in range(dim):
        if is_categorical[i]:
            param_values[:, i] = np.floor(param_values[:, i] * bounds[i, 1])
        elif steps[i] != 0.0:
            param_values[:, i] = search_space.round_normalized_param(param_values[:, i], i)
    return param_values


def _get_gp_search_space_from_optuna_search_space(
    optuna_search_space: dict[str, BaseDistribution],
) -> SearchSpace:
    scale_types = np.zeros(len(optuna_search_space), dtype=np.int64)
    bounds = np.zeros((len(optuna_search_space), 2), dtype=np.float64)
    steps = np.zeros(len(optuna_search_space), dtype=np.float64)
    for i, (name, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, CategoricalDistribution):
            scale_types[i] = ScaleType.CATEGORICAL
            bounds[i, :] = (0.0, len(distribution.choices))
            steps[i] = 1.0
        else:
            assert isinstance(distribution, (FloatDistribution, IntDistribution))
            scale_types[i] = ScaleType.LOG if distribution.log else ScaleType.LINEAR
            steps[i] = 0.0 if distribution.step is None else distribution.step
            bounds[i, :] = (distribution.low, distribution.high)

    return SearchSpace(scale_types=scale_types, bounds=bounds, steps=steps)


def get_search_space_and_normalized_params(
    trials: list[FrozenTrial], optuna_search_space: dict[str, BaseDistribution]
) -> tuple[SearchSpace, np.ndarray]:
    gp_search_space = _get_gp_search_space_from_optuna_search_space(optuna_search_space)
    params = np.zeros((len(trials), len(optuna_search_space)), dtype=np.float64)
    for i, (name, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, CategoricalDistribution):
            params[:, i] = np.array(
                [distribution.to_internal_repr(trial.params[name]) for trial in trials]
            )
        else:
            assert isinstance(distribution, (FloatDistribution, IntDistribution))
            params[:, i] = gp_search_space.normalize_params(
                np.array([trial.params[name] for trial in trials]), param_idx=i
            )
    return gp_search_space, params


def get_unnormalized_param(
    optuna_search_space: dict[str, BaseDistribution],
    normalized_param: np.ndarray,
) -> dict[str, Any]:
    ret = {}
    gp_search_space = _get_gp_search_space_from_optuna_search_space(optuna_search_space)
    for i, (param, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, CategoricalDistribution):
            ret[param] = distribution.to_external_repr(normalized_param[i])
        else:
            assert isinstance(distribution, (FloatDistribution, IntDistribution))
            param_value = float(
                np.clip(
                    gp_search_space.unnormalize_params(normalized_param, i),
                    distribution.low,
                    distribution.high,
                )
            )
            if isinstance(distribution, IntDistribution):
                param_value = round(param_value)
            ret[param] = param_value
    return ret
