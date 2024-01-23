from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any

import numpy as np
import scipy.stats.qmc

import optuna


class ParamType(IntEnum):
    LINEAR = 0
    LOG = 1
    CATEGORICAL = 2


@dataclass(frozen=True)
class SearchSpace:
    param_type: np.ndarray
    bounds: np.ndarray
    step: np.ndarray


def untransform_one_param(
    param_value: np.ndarray, param_type: ParamType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    if param_type == ParamType.CATEGORICAL:
        return param_value
    else:
        bounds2 = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
        if param_type == ParamType.LOG:
            bounds2 = (math.log(bounds2[0]), math.log(bounds2[1]))
        param_value = param_value * (bounds2[1] - bounds2[0]) + bounds2[0]
        if param_type == ParamType.LOG:
            param_value = np.exp(param_value)
        return param_value


def transform_one_param(
    param_value: np.ndarray, param_type: ParamType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    if param_type == ParamType.CATEGORICAL:
        return param_value
    else:
        bounds2 = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
        if param_type == ParamType.LOG:
            bounds2 = (math.log(bounds2[0]), math.log(bounds2[1]))
            param_value = np.log(param_value)
        param_value = (param_value - bounds2[0]) / (bounds2[1] - bounds2[0])
        return param_value


def round_one_transformed_param(
    param_value: np.ndarray, param_type: ParamType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    assert param_type != ParamType.CATEGORICAL
    if step == 0.0:
        return param_value

    param_value = untransform_one_param(param_value, param_type, bounds, step)
    param_value = (param_value - bounds[0] + 0.5 * step) // step * step + bounds[0]
    param_value = transform_one_param(param_value, param_type, bounds, step)
    return param_value


def sample_transformed_params(n: int, search_space: SearchSpace) -> np.ndarray:
    dim = search_space.param_type.shape[0]
    param_types = search_space.param_type
    bounds = search_space.bounds
    steps = search_space.step
    qmc_engine = scipy.stats.qmc.Sobol(dim, scramble=True)
    param_values = qmc_engine.random(n)
    for i in range(dim):
        if param_types[i] == ParamType.CATEGORICAL:
            param_values[:, i] = np.floor(param_values[:, i] * bounds[i, 1])
        elif steps[i] != 0.0:
            param_values[:, i] = round_one_transformed_param(
                param_values[:, i], param_types[i], (bounds[i, 0], bounds[i, 1]), steps[i]
            )
    return param_values


def get_search_space_and_transformed_params(
    trials: list[optuna.trial.FrozenTrial],
    optuna_search_space: dict[str, optuna.distributions.BaseDistribution],
) -> tuple[SearchSpace, np.ndarray]:
    param_type = np.zeros(len(optuna_search_space), dtype=np.int64)
    bounds = np.zeros((len(optuna_search_space), 2), dtype=np.float64)
    step = np.zeros(len(optuna_search_space), dtype=np.float64)
    values = np.zeros((len(trials), len(optuna_search_space)), dtype=np.float64)
    for i, (param, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            param_type[i] = ParamType.CATEGORICAL
            bounds[i, 0] = 0.0
            bounds[i, 1] = len(distribution.choices)
            step[i] = 1.0
            for ti, trial in enumerate(trials):
                values[ti, i] = distribution.to_internal_repr(trial.params[param])
        else:
            assert isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            )
            if distribution.log:
                param_type[i] = ParamType.LOG
            else:
                param_type[i] = ParamType.LINEAR
            if distribution.step is None:
                step[i] = 0.0
            else:
                step[i] = distribution.step
            bounds[i, 0] = distribution.low
            bounds[i, 1] = distribution.high

            external_values = np.zeros((len(trials),))
            for ti, trial in enumerate(trials):
                external_values[ti] = trial.params[param]
            values[:, i] = transform_one_param(
                external_values, param_type[i], (bounds[i, 0], bounds[i, 1]), step[i]
            )
    return SearchSpace(param_type, bounds, step), values


def get_untransformed_param(
    optuna_search_space: dict[str, optuna.distributions.BaseDistribution],
    transformed_param: np.ndarray,
) -> dict[str, Any]:
    ret = {}
    for i, (param, distribution) in enumerate(optuna_search_space.items()):
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            ret[param] = distribution.to_external_repr(transformed_param[i])
        else:
            assert isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            )
            if distribution.log:
                param_type = ParamType.LOG
            else:
                param_type = ParamType.LINEAR
            if distribution.step is None:
                step = 0.0
            else:
                step = distribution.step
            bounds = (distribution.low, distribution.high)
            param_value = float(
                untransform_one_param(transformed_param[i], param_type, bounds, step)
            )
            param_value = min(max(param_value, distribution.low), distribution.high)
            if isinstance(distribution, optuna.distributions.IntDistribution):
                param_value = round(param_value)
            ret[param] = param_value
    return ret
