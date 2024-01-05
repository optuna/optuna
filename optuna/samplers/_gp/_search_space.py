import numba
from typing import NamedTuple
import numpy as np
import math
import scipy.stats.qmc
import optuna
from typing import Any

LINEAR = 0
LOG = 1
CATEGORICAL = 2

class SearchSpace(NamedTuple):
    param_type: np.ndarray
    bounds: np.ndarray
    step: np.ndarray

def untransform_one_param(x: np.ndarray, param_type: int, bounds: tuple[float, float], step: float) -> np.ndarray:
    if param_type == CATEGORICAL:
        return x
    else:
        bounds2 = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
        if param_type == LOG:
            bounds2 = (math.log(bounds2[0]), math.log(bounds2[1]))
        x = x * (bounds2[1] - bounds2[0]) + bounds2[0]
        if param_type == LOG:
            x = np.exp(x)
        return x


def transform_one_param(x: np.ndarray, param_type: int, bounds: tuple[float, float], step: float) -> np.ndarray:
    if param_type == CATEGORICAL:
        return x
    else:
        bounds2 = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
        if param_type == LOG:
            bounds2 = (math.log(bounds2[0]), math.log(bounds2[1]))
            x = np.log(x)
        x = (x - bounds2[0]) / (bounds2[1] - bounds2[0])
        return x

def round_one_transformed_param(x: np.ndarray, param_type: int, bounds: tuple[float, float], step: float) -> np.ndarray:
    assert param_type != CATEGORICAL
    if step == 0.0:
        return x
    
    x = untransform_one_param(x, param_type, bounds, step)
    x = (x - bounds[0] + 0.5 * step) // step * step + bounds[0]
    x = transform_one_param(x, param_type, bounds, step)
    return x

def sample_transformed_params(n: int, search_space: SearchSpace) -> np.ndarray:
    dim = search_space.param_type.shape[0]
    param_types, bounds, steps = search_space
    qmc_engine = scipy.stats.qmc.Sobol(dim, scramble=True)
    xs = qmc_engine.random(n)
    for i in range(dim):
        if param_types[i] == CATEGORICAL:
            xs[:, i] = np.floor(xs[:, i] * bounds[i, 1])
        elif steps[i] != 0.0:
            round_one_transformed_param(xs[:, i], param_types[i], (bounds[i, 0], bounds[i, 1]), steps[i])
    return xs

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
            param_type[i] = CATEGORICAL
            bounds[i, 0] = 0.0
            bounds[i, 1] = len(distribution.choices)
            step[i] = 1.0
            for ti, trial in enumerate(trials):
                values[ti, i] = distribution.to_internal_repr(trial.params[param])
        else:
            assert isinstance(distribution, (
                optuna.distributions.FloatDistribution, 
                optuna.distributions.IntDistribution,
            ))
            if distribution.log:
                param_type[i] = LOG
            else:
                param_type[i] = LINEAR
            if distribution.step is None:
                step[i] = 0.0
            else:
                step[i] = distribution.step
            bounds[i, 0] = distribution.low
            bounds[i, 1] = distribution.high

            external_values = np.zeros((len(trials),))
            for ti, trial in enumerate(trials):
                external_values[ti] = trial.params[param]
            values[:, i] = transform_one_param(external_values, param_type[i], (bounds[i, 0], bounds[i, 1]), step[i])
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
            assert isinstance(distribution, (
                optuna.distributions.FloatDistribution, 
                optuna.distributions.IntDistribution,
            ))
            if distribution.log:
                param_type = LOG
            else:
                param_type = LINEAR
            if distribution.step is None:
                step = 0.0
            else:
                step = distribution.step
            bounds = (distribution.low, distribution.high)
            ret[param] = untransform_one_param(transformed_param[i], param_type, bounds, step)
    return ret