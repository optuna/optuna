from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


_NUMERICAL_DISTRIBUTIONS = (
    UniformDistribution,
    LogUniformDistribution,
    DiscreteUniformDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
)


def crossover(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> Dict:
    if crossover_name == "uniform":
        parents = _selection(study, parent_population, 2, rng, dominates)
        child = _uniform_crossover(parents, rng, swapping_prob)
    elif crossover_name == "blx_alpha":
        parents = _selection(study, parent_population, 2, rng, dominates)
        alpha = 0.5
        child = _blx_alpha(
            parents,
            search_space,
            rng,
            swapping_prob,
            max_resampling_count,
            alpha,
        )
    elif crossover_name == "sbx":
        if len(study.directions) == 1:
            eta = 2
        else:
            eta = 20
        parents = _selection(study, parent_population, 2, rng, dominates)
        child = _sbx(parents, search_space, rng, swapping_prob, max_resampling_count, eta)
    elif crossover_name == "spx":
        n_select = 3
        parents = _selection(study, parent_population, n_select, rng, dominates)
        child = _spx(
            parents,
            search_space,
            rng,
            swapping_prob,
            max_resampling_count,
        )
    else:
        pass
    return child


def _selection(
    study: Study,
    parent_population: Sequence[FrozenTrial],
    n_select: int,
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> List[FrozenTrial]:

    parents = []
    for _ in range(n_select):
        parent = _select_parent(
            study, [t for t in parent_population if t not in parents], rng, dominates
        )
        parents.append(parent)
    return parents


def _select_parent(
    study: Study,
    population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> FrozenTrial:
    # TODO(ohta): Consider to allow users to specify the number of parent candidates.
    population_size = len(population)
    candidate0 = population[rng.choice(population_size)]
    candidate1 = population[rng.choice(population_size)]

    # TODO(ohta): Consider crowding distance.
    if dominates(candidate0, candidate1, study.directions):
        return candidate0
    else:
        return candidate1


def _swap(p0_i: Any, p1_i: Any, rand: float, swapping_prob: float) -> Any:
    if rand < swapping_prob:
        return p1_i
    else:
        return p0_i


def _uniform_crossover(
    parents: List[FrozenTrial], rng: np.random.RandomState, swapping_prob: float
) -> Dict:

    child = {}
    p0, p1 = parents[0], parents[1]
    for param_name in p0.params.keys():
        p0_i, p1_i = p0.params[param_name], p1.params[param_name]
        param = _swap(p0_i, p1_i, rng.rand(), swapping_prob)
        child[param_name] = param
    return child


def _blx_alpha(
    parents: List[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
    alpha: float,
) -> Dict:

    child = {}
    p0, p1 = parents[0], parents[1]

    for param_name in search_space.keys():
        param_distribution = search_space[param_name]
        p0_i, p1_i = p0.params[param_name], p1.params[param_name]

        # categorical data operates on uniform crossover
        if isinstance(param_distribution, CategoricalDistribution):
            param = _swap(p0_i, p1_i, rng.rand(), swapping_prob)
            child[param_name] = param
            continue

        assert isinstance(param_distribution, _NUMERICAL_DISTRIBUTIONS)

        # BLX-alpha
        trans = _SearchSpaceTransform({param_name: param_distribution})
        x0_i = trans.transform({param_name: p0.params[param_name]})[0]
        x1_i = trans.transform({param_name: p1.params[param_name]})[0]
        count = 0
        while True:
            param_array = _blx_alpha_core(x0_i, x1_i, rng, alpha)
            params = trans.untransform(param_array)
            param = float(params.get(param_name, None))
            if param_distribution._contains(param):
                break
            if count >= max_resampling_count:
                param = np.clip(param, param_distribution.low, param_distribution.high)
                break
            count += 1
        child[param_name] = param

    return child


def _blx_alpha_core(
    x0_i: np.float64, x1_i: np.float64, rng: np.random.RandomState, alpha: float
) -> np.ndarray:
    x = np.stack([x0_i, x1_i])
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    diff = alpha * (x_max - x_min)
    low = x_min - diff
    high = x_max + diff
    r = rng.uniform(0, 1, size=1)
    param = (high - low) * r + low
    return param


def _sbx(
    parents: List[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
    eta: float,
) -> Dict:
    child = {}
    p0, p1 = parents[0], parents[1]

    for param_name in search_space.keys():
        param_distribution = search_space[param_name]
        p0_i, p1_i = p0.params[param_name], p1.params[param_name]

        # categorical data operates on uniform crossover
        if isinstance(param_distribution, CategoricalDistribution):
            param = _swap(p0_i, p1_i, rng.rand(), swapping_prob)
            child[param_name] = param
            continue

        assert isinstance(param_distribution, _NUMERICAL_DISTRIBUTIONS)

        # SBX
        trans = _SearchSpaceTransform({param_name: param_distribution})
        x0_i = trans.transform({param_name: p0.params[param_name]})[0]
        x1_i = trans.transform({param_name: p1.params[param_name]})[0]
        count = 0
        while True:
            param_array = _sbx_core(
                x0_i, x1_i, param_distribution.low, param_distribution.high, rng, eta
            )

            params = trans.untransform(param_array)
            param = params.get(param_name, None)
            if param_distribution._contains(param):
                break
            if count >= max_resampling_count:
                param = np.clip(param, param_distribution.low, param_distribution.high)
                break
            count += 1
        child[param_name] = param

    return child


def _sbx_core(
    x0_i: np.float64,
    x1_i: np.float64,
    xl: float,
    xu: float,
    rng: np.random.RandomState,
    eta: float,
) -> np.ndarray:
    # https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1589-07.pdf
    x_min = min(float(x0_i), float(x1_i))
    x_max = max(float(x0_i), float(x1_i))

    x_diff = np.clip(x_max - x_min, 1e-10, None)
    beta1 = 1 + 2 * (x_min - xl) / x_diff
    beta2 = 1 + 2 * (xu - x_max) / x_diff
    alpha1 = 2 - np.power(beta1, -(eta + 1))
    alpha2 = 2 - np.power(beta2, -(eta + 1))

    r = rng.uniform(0, 1, size=x0_i.size)
    if r <= 1 / alpha1:
        betaq1 = np.power(r * alpha1, 1 / (eta + 1))
    else:
        betaq1 = np.power((1 / (2 - r * alpha1)), 1 / (eta + 1))

    if r <= 1 / alpha2:
        betaq2 = np.power(r * alpha2, 1 / (eta + 1))
    else:
        betaq2 = np.power((1 / (2 - r * alpha2)), 1 / (eta + 1))

    c1 = 0.5 * ((x_min + x_max) - betaq1 * x_diff)
    c2 = 0.5 * ((x_min + x_max) + betaq2 * x_diff)

    if rng.rand() < 0.5:
        v = c1
    else:
        v = c2

    return v


def _spx(
    parents: List[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
) -> Dict:

    child = {}
    parents_not_categorical_params: List[List[np.float64]] = [[] for _ in range(len(parents))]
    transes = []
    param_names = []
    for param_name in search_space.keys():
        param_distribution = search_space[param_name]

        parents_param = [p.params[param_name] for p in parents]

        # categorical data operates on uniform crossover
        if isinstance(search_space[param_name], CategoricalDistribution):
            param = _swap(parents_param[0], parents_param[-1], rng.rand(), swapping_prob)
            child[param_name] = param
            continue

        trans = _SearchSpaceTransform({param_name: param_distribution})
        transes.append(trans)
        param_names.append(param_name)
        for parent_index, trial in enumerate(parents):
            param = trans.transform({param_name: trial.params[param_name]})
            parents_not_categorical_params[parent_index].append(param)

    xs = np.array(parents_not_categorical_params)
    epsilon = np.sqrt(len(param_names) + 2)
    count = 0
    while True:
        params_array = _spx_core(xs, rng, epsilon)
        _params = [transes[i].untransform(param) for i, param in enumerate(params_array)]
        params = {}
        for params in _params:
            for param_name in params.keys():
                params[param_name] = params[param_name]
        is_in_constraints = _check_in_constraints(params, search_space)
        if is_in_constraints:
            child.update(params)
            break
        if count >= max_resampling_count:
            for param_name in params.keys():
                param_distribution = search_space[param_name]
                assert isinstance(param_distribution, _NUMERICAL_DISTRIBUTIONS)
                param = np.clip(
                    params[param_name], param_distribution.low, param_distribution.high
                )
                child[param_name] = param
            break
        count += 1
    return child


def _spx_core(xs: np.ndarray, rng: np.random.RandomState, epsilon: float) -> np.ndarray:
    # https://www.smapip.is.tohoku.ac.jp/~smapip/2003/tutorial/textbook/hajime-kita.pdf
    n = xs.shape[0] - 1
    G = xs.sum(axis=0) / xs.shape[0]
    rs = [np.power(rng.uniform(0, 1), 1 / (k + 1)) for k in range(n)]
    xks = [G + epsilon * (pk - G) for pk in xs]
    ck = 0
    for k in range(1, n + 1):
        ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)
    c = xks[-1] + ck
    return c


def _check_in_constraints(params: Dict, search_space: Dict) -> bool:
    contains_flag = True
    for param_name in params.keys():
        param, param_distribution = params[param_name], search_space[param_name]
        if isinstance(param_distribution, CategoricalDistribution):
            continue
        if not param_distribution._contains(param):
            contains_flag = False
            break
    return contains_flag
