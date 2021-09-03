from optuna.distributions import BaseDistribution, CategoricalDistribution
from typing import Callable, Dict, List, Sequence

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.trial import FrozenTrial
from optuna.study import Study
from optuna.study import StudyDirection


def crossover(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], Sequence[float]],
    hyperparameters: Dict,
) -> Dict:
    if crossover_name == "uniform":
        parents = _selection(study, parent_population, 2, rng, dominates)
        child = _uniform_crossover(parents, rng, swapping_prob)
    elif crossover_name == "blx_alpha":
        parents = _selection(study, parent_population, 2, rng, dominates)
        child = _blx_alpha(
            parents,
            search_space,
            rng,
            swapping_prob,
            max_resampling_count,
            hyperparameters["alpha"],
        )
    elif crossover_name == "spx":
        parents = _selection(study, parent_population, hyperparameters["n_select"], rng, dominates)
        epsilon = (
            np.sqrt(len(parents[0].params) + 2)
            if hyperparameters["epsilon"] is None
            else hyperparameters["epsilon"]
        )
        child = _spx(
            parents,
            search_space,
            rng,
            swapping_prob,
            max_resampling_count,
            epsilon,
        )
    else:
        pass
    return child


def select_hyperparameters(
    crossover_name: str, population_size: int, crossover_kwargs: Dict
) -> Dict:
    hyperparameters = {}

    if crossover_name == "uniform":
        pass
    elif crossover_name == "blx_alpha":
        if "alpha" not in crossover_kwargs.keys():
            hyperparameters["alpha"] = 0.5
        else:
            if crossover_kwargs["alpha"] < 0:
                raise ValueError("`alpha` must be greater than or equal to 0.")
            hyperparameters["alpha"] = crossover_kwargs["alpha"]
    elif crossover_name == "spx":
        if "epsilon" not in crossover_kwargs.keys():
            hyperparameters["epsilon"] = None
        else:
            if crossover_kwargs["epsilon"] < 0:
                raise ValueError("`epsilon` must be greater than or equal to 0.")

        if "n_select" not in crossover_kwargs.keys():
            hyperparameters["n_select"] = 3
        else:
            if crossover_kwargs["n_select"] > population_size:
                raise ValueError("`n_select` can't be greater than `population_size`")
    else:
        raise ValueError(f"{crossover_name} is not implemented yet.")

    return hyperparameters


def _selection(
    study: Study,
    parent_population: Sequence[FrozenTrial],
    n_select: int,
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], Sequence[float]],
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
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], Sequence[float]],
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


def _swap(p0_i: np.float64, p1_i: np.float64, rand: np.float, swapping_prob: np.float):
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

        # BLX-alpha
        trans = _SearchSpaceTransform({param_name: param_distribution})
        x0_i = trans.transform({param_name: p0.params[param_name]})
        x1_i = trans.transform({param_name: p1.params[param_name]})
        count = 0
        while True:
            param = _blx_alpha_core(x0_i, x1_i, rng, alpha)
            params = trans.untransform(param)
            param = params.get(param_name, None)
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


def _spx(
    parents: List[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    max_resampling_count: int,
    epsilon: float,
) -> Dict:

    child = {}
    parents_not_categorical_params = [[] for _ in range(len(parents))]
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
    count = 0
    while True:
        _params = _spx_core(xs, rng, epsilon)
        _params = [transes[i].untransform(param) for i, param in enumerate(_params)]
        params = {}
        for param in _params:
            for param_name in param.keys():
                params[param_name] = param[param_name]
        is_in_constraints = _check_in_constraints(params, search_space)
        if is_in_constraints:
            child.update(params)
            break
        if count >= max_resampling_count:
            for param_name in param.keys():
                param = np.clip(
                    params[param_name], search_space[param_name].low, search_space[param_name].high
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
