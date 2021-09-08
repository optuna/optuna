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
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> Dict:

    while True:
        parents = _selection(crossover_name, study, parent_population, rng, dominates)
        child = {}
        transes = []
        distributions = []
        param_names = []
        parents_not_categorical_params: List[List[np.float64]] = [[] for _ in range(len(parents))]
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
            distributions.append(param_distribution)
            param_names.append(param_name)
            for parent_index, trial in enumerate(parents):
                param = trans.transform({param_name: trial.params[param_name]})[0]
                parents_not_categorical_params[parent_index].append(param)

        xs = np.array(parents_not_categorical_params)

        if crossover_name == "uniform":
            params_array = _uniform(xs, rng, swapping_prob)
        elif crossover_name == "blxalpha":
            alpha = 0.5
            params_array = _blxalpha(xs, rng, alpha)
        elif crossover_name == "sbx":
            if len(study.directions) == 1:
                eta = 2
            else:
                eta = 20
            params_array = _sbx(xs, rng, distributions, eta)
        elif crossover_name == "vsbx":
            if len(study.directions) == 1:
                eta = 2
            else:
                eta = 20
            params_array = _vsbx(xs, rng, eta)
        elif crossover_name == "undx":
            sigma_xi = 0.5
            sigma_eta = 0.35 / np.sqrt(len(xs[0]))
            params_array = _undx(xs, rng, sigma_xi, sigma_eta)
        elif crossover_name == "undxm":
            _m = 2
            _n = len(xs[0])
            sigma_xi = 1 / np.sqrt(_m)
            sigma_eta = (
                0.35
                * np.sqrt(_m + 1)
                * np.sqrt(3)
                / np.sqrt(_n - _m)
                / np.sqrt(_m + 2)
                / np.sqrt(2)
            )
            params_array = _undxm(xs, rng, sigma_xi, sigma_eta)
        elif crossover_name == "spx":
            epsilon = np.sqrt(len(xs[0]) + 2)
            params_array = _spx(xs, rng, epsilon)
        else:
            raise ValueError(f"{crossover_name} is not exist in optuna.")

        _params = [
            trans.untransform(np.array([param])) for trans, param in zip(transes, params_array)
        ]
        child = {}
        for param in _params:
            for param_name in param.keys():
                child[param_name] = param[param_name]

        if _check_in_constraints(child, search_space):
            break

    return child


def _selection(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> List[FrozenTrial]:
    if crossover_name in ["uniform", "blxalpha", "sbx", "vsbx"]:
        n_select = 2
    elif crossover_name in ["undx", "spx"]:
        n_select = 3
    elif crossover_name in ["undxm"]:
        n_select = 4
    else:
        raise ValueError(f"{crossover_name} is not exist in optuna.")
    if len(parent_population) < n_select:
        raise ValueError(
            f"Using {crossover_name}, \
            the population size should be greater than or equal to {n_select}."
        )

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


def _uniform(xs: np.ndarray, rng: np.random.RandomState, swapping_prob: float) -> np.ndarray:
    child = []
    x0, x1 = xs[0], xs[1]
    for x0_i, x1_i in zip(x0, x1):
        param = _swap(x0_i, x1_i, rng.rand(), swapping_prob)
        child.append(param)
    return np.array(child)


def _blxalpha(xs: np.ndarray, rng: np.random.RandomState, alpha: float) -> np.ndarray:
    x_min = xs.min(axis=0)
    x_max = xs.max(axis=0)
    diff = alpha * (x_max - x_min)
    low = x_min - diff
    high = x_max + diff
    r = rng.uniform(0, 1, size=len(diff))
    param = (high - low) * r + low
    return param


def _sbx(
    xs: np.ndarray,
    rng: np.random.RandomState,
    distributions: List[BaseDistribution],
    eta: float,
) -> np.ndarray:
    # https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1589-07.pdf

    _xl = []
    _xu = []
    for distribution in distributions:
        assert isinstance(distribution, _NUMERICAL_DISTRIBUTIONS)
        _xl.append(distribution.low)
        _xu.append(distribution.high)
    xl = np.array(_xl)
    xu = np.array(_xu)

    x_min = np.min(xs, axis=0)
    x_max = np.max(xs, axis=0)

    x_diff = np.clip(x_max - x_min, 1e-10, None)
    beta1 = 1 + 2 * (x_min - xl) / x_diff
    beta2 = 1 + 2 * (xu - x_max) / x_diff
    alpha1 = 2 - np.power(beta1, -(eta + 1))
    alpha2 = 2 - np.power(beta2, -(eta + 1))

    r = rng.uniform(0, 1, size=len(xs[0]))
    mask1 = r > 1 / alpha1
    betaq1 = np.power(r * alpha1, 1 / (eta + 1))
    betaq1[mask1] = np.power((1 / (2 - r * alpha1)), 1 / (eta + 1))[mask1]

    mask2 = r > 1 / alpha2
    betaq2 = np.power(r * alpha2, 1 / (eta + 1))
    betaq2[mask2] = np.power((1 / (2 - r * alpha2)), 1 / (eta + 1))[mask2]

    c1 = 0.5 * ((x_min + x_max) - betaq1 * x_diff)
    c2 = 0.5 * ((x_min + x_max) + betaq2 * x_diff)

    v = []
    for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, xs[0], xs[1]):
        if rng.rand() < 0.5:
            if rng.rand() < 0.5:
                v.append(c1_i)
            else:
                v.append(c2_i)
        else:
            if rng.rand() < 0.5:
                v.append(x1_i)
            else:
                v.append(x2_i)
    return np.array(v)


def _vsbx(
    xs: np.ndarray,
    rng: np.random.RandomState,
    eta: float,
) -> np.ndarray:
    r = rng.uniform(0, 1, size=len(xs[0]))
    x0, x1 = xs[0], xs[1]
    beta_1 = np.power(1 / 2 * r, 1 / (eta + 1))
    beta_2 = np.power(1 / 2 * (1 - r), 1 / (eta + 1))
    mask = r > 0.5
    c1 = 0.5 * ((1 + beta_1) * x0 + (1 - beta_1) * x1)
    c1[mask] = 0.5 * ((1 - beta_1) * x0 + (1 + beta_1) * x1)[mask]
    c2 = 0.5 * ((3 - beta_2) * x0 - (1 - beta_2) * x1)
    c2[mask] = 0.5 * (-(1 - beta_2) * x0 + (3 - beta_2) * x1)[mask]

    v = []
    for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, x0, x1):
        if rng.rand() < 0.5:
            if rng.rand() < 0.5:
                v.append(c1_i)
            else:
                v.append(c2_i)
        else:
            if rng.rand() < 0.5:
                v.append(x1_i)
            else:
                v.append(x2_i)
    return np.array(v)


def _undx(
    xs: np.ndarray, rng: np.random.RandomState, sigma_xi: float, sigma_eta: float
) -> np.ndarray:
    # https://www.jstage.jst.go.jp/article/sicetr1965/36/10/36_10_875/_pdf
    x0, x1, x2 = xs[0], xs[1], xs[2]
    n = len(x0)
    xp = (x0 + x1) / 2
    d = x0 - x1
    D = _distance_from_x_to_PSL(x0, x1, x2)
    xi = rng.normal(0, sigma_xi ** 2)
    etas = rng.normal(0, sigma_eta, size=n)
    es = _orthonormal_basis_vector_to_PSL(x0, x1)
    one = xp
    two = xi * d
    three = np.zeros(len(es[0]))
    for i in range(n - 1):
        three += etas[i] * es[i]
    three *= D
    return one + two + three


def _normalized_x1_to_x2(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    v_12 = x2 - x1
    m_12 = np.linalg.norm(v_12, ord=2)
    e_12 = v_12 / np.clip(m_12, 1e-10, None)
    return e_12


def _distance_from_x_to_PSL(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    e_12 = _normalized_x1_to_x2(x1, x2)
    v_13 = x3 - x1
    v_12_3 = v_13 - np.dot(v_13, e_12) * e_12
    m_12_3 = np.linalg.norm(v_12_3, ord=2)
    return m_12_3


def _orthonormal_basis_vector_to_PSL(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    n = len(x1)
    e_12 = _normalized_x1_to_x2(x1, x2)
    basis_matrix = np.identity(n)
    if np.count_nonzero(e_12) != 0:
        v_01 = x1 - np.zeros(len(x1))
        basis_matrix[0] = v_01 - np.dot(v_01, e_12) * e_12
    basis_matrix_t = basis_matrix.T
    Q, _ = np.linalg.qr(basis_matrix_t)
    return Q.T


def _undxm(
    xs: np.ndarray, rng: np.random.RandomState, sigma_xi: float, sigma_eta: float
) -> np.ndarray:
    # https://www.jstage.jst.go.jp/article/sicetr1965/36/10/36_10_875/_pdf

    x_mp2, xs = xs[-1], xs[:-1]  # (1), (3)
    m = len(xs) - 1
    dim = len(x_mp2)
    p = np.sum(xs, axis=0) / (m + 1)  # (2)
    ds = [x - p for x in xs]  # (2)
    n = normal(rng, ds[:-1])  # Normal to the plane that contains d_i(1,..,m) (4)
    d_mp2 = x_mp2 - p  # (4)
    D = np.dot(d_mp2, n) / np.linalg.norm(n)  # (4)
    es = _orthonormal_basis_vector_from_ds(
        ds[:-1]
    )  # orthonormal basis of the subspace orthogonal to d_i(1,..,m) (5)

    one = p

    ws = rng.normal(0, sigma_xi ** 2, size=m)
    two = np.zeros(dim)

    for i in range(m):
        two += ws[i] * ds[i]

    three = np.zeros(dim)
    for i in range(dim - m):
        vs = rng.normal(0, sigma_eta ** 2, size=dim)
        three += vs - np.dot(vs, es[i]) * es[i]
    three *= D
    return one + two + three


def normal(rng: np.random.RandomState, ds: List[np.ndarray]) -> np.ndarray:
    d = rng.normal(0, 1, size=ds[0].shape[0])
    ds.append(d)
    X = np.stack(ds)
    Q, _ = np.linalg.qr(X.T)
    return Q.T[-1]


def _orthonormal_basis_vector_from_ds(ds: List[np.ndarray]) -> np.ndarray:
    X = np.stack(ds)
    Q, _ = np.linalg.qr(X.T)
    return Q.T[-1]


def _spx(xs: np.ndarray, rng: np.random.RandomState, epsilon: float) -> np.ndarray:
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
