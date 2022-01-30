from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
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
    FloatDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
    IntDistribution,
)


def try_crossover(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
    numerical_search_space: Dict[str, BaseDistribution],
    numerical_distributions: List[BaseDistribution],
    numerical_transform: Optional[_SearchSpaceTransform],
) -> Dict[str, Any]:

    parents = _select_parents(crossover_name, study, parent_population, rng, dominates)
    child_params: Dict[str, Any] = {}

    for param_name in search_space.keys():
        # Categorical parameters always use uniform crossover.
        if isinstance(search_space[param_name], CategoricalDistribution):
            param = (
                parents[0].params[param_name]
                if rng.rand() < swapping_prob
                else parents[-1].params[param_name]
            )
            child_params[param_name] = param

    if numerical_transform is None:
        return child_params

    # The following is applied only for numerical parameters.
    parents_numerical_params_array = np.stack(
        [
            numerical_transform.transform(
                {
                    param_key: parent.params[param_key]
                    for param_key in numerical_search_space.keys()
                }
            )
            for parent in parents
        ]
    )  # Parent individual with NUMERICAL_DISTRIBUTIONS parameter.
    if crossover_name == "uniform":
        child_params_array = _uniform(
            parents_numerical_params_array[0],
            parents_numerical_params_array[1],
            rng,
            swapping_prob,
        )
    elif crossover_name == "blxalpha":
        alpha = 0.5
        child_params_array = _blxalpha(
            parents_numerical_params_array[0], parents_numerical_params_array[1], rng, alpha
        )
    elif crossover_name == "sbx":
        if len(study.directions) == 1:
            eta = 2
        else:
            eta = 20
        child_params_array = _sbx(
            parents_numerical_params_array[0],
            parents_numerical_params_array[1],
            rng,
            numerical_distributions,
            eta,
        )
    elif crossover_name == "vsbx":
        if len(study.directions) == 1:
            eta = 2
        else:
            eta = 20
        child_params_array = _vsbx(
            parents_numerical_params_array[0], parents_numerical_params_array[1], rng, eta
        )
    elif crossover_name == "undx":
        sigma_xi = 0.5
        sigma_eta = 0.35 / np.sqrt(len(parents_numerical_params_array[0]))
        child_params_array = _undx(
            parents_numerical_params_array[0],
            parents_numerical_params_array[1],
            parents_numerical_params_array[2],
            rng,
            sigma_xi,
            sigma_eta,
        )
    elif crossover_name == "spx":
        epsilon = np.sqrt(len(parents_numerical_params_array[0]) + 2)
        child_params_array = _spx(parents_numerical_params_array, rng, epsilon)
    else:
        assert False

    child_numerical_params = numerical_transform.untransform(child_params_array)
    child_params.update(child_numerical_params)
    return child_params


def crossover(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: Dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> Dict[str, Any]:

    numerical_search_space: Dict[str, BaseDistribution] = {}
    numerical_distributions: List[BaseDistribution] = []

    for key, value in search_space.items():
        if isinstance(value, _NUMERICAL_DISTRIBUTIONS):
            numerical_search_space[key] = value
            numerical_distributions.append(value)

    numerical_transform: Optional[_SearchSpaceTransform] = None
    if len(numerical_distributions) != 0:
        numerical_transform = _SearchSpaceTransform(numerical_search_space)

    while True:  # Repeat while parameters lie outside search space boundaries.
        child_params = try_crossover(
            crossover_name,
            study,
            parent_population,
            search_space,
            rng,
            swapping_prob,
            dominates,
            numerical_search_space,
            numerical_distributions,
            numerical_transform,
        )

        if _is_contained(child_params, search_space):
            break

    return child_params


def get_n_parents(crossover_name: str) -> int:
    # Select the number of parent individuals to be used for crossover.
    if crossover_name in ["uniform", "blxalpha", "sbx", "vsbx"]:
        n_parents = 2
    elif crossover_name in ["undx", "spx"]:
        n_parents = 3
    else:
        assert False
    return n_parents


def _select_parents(
    crossover_name: str,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> List[FrozenTrial]:
    n_parents = get_n_parents(crossover_name)
    parents = []
    for _ in range(n_parents):
        parent = _select_parent(
            study, [t for t in parent_population if t not in parents], rng, dominates
        )
        parents.append(parent)
    return parents


def _select_parent(
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> FrozenTrial:
    population_size = len(parent_population)
    candidate0 = parent_population[rng.choice(population_size)]
    candidate1 = parent_population[rng.choice(population_size)]

    # TODO(ohta): Consider crowding distance.
    if dominates(candidate0, candidate1, study.directions):
        return candidate0
    else:
        return candidate1


def _uniform(
    x1: np.ndarray, x2: np.ndarray, rng: np.random.RandomState, swapping_prob: float
) -> np.ndarray:
    # https://www.researchgate.net/publication/201976488_Uniform_Crossover_in_Genetic_Algorithms
    # Section 1 Introduction

    assert x1.shape == x2.shape
    assert x1.ndim == 1

    child_params_list = []
    for x1_i, x2_i in zip(x1, x2):
        param = x1_i if rng.rand() < swapping_prob else x2_i
        child_params_list.append(param)
    child_params_array = np.array(child_params_list)
    return child_params_array


def _blxalpha(
    x1: np.ndarray, x2: np.ndarray, rng: np.random.RandomState, alpha: float
) -> np.ndarray:
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.465.6900&rep=rep1&type=pdf
    # Section 2 Crossover Operators for RCGA 2.1 Blend Crossover

    assert x1.shape == x2.shape
    assert x1.ndim == 1

    xs = np.stack([x1, x2])

    x_min = xs.min(axis=0)
    x_max = xs.max(axis=0)
    diff = alpha * (x_max - x_min)  # Equation (1).
    low = x_min - diff  # Equation (1).
    high = x_max + diff  # Equation (1).
    r = rng.uniform(0, 1, size=len(diff))
    child_params_array = (high - low) * r + low
    return child_params_array


def _sbx(
    x1: np.ndarray,
    x2: np.ndarray,
    rng: np.random.RandomState,
    distributions: List[BaseDistribution],
    eta: float,
) -> np.ndarray:
    # https://www.researchgate.net/profile/M-M-Raghuwanshi/publication/267198495_Simulated_Binary_Crossover_with_Lognormal_Distribution/links/5576c78408ae7536375205d7/Simulated-Binary-Crossover-with-Lognormal-Distribution.pdf
    # Section 2 Simulated Binary Crossover (SBX)

    # To avoid generating solutions that violate the box constraints,
    # alpha1, alpha2, xls and xus are introduced, unlike the reference.
    xls_list = []
    xus_list = []
    for distribution in distributions:
        assert isinstance(distribution, _NUMERICAL_DISTRIBUTIONS)
        xls_list.append(distribution.low)
        xus_list.append(distribution.high)
    xls = np.array(xls_list)
    xus = np.array(xus_list)

    assert x1.shape == x2.shape
    assert x1.ndim == 1

    xs = np.stack([x1, x2])
    xs_min = np.min(xs, axis=0)
    xs_max = np.max(xs, axis=0)

    xs_diff = np.clip(xs_max - xs_min, 1e-10, None)
    beta1 = 1 + 2 * (xs_min - xls) / xs_diff
    beta2 = 1 + 2 * (xus - xs_max) / xs_diff
    alpha1 = 2 - np.power(beta1, -(eta + 1))
    alpha2 = 2 - np.power(beta2, -(eta + 1))

    us = rng.uniform(0, 1, size=len(xs[0]))

    mask1 = us > 1 / alpha1  # Equation (3).
    betaq1 = np.power(us * alpha1, 1 / (eta + 1))  # Equation (3).
    betaq1[mask1] = np.power((1 / (2 - us * alpha1)), 1 / (eta + 1))[mask1]  # Equation (3).

    mask2 = us > 1 / alpha2  # Equation (3).
    betaq2 = np.power(us * alpha2, 1 / (eta + 1))  # Equation (3)
    betaq2[mask2] = np.power((1 / (2 - us * alpha2)), 1 / (eta + 1))[mask2]  # Equation (3).

    c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)  # Equation (4).
    c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)  # Equation (5).

    # SBX applies crossover with establishment 0.5, and with probability 0.5,
    # the gene of the parent individual is the gene of the child individual.
    # The original SBX creates two child individuals,
    # but optuna's implementation creates only one child individual.
    # Therefore, when there is no crossover,
    # the gene is selected with equal probability from the parent individuals x1 and x2.

    child_params_list = []
    for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, xs[0], xs[1]):
        if rng.rand() < 0.5:
            if rng.rand() < 0.5:
                child_params_list.append(c1_i)
            else:
                child_params_list.append(c2_i)
        else:
            if rng.rand() < 0.5:
                child_params_list.append(x1_i)
            else:
                child_params_list.append(x2_i)
    child_params_array = np.array(child_params_list)
    return child_params_array


def _vsbx(
    x1: np.ndarray,
    x2: np.ndarray,
    rng: np.random.RandomState,
    eta: float,
) -> np.ndarray:
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.422.952&rep=rep1&type=pdf
    # Section 3.2 Crossover Schemes (vSBX)

    assert x1.shape == x2.shape
    assert x1.ndim == 1

    us = rng.uniform(0, 1, size=len(x1))
    beta_1 = np.power(1 / 2 * us, 1 / (eta + 1))
    beta_2 = np.power(1 / 2 * (1 - us), 1 / (eta + 1))
    mask = us > 0.5
    c1 = 0.5 * ((1 + beta_1) * x1 + (1 - beta_1) * x2)
    c1[mask] = 0.5 * ((1 - beta_1) * x1 + (1 + beta_1) * x2)[mask]
    c2 = 0.5 * ((3 - beta_2) * x1 - (1 - beta_2) * x2)
    c2[mask] = 0.5 * (-(1 - beta_2) * x1 + (3 - beta_2) * x2)[mask]

    # vSBX applies crossover with establishment 0.5, and with probability 0.5,
    # the gene of the parent individual is the gene of the child individual.
    # The original SBX creates two child individuals,
    # but optuna's implementation creates only one child individual.
    # Therefore, when there is no crossover,
    # the gene is selected with equal probability from the parent individuals x1 and x2.

    child_params_list = []
    for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, x1, x2):
        if rng.rand() < 0.5:
            if rng.rand() < 0.5:
                child_params_list.append(c1_i)
            else:
                child_params_list.append(c2_i)
        else:
            if rng.rand() < 0.5:
                child_params_list.append(x1_i)
            else:
                child_params_list.append(x2_i)
    child_params_array = np.array(child_params_list)
    return child_params_array


def _undx(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    rng: np.random.RandomState,
    sigma_xi: float,
    sigma_eta: float,
) -> np.ndarray:
    # https://ieeexplore.ieee.org/document/782672
    # Section 2 Unimodal Normal Distribution Crossover

    assert x1.shape == x2.shape == x3.shape
    assert x1.ndim == 1

    def _normalized_x1_to_x2(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # Compute the normalized vector from x1 to x2.

        v_12 = x2 - x1
        m_12 = np.linalg.norm(v_12, ord=2)
        e_12 = v_12 / np.clip(m_12, 1e-10, None)
        return e_12

    def _distance_from_x_to_psl(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
        # The line connecting x1 to x2 is called psl (primary search line).
        # Compute the 2-norm of the vector orthogonal to psl from x3.

        e_12 = _normalized_x1_to_x2(x1, x2)  # Normalized vector from x1 to x2.
        v_13 = x3 - x1  # Vector from x1 to x3.
        v_12_3 = v_13 - np.dot(v_13, e_12) * e_12  # Vector orthogonal to v_12 through x3.
        m_12_3 = np.linalg.norm(v_12_3, ord=2)  # 2-norm of v_12_3.
        return m_12_3

    def _orthonormal_basis_vector_to_psl(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # Compute orthogonal basis vectors for the subspace orthogonal to psl.

        n = len(x1)
        e_12 = _normalized_x1_to_x2(x1, x2)  # Normalized vector from x1 to x2.
        basis_matrix = np.identity(n)
        if np.count_nonzero(e_12) != 0:
            basis_matrix[0] = e_12
        basis_matrix_t = basis_matrix.T
        Q, _ = np.linalg.qr(basis_matrix_t)
        return Q.T[1:]

    n = len(x1)
    xp = (x1 + x2) / 2  # Section 2 (2).
    d = x1 - x2  # Section 2 (3).

    xi = rng.normal(0, sigma_xi**2)
    etas = rng.normal(0, sigma_eta**2, size=n)
    es = _orthonormal_basis_vector_to_psl(
        x1, x2
    )  # Orthonormal basis vectors of the subspace orthogonal to the psl.
    one = xp  # Section 2 (5).
    two = xi * d  # Section 2 (5).

    if n > 1:  # When n=1, there is no subsearch component.
        three = np.zeros(n)  # Section 2 (5).
        D = _distance_from_x_to_psl(x1, x2, x3)  # Section 2 (4).
        for i in range(n - 1):
            three += etas[i] * es[i]
        three *= D
        child_params_array = one + two + three
    else:
        child_params_array = one + two
    return child_params_array


def _spx(xs: np.ndarray, rng: np.random.RandomState, epsilon: float) -> np.ndarray:
    # https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover
    # Section 2 A Brief Review of SPX

    assert xs.ndim == 2
    # In this case, `len(xs)==3` because `n_select` is fixed at 3.

    n = xs.shape[0] - 1
    G = xs.sum(axis=0) / xs.shape[0]  # Equation (1).
    rs = [np.power(rng.uniform(0, 1), 1 / (k + 1)) for k in range(n)]  # Equation (2).
    xks = [G + epsilon * (pk - G) for pk in xs]  # Equation (3).
    ck = 0  # Equation (4).
    for k in range(1, n + 1):
        ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)

    child_params_array = xks[-1] + ck  # Equation (5).
    return child_params_array


def _is_contained(params: Dict[str, Any], search_space: Dict[str, BaseDistribution]) -> bool:
    for param_name in params.keys():
        param, param_distribution = params[param_name], search_space[param_name]

        if not param_distribution._contains(param_distribution.to_internal_repr(param)):
            return False
    return True
