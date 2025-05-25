from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest

from optuna._hypervolume.box_decomposition import get_non_dominated_box_bounds
from optuna._hypervolume.wfg import compute_hypervolume
from optuna.study._multi_objective import _is_pareto_front


_EPS = 1e-12


class InstanceGenerator(Protocol):
    def __call__(self, n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
        raise NotImplementedError


def _extract_pareto_sols(loss_values: np.ndarray) -> np.ndarray:
    sorted_loss_vals = np.unique(loss_values, axis=0)
    on_front = _is_pareto_front(sorted_loss_vals, assume_unique_lexsorted=True)
    return sorted_loss_vals[on_front]


def _generate_uniform_samples(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.maximum(_EPS, rng.random((n_trials, n_objectives)))


def _generate_instances_with_negative(n_objectives: int, n_trials: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return _extract_pareto_sols(rng.normal(size=(n_trials, n_objectives)))


def _generate_concave_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = points / np.maximum(_EPS, np.sum(points**2, axis=-1))[:, np.newaxis]
    return _extract_pareto_sols(loss_values)


def _generate_convex_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = 1 - points / np.maximum(_EPS, np.sum(points**2, axis=-1))[:, np.newaxis]
    return _extract_pareto_sols(loss_values)


def _generate_linear_instances(n_trials: int, n_objectives: int, seed: int) -> np.ndarray:
    """See Section 4.2 of https://arxiv.org/pdf/1510.01963"""
    points = _generate_uniform_samples(n_trials, n_objectives, seed)
    loss_values = points / np.maximum(_EPS, np.sum(points, axis=-1))[:, np.newaxis]
    return _extract_pareto_sols(loss_values)


def _calculate_hypervolume_improvement(
    lbs: np.ndarray, ubs: np.ndarray, new_points: np.ndarray
) -> np.ndarray:
    diff = np.maximum(0.0, ubs - np.maximum(new_points[..., np.newaxis, :], lbs))
    # The minimization version of Eq. (1) in https://arxiv.org/pdf/2006.05078.
    return np.sum(np.prod(diff, axis=-1), axis=-1)


def _verify_exact_hypervolume_improvement(pareto_sols: np.ndarray) -> None:
    ref_point = np.max(pareto_sols, axis=0)
    ref_point = np.maximum(ref_point * 1.1, ref_point * 0.9)
    hv = compute_hypervolume(pareto_sols, ref_point, assume_pareto=True)
    for i in range(pareto_sols.shape[0]):
        # LOO means leave one out.
        loo = np.arange(pareto_sols.shape[0]) != i
        correct = hv - compute_hypervolume(pareto_sols[loo], ref_point)
        lbs, ubs = get_non_dominated_box_bounds(pareto_sols[loo], ref_point)
        out = _calculate_hypervolume_improvement(lbs, ubs, pareto_sols[np.newaxis, i])
        assert out.shape == (1,)
        assert np.isclose(out[0], correct)


@pytest.mark.parametrize(
    "gen",
    (
        _generate_concave_instances,
        _generate_convex_instances,
        _generate_instances_with_negative,
        _generate_linear_instances,
    ),
)
@pytest.mark.parametrize("n_objectives", [2, 3, 4])
def test_exact_box_decomposition(gen: InstanceGenerator, n_objectives: int) -> None:
    n_trials = 30 if n_objectives == 4 else 60
    pareto_sols = gen(n_objectives=n_objectives, n_trials=n_trials, seed=42)
    _verify_exact_hypervolume_improvement(pareto_sols)


@pytest.mark.parametrize(
    "gen",
    (
        _generate_concave_instances,
        _generate_convex_instances,
        _generate_instances_with_negative,
        _generate_linear_instances,
    ),
)
@pytest.mark.parametrize("n_objectives", [2, 3, 4])
def test_box_decomposition_with_non_general_position(
    gen: InstanceGenerator, n_objectives: int
) -> None:
    # By using integer values, duplications can be guaranteed.
    # We are testing Proposition 2.2 in the paper: https://arxiv.org/abs/1510.01963
    # General position means that no two distinct points share the same value in any dimensions.
    pareto_sols = _extract_pareto_sols(
        np.round(gen(n_objectives=n_objectives, n_trials=100, seed=42) * 5)
    )
    _verify_exact_hypervolume_improvement(pareto_sols)


@pytest.mark.parametrize("n_objectives", [2, 3, 4])
def test_box_decomposition_with_1_solution(n_objectives: int) -> None:
    pareto_sols = np.ones((1, n_objectives))
    ref_point = np.ones(n_objectives) * 10.0
    lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
    hvi = _calculate_hypervolume_improvement(lbs, ubs, np.zeros((1, n_objectives)))[0]
    assert np.isclose(hvi, 10**n_objectives - 9**n_objectives)


@pytest.mark.parametrize("n_objectives", [2, 3, 4])
def test_box_decomposition_with_empty_set(n_objectives: int) -> None:
    pareto_sols = np.ones((0, n_objectives))
    ref_point = np.ones(n_objectives) * 10.0
    lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
    hvi = _calculate_hypervolume_improvement(lbs, ubs, np.zeros((1, n_objectives)))[0]
    assert np.isclose(hvi, 10**n_objectives)


@pytest.mark.parametrize(
    "values", [np.array([[np.inf, 0.0]]), np.array([[-np.inf, 0.0]]), np.array([[np.nan, 0.0]])]
)
def test_assertion_for_non_finite_values(values: np.ndarray) -> None:
    # NOTE(nabenabe): Since GPSampler trains regression models, values must be clipped, which is
    # done by `optuna._gp.gp.warn_and_convert_inf`. If `values` includes any non-finite values,
    # hypervolume improvement becomes either `inf` or `nan` anyways.
    with pytest.raises(AssertionError):
        get_non_dominated_box_bounds(values, np.ones(values.shape[-1]) * 10.0)
