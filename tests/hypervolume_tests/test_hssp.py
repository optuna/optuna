from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import optuna


def _compute_hssp_truth_and_approx(test_case: np.ndarray, subset_size: int) -> tuple[float, float]:
    r = 1.1 * np.max(test_case, axis=0)
    truth = 0.0
    for subset in itertools.permutations(test_case, subset_size):
        hv = optuna._hypervolume.compute_hypervolume(np.asarray(subset), r)
        assert not math.isnan(hv)
        truth = max(truth, hv)
    indices = optuna._hypervolume.hssp._solve_hssp(
        test_case, np.arange(len(test_case)), subset_size, r
    )
    approx = optuna._hypervolume.compute_hypervolume(test_case[indices], r)
    assert not math.isnan(approx)
    return truth, approx


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_hssp(dim: int) -> None:
    rng = np.random.RandomState(128)

    for i in range(1, 9):
        subset_size = np.random.randint(1, i + 1)
        test_case = rng.rand(8, dim)
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert approx / truth > 0.6321  # 1 - 1/e


def test_solve_hssp_infinite_loss() -> None:
    rng = np.random.RandomState(128)

    subset_size = 4
    for dim in range(2, 4):
        test_case = rng.rand(9, dim)
        test_case[-1].fill(float("inf"))
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert np.isinf(truth)
        assert np.isinf(approx)

        test_case = rng.rand(9, dim)
        test_case[-1].fill(-float("inf"))
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert np.isinf(truth)
        assert np.isinf(approx)


def test_solve_hssp_duplicated_infinite_loss() -> None:
    test_case = np.array([[np.inf, 0, 0], [np.inf, 0, 0], [0, np.inf, 0], [0, 0, np.inf]])
    r = np.full(3, np.inf)
    res = optuna._hypervolume._solve_hssp(
        rank_i_loss_vals=test_case, rank_i_indices=np.arange(4), subset_size=2, reference_point=r
    )
    assert (0 not in res) or (1 not in res)
