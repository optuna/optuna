import itertools
from typing import Tuple

import numpy as np
import pytest

import optuna


def _compute_hssp_truth_and_approx(test_case: np.ndarray, subset_size: int) -> Tuple[float, float]:
    r = 1.1 * np.max(test_case, axis=0)
    truth = 0.0
    for subset in itertools.permutations(test_case, subset_size):
        truth = max(truth, optuna._hypervolume.WFG().compute(np.asarray(subset), r))
    indices = optuna._hypervolume.hssp._solve_hssp(
        test_case, np.arange(len(test_case)), subset_size, r
    )
    approx = optuna._hypervolume.WFG().compute(test_case[indices], r)
    return truth, approx


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_hssp(dim: int) -> None:
    rng = np.random.RandomState(128)

    for i in range(1, 9):
        subset_size = np.random.randint(1, i + 1)
        test_case = rng.rand(8, dim)
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert approx / truth > 0.6321  # 1 - 1/e


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solve_hssp_infinite_loss() -> None:
    rng = np.random.RandomState(128)

    subset_size = 4
    test_case = rng.rand(9, 2)
    test_case[-1].fill(float("inf"))
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)

    test_case = rng.rand(9, 3)
    test_case[-1].fill(float("inf"))
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert truth == 0
    assert np.isnan(approx)

    for dim in range(2, 4):
        test_case = rng.rand(9, dim)
        test_case[-1].fill(-float("inf"))
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert np.isinf(truth)
        assert np.isinf(approx)
