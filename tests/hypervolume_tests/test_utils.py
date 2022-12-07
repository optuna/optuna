import numpy as np
import random
from typing import Tuple
import itertools
import optuna
import pytest

def test_compute_2points_volume() -> None:
    p1 = np.ones(10)
    p2 = np.zeros(10)
    assert 1 == optuna._hypervolume._compute_2points_volume(p1, p2)
    assert 1 == optuna._hypervolume._compute_2points_volume(p2, p1)

    p1 = np.ones(10) * 2
    p2 = np.ones(10)
    assert 1 == optuna._hypervolume._compute_2points_volume(p1, p2)


def test_compute_2d() -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        np.random.shuffle(s)
        v = optuna._hypervolume._compute_2d(s, r)
        assert v == n * n - n * (n - 1) // 2


def _compute_hssp_truth_and_approx(test_case: np.ndarray, subset_size: int) -> Tuple[float, float]:
    r = 1.1 * np.max(test_case, axis=0)
    truth = 0.0
    for subset in itertools.permutations(test_case, subset_size):
        truth = max(truth, optuna._hypervolume.WFG().compute(np.asarray(subset), r))
    indices = optuna._hypervolume.utils._solve_hssp(test_case, np.arange(len(test_case)), subset_size, r)
    approx = optuna._hypervolume.WFG().compute(test_case[indices], r)
    return truth, approx


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_hssp(dim: int) -> None:
    random.seed(128)

    for i in range(8):
        subset_size = int(random.random() * i) + 1
        test_case = np.asarray([[random.random() for _ in range(dim)] for _ in range(8)])
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert approx / truth > 0.6321  # 1 - 1/e


def test_solve_hssp_infinite_loss() -> None:
    random.seed(128)

    subset_size = int(random.random() * 4) + 1
    test_case = np.asarray([[random.random() for _ in range(2)] for _ in range(8)])
    test_case = np.vstack([test_case, [float("inf") for _ in range(2)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)

    test_case = np.asarray([[random.random() for _ in range(3)] for _ in range(8)])
    test_case = np.vstack([test_case, [float("inf") for _ in range(3)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert truth == 0
    assert np.isnan(approx)

    test_case = np.asarray([[random.random() for _ in range(2)] for _ in range(8)])
    test_case = np.vstack([test_case, [-float("inf") for _ in range(2)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)

    test_case = np.asarray([[random.random() for _ in range(3)] for _ in range(8)])
    test_case = np.vstack([test_case, [-float("inf") for _ in range(3)]])
    truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
    assert np.isinf(truth)
    assert np.isinf(approx)
