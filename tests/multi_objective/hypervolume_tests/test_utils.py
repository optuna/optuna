import numpy as np

import optuna


def test_compute_2points_volume() -> None:
    p1 = np.ones(10)
    p2 = np.zeros(10)
    assert 1 == optuna.multi_objective._hypervolume._compute_2points_volume(p1, p2)
    assert 1 == optuna.multi_objective._hypervolume._compute_2points_volume(p2, p1)

    p1 = np.ones(10) * 2
    p2 = np.ones(10)
    assert 1 == optuna.multi_objective._hypervolume._compute_2points_volume(p1, p2)


def test_dominates_or_equal() -> None:
    p1 = np.asarray([1, 1, 1])
    p2 = np.asarray([1, 0, 0])
    assert optuna.multi_objective._hypervolume._dominates_or_equal(p1, p2)

    p1 = np.asarray([1, 0, 0])
    p2 = np.asarray([1, 0, 1])
    assert not optuna.multi_objective._hypervolume._dominates_or_equal(p1, p2)

    p1 = np.asarray([1, 1, 1])
    p2 = np.asarray([1, 1, 1])
    assert optuna.multi_objective._hypervolume._dominates_or_equal(p1, p2)

    p1 = np.asarray([1, 0, 1])
    p2 = np.asarray([1, 1, 0])
    assert not optuna.multi_objective._hypervolume._dominates_or_equal(p1, p2)
