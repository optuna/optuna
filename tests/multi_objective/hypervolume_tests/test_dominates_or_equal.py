import numpy as np

import optuna


def test_is_dom_or_equal() -> None:
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
