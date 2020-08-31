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
