import numpy as np

import optuna


def test_is_dom_or_equal() -> None:
    p1 = np.ones(10)
    p2 = np.zeros(10)
    assert 1 == optuna.multi_objective._hypervolume._compute_2points_volume(p1, p2)
