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


def test_compute_2d() -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        np.random.shuffle(s)
        v = optuna.multi_objective._hypervolume._compute_2d(s, r)
        assert v == n * n - n * (n - 1) // 2
