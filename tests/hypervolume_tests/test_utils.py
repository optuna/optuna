import numpy as np

import optuna


def test_compute_2d() -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        np.random.shuffle(s)
        v = optuna._hypervolume._compute_2d(s, r)
        assert v == n * n - n * (n - 1) // 2
