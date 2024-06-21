import numpy as np

import optuna


def test_compute_2d() -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        np.random.shuffle(s)
        on_front = optuna.study._multi_objective._is_pareto_front(s, assume_unique_lexsorted=False)
        pareto_sols = s[on_front]
        v = optuna._hypervolume._compute_2d(pareto_sols[np.argsort(pareto_sols[:, 0])], r)
        assert v == n * n - n * (n - 1) // 2
