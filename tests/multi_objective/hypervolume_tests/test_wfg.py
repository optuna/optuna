import numpy as np

import optuna


def test_cmp_points() -> None:
    wfg = optuna.multi_objective.hypervolume.WFG()
    wfg._slice = 2

    point1 = np.asarray([1, 0])
    point2 = np.asarray([0, 1])
    assert not wfg._comp_points(point1, point2)
    assert wfg._comp_points(point2, point1)

    point1 = np.asarray([0, 1, 1])
    point2 = np.asarray([0, 0, 1])
    assert wfg._comp_points(point1, point2)

    point1 = np.asarray([1, 1, 1])
    point2 = np.asarray([1, 1, 0])
    assert not wfg._comp_points(point1, point2)


def test_wfg_2d() -> None:
    for n in range(2, 20):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        v = optuna.multi_objective.hypervolume.WFG().compute(s, r)
        assert v == n * n - n * (n - 1) // 2


def test_wfg_nd() -> None:
    for n in range(2, 20):
        r = 2 * np.ones(n)
        s = np.asarray([np.hstack((np.zeros(i), [1], np.zeros(n - i - 1))) for i in range(n)])
        v = optuna.multi_objective.hypervolume.WFG().compute(s, r)
        print(r, '\n', s)
        assert v == 2 ** n - 1
