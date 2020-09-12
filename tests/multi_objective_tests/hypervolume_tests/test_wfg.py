import numpy as np
import pytest

import optuna


def test_wfg_2d() -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.asarray([[n - 1 - i, i] for i in range(n)])
        for i in range(n + 1):
            s = np.vstack((s, np.asarray([i, n - i])))
        np.random.shuffle(s)
        v = optuna.multi_objective._hypervolume.WFG().compute(s, r)
        assert v == n * n - n * (n - 1) // 2


def test_wfg_3d() -> None:
    n = 3
    r = 10 * np.ones(n)
    s = [np.hstack((np.zeros(i), [1], np.zeros(n - i - 1))) for i in range(n)]
    for _ in range(10):
        s.append(np.random.randint(1, 10, size=(n,)))
    s = np.asarray(s)
    np.random.shuffle(s)
    v = optuna.multi_objective._hypervolume.WFG().compute(s, r)
    assert v == 10 ** n - 1


def test_wfg_nd() -> None:
    for n in range(2, 10):
        r = 10 * np.ones(n)
        s = [np.hstack((np.zeros(i), [1], np.zeros(n - i - 1))) for i in range(n)]
        for _ in range(10):
            s.append(np.random.randint(1, 10, size=(n,)))
        s = np.asarray(s)
        np.random.shuffle(s)
        v = optuna.multi_objective._hypervolume.WFG().compute(s, r)
        assert v == 10 ** n - 1


def test_wfg_duplicate_points() -> None:
    n = 3
    r = 10 * np.ones(n)
    s = [np.hstack((np.zeros(i), [1], np.zeros(n - i - 1))) for i in range(n)]
    for _ in range(10):
        s.append(np.random.randint(1, 10, size=(n,)))
    s = np.asarray(s)
    v = optuna.multi_objective._hypervolume.WFG().compute(s, r)

    # Add an already existing point.
    s = np.vstack([s, s[-1]])

    np.random.shuffle(s)
    v_with_duplicate_point = optuna.multi_objective._hypervolume.WFG().compute(s, r)
    assert v == v_with_duplicate_point


def test_invalid_input() -> None:
    r = np.ones(3)
    s = np.atleast_2d(2 * np.ones(3))
    with pytest.raises(ValueError):
        _ = optuna.multi_objective._hypervolume.WFG().compute(s, r)
