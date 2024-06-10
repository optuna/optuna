import numpy as np
import pytest

import optuna


def _shuffle_and_filter_sols(sols: np.ndarray, assume_pareto: bool) -> np.ndarray:
    np.random.shuffle(sols)
    if assume_pareto:
        on_front = optuna.study._multi_objective._is_pareto_front(sols, False)
        sols = sols[on_front]
    return sols


@pytest.mark.parametrize("assume_pareto", (True, False))
def test_wfg_2d(assume_pareto: bool) -> None:
    for n in range(2, 30):
        r = n * np.ones(2)
        s = np.empty((2 * n + 1, 2), dtype=int)
        s[:n] = np.stack([np.arange(n), np.arange(n)[::-1]], axis=-1)
        s[n:] = np.stack([np.arange(n + 1), np.arange(n + 1)[::-1]], axis=-1)
        s = _shuffle_and_filter_sols(s, assume_pareto)
        assert optuna._hypervolume.WFG().compute(s, r, assume_pareto) == n * n - n * (n - 1) // 2


@pytest.mark.parametrize("n_objs", list(range(2, 10)))
@pytest.mark.parametrize("assume_pareto", (True, False))
def test_wfg_nd(n_objs: int, assume_pareto: bool) -> None:
    r = 10 * np.ones(n_objs)
    s = np.vstack([np.identity(n_objs), np.random.randint(1, 10, size=(10, n_objs))])
    s = _shuffle_and_filter_sols(s, assume_pareto)
    assert optuna._hypervolume.WFG().compute(s, r, assume_pareto) == 10**n_objs - 1


@pytest.mark.parametrize("assume_pareto", (True, False))
def test_wfg_duplicate_points(assume_pareto: bool) -> None:
    n = 3
    r = 10 * np.ones(n)
    s = np.vstack([np.identity(n), np.random.randint(1, 10, size=(10, n))])
    ground_truth = optuna._hypervolume.WFG().compute(s, r, assume_pareto=False)
    s = np.vstack([s, s[-1]])  # Add an already existing point.
    s = _shuffle_and_filter_sols(s, assume_pareto)
    assert optuna._hypervolume.WFG().compute(s, r, assume_pareto) == ground_truth


def test_invalid_input() -> None:
    r = np.ones(3)
    s = np.atleast_2d(2 * np.ones(3))
    with pytest.raises(ValueError):
        _ = optuna._hypervolume.WFG().compute(s, r)
