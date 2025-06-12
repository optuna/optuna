import numpy as np
import pytest

import optuna


def _shuffle_and_filter_sols(
    sols: np.ndarray, assume_pareto: bool, rng: np.random.RandomState
) -> np.ndarray:
    rng.shuffle(sols)
    if assume_pareto:
        on_front = optuna.study._multi_objective._is_pareto_front(sols, False)
        sols = sols[on_front]
    return sols


@pytest.mark.parametrize("assume_pareto", (True, False))
@pytest.mark.parametrize("n_sols", list(range(2, 30)))
def test_wfg_2d(assume_pareto: bool, n_sols: int) -> None:
    n = n_sols
    rng = np.random.RandomState(42)
    r = n * np.ones(2)
    s = np.empty((2 * n + 1, 2), dtype=int)
    s[:n] = np.stack([np.arange(n), np.arange(n)[::-1]], axis=-1)
    s[n:] = np.stack([np.arange(n + 1), np.arange(n + 1)[::-1]], axis=-1)
    s = _shuffle_and_filter_sols(s, assume_pareto, rng)
    assert optuna._hypervolume.compute_hypervolume(s, r, assume_pareto) == n * n - n * (n - 1) // 2


@pytest.mark.parametrize("assume_pareto", (True, False))
@pytest.mark.parametrize("n_sols", list(range(2, 10)))
def test_wfg_3d(assume_pareto: bool, n_sols: int) -> None:
    n = n_sols
    rng = np.random.RandomState(42)
    r = n * np.ones(3)
    sa: list[list[int]] = []
    for x in range(n):
        for y in range(n - x):
            sa.append([x, y, n - 1 - x - y])
    s = np.array(sa, dtype=int)
    s = _shuffle_and_filter_sols(s, assume_pareto, rng)
    assert (
        optuna._hypervolume.compute_hypervolume(s, r, assume_pareto)
        == n * n * n - (n - 1) * n * (n + 1) // 6
    )


@pytest.mark.parametrize("assume_pareto", (True, False))
@pytest.mark.parametrize("n_sols", list(range(2, 10)))
def test_wfg_3d_against_hv(assume_pareto: bool, n_sols: int) -> None:
    n = n_sols
    rng = np.random.RandomState(42)
    r = n * np.ones(3)
    s = rng.randint(0, n, size=(n, 3), dtype=int)
    s = s[s[:, 0].argsort()]
    assert optuna._hypervolume.wfg._compute_3d(s, r) == optuna._hypervolume.wfg._compute_hv(s, r)


@pytest.mark.parametrize("n_objs", list(range(2, 10)))
@pytest.mark.parametrize("assume_pareto", (True, False))
def test_wfg_nd(n_objs: int, assume_pareto: bool) -> None:
    rng = np.random.RandomState(42)
    r = 10 * np.ones(n_objs)
    s = np.vstack([np.identity(n_objs), rng.randint(1, 10, size=(10, n_objs))])
    s = _shuffle_and_filter_sols(s, assume_pareto, rng)
    assert optuna._hypervolume.compute_hypervolume(s, r, assume_pareto) == 10**n_objs - 1


@pytest.mark.parametrize("n_objs", list(range(2, 10)))
def test_wfg_with_inf(n_objs: int) -> None:
    s = np.ones((1, n_objs), dtype=float)
    s[0, n_objs // 2] = np.inf
    r = 1.1 * s
    assert optuna._hypervolume.compute_hypervolume(s, r) == np.inf


@pytest.mark.parametrize("n_objs", list(range(2, 10)))
def test_wfg_with_nan(n_objs: int) -> None:
    s = np.ones((1, n_objs), dtype=float)
    s[0, n_objs // 2] = np.inf
    r = 1.1 * s
    r[-1] = np.nan
    with pytest.raises(ValueError):
        optuna._hypervolume.compute_hypervolume(s, r)


@pytest.mark.parametrize("assume_pareto", (True, False))
def test_wfg_duplicate_points(assume_pareto: bool) -> None:
    rng = np.random.RandomState(42)
    n = 3
    r = 10 * np.ones(n)
    s = np.vstack([np.identity(n), rng.randint(1, 10, size=(10, n))])
    ground_truth = optuna._hypervolume.compute_hypervolume(s, r, assume_pareto=False)
    s = np.vstack([s, s[-1]])  # Add an already existing point.
    s = _shuffle_and_filter_sols(s, assume_pareto, rng)
    assert optuna._hypervolume.compute_hypervolume(s, r, assume_pareto) == ground_truth


def test_invalid_input() -> None:
    r = np.ones(3)
    s = np.atleast_2d(2 * np.ones(3))
    with pytest.raises(ValueError):
        _ = optuna._hypervolume.compute_hypervolume(s, r)
