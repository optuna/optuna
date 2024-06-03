import itertools
import math
from typing import Tuple

import numpy as np
import pytest

import optuna


def _compute_hssp_truth_and_approx(test_case: np.ndarray, subset_size: int) -> Tuple[float, float]:
    r = 1.1 * np.max(test_case, axis=0)
    truth = 0.0
    for subset in itertools.permutations(test_case, subset_size):
        hv = optuna._hypervolume.WFG().compute(np.asarray(subset), r)
        assert not math.isnan(hv)
        truth = max(truth, hv)
    indices = optuna._hypervolume.hssp._solve_hssp(
        test_case, np.arange(len(test_case)), subset_size, r
    )
    approx = optuna._hypervolume.WFG().compute(test_case[indices], r)
    assert not math.isnan(approx)
    return truth, approx


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_hssp(dim: int) -> None:
    rng = np.random.RandomState(128)

    for i in range(1, 9):
        subset_size = np.random.randint(1, i + 1)
        test_case = rng.rand(8, dim)
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert approx / truth > 0.6321  # 1 - 1/e


def test_solve_hssp_infinite_loss() -> None:
    rng = np.random.RandomState(128)

    subset_size = 4
    for dim in range(2, 4):
        test_case = rng.rand(9, dim)
        test_case[-1].fill(float("inf"))
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert np.isinf(truth)
        assert np.isinf(approx)

        test_case = rng.rand(9, dim)
        test_case[-1].fill(-float("inf"))
        truth, approx = _compute_hssp_truth_and_approx(test_case, subset_size)
        assert np.isinf(truth)
        assert np.isinf(approx)


def test_solve_hssp_duplicated_infinite_loss() -> None:
    test_case = np.array([[np.inf, 0, 0], [np.inf, 0, 0], [0, np.inf, 0], [0, 0, np.inf]])
    r = np.full(3, np.inf)
    res = optuna._hypervolume._solve_hssp(
        rank_i_loss_vals=test_case, rank_i_indices=np.arange(4), subset_size=2, reference_point=r
    )
    assert (0 not in res) or (1 not in res)


def _solve_hssp_and_check_cache(
    study: optuna.Study,
    pareto_sols: np.ndarray,
    pareto_indices: np.ndarray,
    subset_size: int, 
    ref_point: np.ndarray,
) -> np.ndarray:
    selected_indices = optuna._hypervolume._solve_hssp_with_cache(
        study, pareto_sols, pareto_indices, subset_size=subset_size, reference_point=ref_point
    )
    hssp_cache = study._storage.get_study_system_attrs(study._study_id)["hssp_cache"]
    assert np.allclose(hssp_cache["rank_i_loss_vals"], pareto_sols)
    assert np.array_equal(hssp_cache["rank_i_indices"], pareto_indices)
    assert hssp_cache["subset_size"] == subset_size
    assert np.allclose(hssp_cache["reference_point"], ref_point)
    return selected_indices


@pytest.mark.parametrize("storage", (optuna.storages.RDBStorage("sqlite:///:memory:"), optuna.storages.InMemoryStorage()))
def test_solve_hssp_with_cache(storage: optuna.storages.BaseStorage) -> None:
    study = optuna.create_study(directions=["minimize"]*2, storage=storage)
    n_trials = 100
    n_objectives = 3
    rng = np.random.RandomState(42)
    loss_vals = rng.random((n_trials, n_objectives))
    indices = np.arange(n_trials)
    on_front = optuna.study._multi_objective._is_pareto_front(
        loss_vals, assume_unique_lexsorted=False
    )
    pareto_sols = loss_vals[on_front]
    pareto_indices = indices[on_front]
    ref_point = np.ones(n_objectives, dtype=float)
    is_rank2 = optuna.study._multi_objective._is_pareto_front(
        loss_vals[~on_front], assume_unique_lexsorted=False
    )
    rank_2_loss_vals = loss_vals[~on_front][is_rank2]
    rank_2_indices = indices[~on_front][is_rank2]
    subset_size = min(rank_2_indices.size, pareto_indices.size) // 2
    for i in range(2):
        selected_indices_for_pareto = _solve_hssp_and_check_cache(
            study, pareto_sols, pareto_indices, subset_size, ref_point
        )

    selected_indices_for_rank_2 = _solve_hssp_and_check_cache(
        study, rank_2_loss_vals, rank_2_indices, subset_size, ref_point
    )
    # Cache should be deleted, meaning that the following condition must be satisfied.
    assert np.all(np.sort(selected_indices_for_pareto) != np.sort(selected_indices_for_rank_2))
