from __future__ import annotations

import numpy as np
import pytest

from optuna._hypervolume.box_decomposition import _get_non_dominated_hyper_rectangle_bounds
from optuna._hypervolume.wfg import compute_hypervolume
from optuna.study._multi_objective import _is_pareto_front


def _generate_pareto_sols(n_objectives: int, n_trials: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sorted_loss_vals = np.unique(rng.random((n_trials, n_objectives)), axis=0)
    on_front = _is_pareto_front(sorted_loss_vals, assume_unique_lexsorted=True)
    return sorted_loss_vals[on_front]


@pytest.mark.parametrize("n_objectives", range(2, 5))
def test_exact_box_decomposition(n_objectives: int) -> None:
    pareto_sols = _generate_pareto_sols(n_objectives, n_trials=100, seed=42)
    ref_point = np.ones(pareto_sols.shape[-1]) * 1.1
    n_sols = pareto_sols.shape[0]
    # LOO means leave one out.
    loo_mat = ~np.eye(n_sols, dtype=bool)

    hv = compute_hypervolume(pareto_sols, ref_point, assume_pareto=True)
    correct = hv - np.array([compute_hypervolume(pareto_sols[loo], ref_point) for loo in loo_mat])
    ans = np.empty_like(correct)
    for i, loo in enumerate(loo_mat):
        lbs, ubs = _get_non_dominated_hyper_rectangle_bounds(pareto_sols[loo], ref_point)
        new_points = pareto_sols[np.newaxis, i]
        diff = np.maximum(0.0, ubs - np.maximum(new_points[..., np.newaxis, :], lbs))
        ans[i] = np.sum(np.prod(diff, axis=-1), axis=-1)

    assert np.allclose(ans, correct)
