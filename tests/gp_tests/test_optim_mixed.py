from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import optuna._gp.acqf as acqf_module
import optuna._gp.optim_mixed as optim_mixed
from optuna._gp.search_space import SearchSpace
from optuna.distributions import FloatDistribution


class _DegenerateAcquisitionFunc(acqf_module.BaseAcquisitionFunc):
    def __init__(self) -> None:
        search_space = SearchSpace({"x": FloatDistribution(0.0, 1.0)})
        super().__init__(np.ones(search_space.dim), search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 0] * 0.0

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.0] + [-np.inf] * (len(x) - 1))


def test_optimize_acqf_mixed_with_zero_sum_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_local_search_mixed_batched(
        acqf: acqf_module.BaseAcquisitionFunc,
        xs0: np.ndarray,
        *,
        tol: float = 1e-4,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        return xs0, np.zeros(len(xs0))

    monkeypatch.setattr(optim_mixed, "local_search_mixed_batched", mock_local_search_mixed_batched)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        optim_mixed.optimize_acqf_mixed(
            _DegenerateAcquisitionFunc(),
            n_preliminary_samples=4,
            n_local_search=2,
            rng=np.random.RandomState(0),
        )
