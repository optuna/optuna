from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp.acqf import AcquisitionFunctionParams
from optuna._gp.acqf import AcquisitionFunctionType
from optuna._gp.acqf import ConstrainedAcquisitionFunctionParams
from optuna._gp.acqf import create_acqf_params
from optuna._gp.acqf import eval_acqf
from optuna._gp.acqf import MultiObjectiveAcquisitionFunctionParams
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


def verify_eval_acqf(x: np.ndarray, acqf_params: AcquisitionFunctionParams) -> None:
    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)
    acqf_value = eval_acqf(acqf_params, x_tensor)
    acqf_value.sum().backward()  # type: ignore
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None
    assert acqf_value.shape == x.shape[:-1]
    assert torch.all(torch.isfinite(acqf_value))
    assert torch.all(torch.isfinite(acqf_grad))


@pytest.fixture
def X() -> np.ndarray:
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]])


@pytest.fixture
def kernel_params() -> KernelParamsTensor:
    return KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )


@pytest.fixture
def search_space() -> SearchSpace:
    n_dims = 2
    return SearchSpace(
        scale_types=np.full(n_dims, ScaleType.LINEAR),
        bounds=np.array([[0.0, 1.0] * n_dims]),
        steps=np.zeros(n_dims),
    )


parametrized_x = pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
)

parametrized_additional_values = pytest.mark.parametrize(
    "additional_values",
    [
        np.array([[0.2], [0.3], [-0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, -0.4], [-0.2, -0.3, -0.4]]),
        np.array([[-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4]]),
    ],
)


@pytest.mark.parametrize(
    "acqf_type, beta",
    [
        (AcquisitionFunctionType.LOG_EI, None),
        (AcquisitionFunctionType.UCB, 2.0),
        (AcquisitionFunctionType.LCB, 2.0),
        (AcquisitionFunctionType.LOG_PI, None),
    ],
)
@parametrized_x
def test_eval_acqf(
    acqf_type: AcquisitionFunctionType,
    beta: float | None,
    x: np.ndarray,
    kernel_params: KernelParamsTensor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    Y = np.array([1.0, 2.0, 3.0])
    acqf_params = create_acqf_params(
        acqf_type=acqf_type,
        kernel_params=kernel_params,
        search_space=search_space,
        X=X,
        Y=Y,
        beta=beta,
        acqf_stabilizing_noise=0.0,
    )
    verify_eval_acqf(x, acqf_params)


@parametrized_x
@parametrized_additional_values
def test_eval_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    kernel_params: KernelParamsTensor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    c = additional_values.copy()
    Y = np.array([1.0, 2.0, 3.0])
    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    acqf_params = create_acqf_params(
        acqf_type=AcquisitionFunctionType.LOG_EI,
        kernel_params=kernel_params,
        search_space=search_space,
        X=X,
        Y=Y,
        max_Y=-np.inf if is_all_infeasible else np.max(Y[is_feasible]),
        acqf_stabilizing_noise=0.0,
    )
    constraints_acqf_params = [
        create_acqf_params(
            acqf_type=AcquisitionFunctionType.LOG_PI,
            kernel_params=kernel_params,
            search_space=search_space,
            X=X,
            Y=vals,
            acqf_stabilizing_noise=0.0,
            max_Y=0.0,
        )
        for vals in c.T
    ]
    acqf_params_with_constraints = ConstrainedAcquisitionFunctionParams.from_acqf_params(
        acqf_params, constraints_acqf_params
    )
    verify_eval_acqf(x, acqf_params_with_constraints)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf(
    x: np.ndarray,
    additional_values: np.ndarray,
    kernel_params: KernelParamsTensor,
    search_space: SearchSpace,
    X: np.ndarray,
) -> None:
    Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
    n_objectives = Y.shape[-1]
    acqf_params_for_objectives = []
    for i in range(n_objectives):
        acqf_params_for_objectives.append(
            create_acqf_params(
                AcquisitionFunctionType.LOG_EHVI,
                kernel_params=kernel_params,
                search_space=search_space,
                X=X,
                Y=Y[:, i],
                acqf_stabilizing_noise=0.0,
            )
        )

    acqf_params = MultiObjectiveAcquisitionFunctionParams.from_acqf_params(
        acqf_params_for_objectives=acqf_params_for_objectives,
        Y=Y,
        n_qmc_samples=32,
        qmc_seed=42,
    )
    verify_eval_acqf(x, acqf_params)
