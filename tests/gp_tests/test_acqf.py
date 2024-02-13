from __future__ import annotations

import sys

import numpy as np
import pytest


# TODO(contramundum53): Remove this block after torch supports Python 3.12.
if sys.version_info >= (3, 12):
    pytest.skip("PyTorch does not support python 3.12.", allow_module_level=True)

import torch

from optuna._gp.acqf import AcquisitionFunctionType
from optuna._gp.acqf import create_acqf_params
from optuna._gp.acqf import eval_acqf
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


@pytest.mark.parametrize(
    "acqf_type, beta",
    [
        (AcquisitionFunctionType.LOG_EI, None),
        (AcquisitionFunctionType.UCB, 2.0),
        (AcquisitionFunctionType.LCB, 2.0),
    ],
)
@pytest.mark.parametrize(
    "x", [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])]  # unbatched  # batched
)
def test_eval_acqf(
    acqf_type: AcquisitionFunctionType,
    beta: float | None,
    x: np.ndarray,
) -> None:
    n_dims = 2
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]])
    Y = np.array([1.0, 2.0, 3.0])
    kernel_params = KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(4.0, dtype=torch.float64),
        noise_var=torch.tensor(0.1, dtype=torch.float64),
    )
    search_space = SearchSpace(
        scale_types=np.full(n_dims, ScaleType.LINEAR),
        bounds=np.array([[0.0, 1.0] * n_dims]),
        steps=np.zeros(n_dims),
    )

    acqf_params = create_acqf_params(
        acqf_type=acqf_type,
        kernel_params=kernel_params,
        search_space=search_space,
        X=X,
        Y=Y,
        beta=beta,
        acqf_stabilizing_noise=0.0,
    )

    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)

    acqf_value = eval_acqf(acqf_params, x_tensor)
    acqf_value.sum().backward()  # type: ignore
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None

    assert acqf_value.shape == x.shape[:-1]

    assert torch.all(torch.isfinite(acqf_value))
    assert torch.all(torch.isfinite(acqf_grad))
