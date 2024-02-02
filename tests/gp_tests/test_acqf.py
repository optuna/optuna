from optuna._gp.gp import KernelParamsTensor, kernel, posterior
from optuna._gp.acqf import AcquisitionFunctionType, eval_acqf, AcquisitionFunctionParams, create_acqf_params
from optuna._gp.search_space import SearchSpace, ScaleType
import numpy as np
import pytest

from botorch.models.model import Model
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.means import ZeroMean
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from typing import Callable, Any
import torch

@pytest.mark.parametrize(
    "acqf_type, beta, botorch_acqf_gen",
    [
        (AcquisitionFunctionType.LOG_EI, None, lambda model, acqf_params: LogExpectedImprovement(model, best_f=acqf_params.max_Y)),
        (AcquisitionFunctionType.UCB, 2.0, lambda model, acqf_params: UpperConfidenceBound(model, beta=acqf_params.beta))
    ]
)
@pytest.mark.parametrize("x",[
    np.array([0.15, 0.12]),  # unbatched
    np.array([[0.15, 0.12], [0.0, 1.0]])  # batched
])
def test_posterior_and_eval_acqf(acqf_type: AcquisitionFunctionType, beta: float | None, botorch_acqf_gen: Callable[[Model, AcquisitionFunctionParams], Any], x: np.ndarray) -> None:

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
        steps=np.zeros(n_dims)
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

    prior_cov_fX_fX = kernel(torch.zeros(n_dims, dtype=torch.bool), kernel_params, torch.from_numpy(X), torch.from_numpy(X))
    posterior_mean_fx, posterior_var_fx = posterior(kernel_params, torch.from_numpy(X), torch.zeros(n_dims, dtype=torch.bool), acqf_params.cov_Y_Y_inv, acqf_params.cov_Y_Y_inv_Y, torch.from_numpy(x))

    acqf_value = eval_acqf(acqf_params, x_tensor)
    acqf_value.sum().backward()
    acqf_grad = x_tensor.grad
    assert acqf_grad is not None


    gpytorch_likelihood = GaussianLikelihood()
    gpytorch_likelihood.noise_covar.noise = kernel_params.noise_var
    matern_kernel = MaternKernel(nu=2.5, ard_num_dims=n_dims)
    matern_kernel.lengthscale = kernel_params.inverse_squared_lengthscales.rsqrt()
    covar_module = ScaleKernel(matern_kernel)
    covar_module.outputscale = kernel_params.kernel_scale

    botorch_model = SingleTaskGP(
        train_X = torch.from_numpy(X),
        train_Y = torch.from_numpy(Y)[:, None],
        likelihood=gpytorch_likelihood,
        covar_module=covar_module,
        mean_module=ZeroMean(),
    )
    botorch_prior_fX = botorch_model(torch.from_numpy(X))
    assert torch.allclose(botorch_prior_fX.covariance_matrix, prior_cov_fX_fX)

    botorch_model.eval()

    botorch_acqf = botorch_acqf_gen(botorch_model, acqf_params)

    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)
    botorch_posterior_fx = botorch_model.posterior(x_tensor[..., None, :])
    assert torch.allclose(posterior_mean_fx, botorch_posterior_fx.mean[..., 0, 0])
    assert torch.allclose(posterior_var_fx, botorch_posterior_fx.variance[..., 0, 0])

    botorch_acqf_value = botorch_acqf(x_tensor[..., None, :])
    botorch_acqf_value.sum().backward()
    botorch_acqf_grad = x_tensor.grad
    assert botorch_acqf_grad is not None
    assert torch.allclose(acqf_value, botorch_acqf_value)
    assert torch.allclose(acqf_grad, botorch_acqf_grad)




