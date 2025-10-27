"""Notations in this Gaussian process implementation

X_train: Observed parameter values with the shape of (len(trials), len(params)).
y_train: Observed objective values with the shape of (len(trials), ).
x: (Possibly batched) parameter value(s) to evaluate with the shape of (..., len(params)).
cov_fX_fX: Kernel matrix X = V[f(X)] with the shape of (len(trials), len(trials)).
cov_fx_fX: Kernel matrix Cov[f(x), f(X)] with the shape of (..., len(trials)).
cov_fx_fx: Kernel scalar value x = V[f(x)]. This value is constant for the Matern 5/2 kernel.
cov_Y_Y_inv:
    The inverse of the covariance matrix (V[f(X) + noise_var])^-1 with the shape of
    (len(trials), len(trials)).
cov_Y_Y_inv_Y: `cov_Y_Y_inv @ y` with the shape of (len(trials), ).
max_Y: The maximum of Y (Note that we transform the objective values such that it is maximized.)
sqd: The squared differences of each dimension between two points.
is_categorical:
    A boolean array with the shape of (len(params), ). If is_categorical[i] is True, the i-th
    parameter is categorical.
"""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy
    import torch
else:
    from optuna._imports import _LazyImport

    scipy = _LazyImport("scipy")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


def warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    warnings.warn("Clip non-finite values to the min/max finite values for GP fittings.")
    is_any_finite = np.any(is_values_finite, axis=0)
    # NOTE(nabenabe): values cannot include nan to apply np.clip properly, but Optuna anyways won't
    # pass nan in values by design.
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        """
        This method calculates `exp(-sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)` where
        `sqrt5d = sqrt(5 * squared_distance)`.

        Please note that automatic differentiation by PyTorch does not work well at
        `squared_distance = 0` due to zero division, so we manually save the derivative, i.e.,
        `-5/6 * (1 + sqrt5d) * exp(-sqrt5d)`, for the exact derivative calculation.

        Notice that the derivative of this function is taken w.r.t. d**2, but not w.r.t. d.
        """
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        """
        Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function, then
        deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        """
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class GPRegressor:
    def __init__(
        self,
        is_categorical: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        inverse_squared_lengthscales: torch.Tensor,  # (len(params), )
        kernel_scale: torch.Tensor,  # Scalar
        noise_var: torch.Tensor,  # Scalar
    ) -> None:
        self._is_categorical = is_categorical
        self._X_train = X_train
        self._y_train = y_train
        self._squared_X_diff = (X_train.unsqueeze(-2) - X_train.unsqueeze(-3)).square_()
        if self._is_categorical.any():
            self._squared_X_diff[..., self._is_categorical] = (
                self._squared_X_diff[..., self._is_categorical] > 0.0
            ).type(torch.float64)
        self._cov_Y_Y_chol: torch.Tensor | None = None
        self._cov_Y_Y_inv_Y: torch.Tensor | None = None
        # TODO(nabenabe): Rename the attributes to private with `_`.
        self.inverse_squared_lengthscales = inverse_squared_lengthscales
        self.kernel_scale = kernel_scale
        self.noise_var = noise_var

    @property
    def length_scales(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.inverse_squared_lengthscales.detach().numpy())

    def _cache_matrix(self) -> None:
        assert (
            self._cov_Y_Y_chol is None and self._cov_Y_Y_inv_Y is None
        ), "Cannot call cache_matrix more than once."
        with torch.no_grad():
            cov_Y_Y = self.kernel().detach().numpy()

        cov_Y_Y[np.diag_indices(self._X_train.shape[0])] += self.noise_var.item()
        cov_Y_Y_chol = np.linalg.cholesky(cov_Y_Y)
        # cov_Y_Y_inv @ y = v --> y = cov_Y_Y @ v --> y = cov_Y_Y_chol @ cov_Y_Y_chol.T @ v
        # NOTE(nabenabe): Don't use np.linalg.inv because it is too slow und unstable.
        # cf. https://github.com/optuna/optuna/issues/6230
        cov_Y_Y_inv_Y = scipy.linalg.solve_triangular(
            cov_Y_Y_chol.T,
            scipy.linalg.solve_triangular(cov_Y_Y_chol, self._y_train.numpy(), lower=True),
            lower=False,
        )
        # NOTE(nabenabe): Here we use NumPy to guarantee the reproducibility from the past.
        self._cov_Y_Y_chol = torch.from_numpy(cov_Y_Y_chol)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv_Y)
        self.inverse_squared_lengthscales = self.inverse_squared_lengthscales.detach()
        self.inverse_squared_lengthscales.grad = None
        self.kernel_scale = self.kernel_scale.detach()
        self.kernel_scale.grad = None
        self.noise_var = self.noise_var.detach()
        self.noise_var.grad = None

    def kernel(
        self, X1: torch.Tensor | None = None, X2: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Return the kernel matrix with the shape of (..., n_A, n_B) given X1 and X2 each with the
        shapes of (..., n_A, len(params)) and (..., n_B, len(params)).

        If x1 and x2 have the shape of (len(params), ), kernel(x1, x2) is computed as:
            kernel_scale * Matern52Kernel.apply(
                sqd(x1, x2) @ inverse_squared_lengthscales
            )
        where if x1[i] is continuous, sqd(x1, x2)[i] = (x1[i] - x2[i]) ** 2 and if x1[i] is
        categorical, sqd(x1, x2)[i] = int(x1[i] != x2[i]).
        Note that the distance for categorical parameters is the Hamming distance.
        """
        if X1 is None:
            assert X2 is None
            sqd = self._squared_X_diff
        else:
            if X2 is None:
                X2 = self._X_train

            sqd = (X1 - X2 if X1.ndim == 1 else X1.unsqueeze(-2) - X2.unsqueeze(-3)).square_()
            if self._is_categorical.any():
                sqd[..., self._is_categorical] = (sqd[..., self._is_categorical] > 0.0).type(
                    torch.float64
                )
        sqdist = sqd.matmul(self.inverse_squared_lengthscales)
        return Matern52Kernel.apply(sqdist) * self.kernel_scale  # type: ignore

    def posterior(self, x: torch.Tensor, joint: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes the posterior mean and variance given the points `x` where both mean
        and variance tensors will have the shape of x.shape[:-1].
        If ``joint=True``, the joint posterior will be computed.

        The posterior mean and variance are computed as:
            mean = cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ y, and
            var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ cov_fx_fX.T.

        Please note that we clamp the variance to avoid negative values due to numerical errors.
        """
        assert (
            self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None
        ), "Call cache_matrix before calling posterior."
        is_single_point = x.ndim == 1
        x_ = x if not is_single_point else x.unsqueeze(0)
        mean = torch.linalg.vecdot(cov_fx_fX := self.kernel(x_), self._cov_Y_Y_inv_Y)
        # K @ inv(C) = V --> K = V @ C --> K = V @ L @ L.T
        V = torch.linalg.solve_triangular(
            self._cov_Y_Y_chol,
            torch.linalg.solve_triangular(self._cov_Y_Y_chol.T, cov_fx_fX, upper=True, left=False),
            upper=False,
            left=False,
        )
        if joint:
            assert not is_single_point, "Call posterior with joint=False for a single point."
            cov_fx_fx = self.kernel(x_, x_)
            # NOTE(nabenabe): Indeed, var_ here is a covariance matrix.
            var_ = cov_fx_fx - V.matmul(cov_fx_fX.transpose(-1, -2))
            var_.diagonal(dim1=-2, dim2=-1).clamp_min_(0.0)
        else:
            cov_fx_fx = self.kernel_scale  # kernel(x, x) = kernel_scale
            var_ = cov_fx_fx - torch.linalg.vecdot(cov_fx_fX, V)
            var_.clamp_min_(0.0)
        return (mean.squeeze(0), var_.squeeze(0)) if is_single_point else (mean, var_)

    def marginal_log_likelihood(self) -> torch.Tensor:  # Scalar
        """
        This method computes the marginal log-likelihood of the kernel hyperparameters given the
        training dataset (X, y).
        Assume that N = len(X) in this method.

        Mathematically, the closed form is given as:
            -0.5 * log((2*pi)**N * det(C)) - 0.5 * y.T @ inv(C) @ y
            = -0.5 * log(det(C)) - 0.5 * y.T @ inv(C) @ y + const,
        where C = cov_Y_Y = cov_fX_fX + noise_var * I and inv(...) is the inverse operator.

        We exploit the full advantages of the Cholesky decomposition (C = L @ L.T) in this method:
            1. The determinant of a lower triangular matrix is the diagonal product, which can be
               computed with N flops where log(det(C)) = log(det(L.T @ L)) = 2 * log(det(L)).
            2. Solving linear system L @ u = y, which yields u = inv(L) @ y, costs N**2 flops.
        Note that given `u = inv(L) @ y` and `inv(C) = inv(L @ L.T) = inv(L).T @ inv(L)`,
        y.T @ inv(C) @ y is calculated as (inv(L) @ y) @ (inv(L) @ y).

        In principle, we could invert the matrix C first, but in this case, it costs:
            1. 1/3*N**3 flops for the determinant of inv(C).
            2. 2*N**2-N flops to solve C @ alpha = y, which is alpha = inv(C) @ y.

        Since the Cholesky decomposition costs 1/3*N**3 flops and the matrix inversion costs
        2/3*N**3 flops, the overall cost for the former is 1/3*N**3+N**2+N flops and that for the
        latter is N**3+2*N**2-N flops.
        """
        n_points = self._X_train.shape[0]
        const = -0.5 * n_points * math.log(2 * math.pi)
        cov_Y_Y = self.kernel() + self.noise_var * torch.eye(n_points, dtype=torch.float64)
        L = torch.linalg.cholesky(cov_Y_Y)
        logdet_part = -L.diagonal().log().sum()
        inv_L_y = torch.linalg.solve_triangular(L, self._y_train[:, None], upper=False)[:, 0]
        quad_part = -0.5 * (inv_L_y @ inv_L_y)
        return logdet_part + const + quad_part

    def _fit_kernel_params(
        self,
        log_prior: Callable[[GPRegressor], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> GPRegressor:
        n_params = self._X_train.shape[1]

        # We apply log transform to enforce the positivity of the kernel parameters.
        # Note that we cannot just use the constraint because of the numerical unstability
        # of the marginal log likelihood.
        # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
        # pathological behavior of maximum likelihood estimation.
        initial_raw_params = np.concatenate(
            [
                np.log(self.inverse_squared_lengthscales.detach().numpy()),
                [
                    np.log(self.kernel_scale.item()),
                    # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                    np.log(self.noise_var.item() - 0.99 * minimum_noise),
                ],
            ]
        )

        def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                self.inverse_squared_lengthscales = torch.exp(raw_params_tensor[:n_params])
                self.kernel_scale = torch.exp(raw_params_tensor[n_params])
                self.noise_var = (
                    torch.tensor(minimum_noise, dtype=torch.float64)
                    if deterministic_objective
                    else torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise
                )
                loss = -self.marginal_log_likelihood() - log_prior(self)
                loss.backward()  # type: ignore
                # scipy.minimize requires all the gradients to be zero for termination.
                raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
                assert not deterministic_objective or raw_noise_var_grad == 0
            return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

        with single_blas_thread_if_scipy_v1_15_or_newer():
            # jac=True means loss_func returns the gradient for gradient descent.
            res = scipy.optimize.minimize(
                # Too small `gtol` causes instability in loss_func optimization.
                loss_func,
                initial_raw_params,
                jac=True,
                method="l-bfgs-b",
                options={"gtol": gtol},
            )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        raw_params_opt_tensor = torch.from_numpy(res.x)
        self.inverse_squared_lengthscales = torch.exp(raw_params_opt_tensor[:n_params])
        self.kernel_scale = torch.exp(raw_params_opt_tensor[n_params])
        self.noise_var = (
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        )
        self._cache_matrix()
        return self


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    gpr_cache: GPRegressor | None = None,
    gtol: float = 1e-2,
) -> GPRegressor:
    default_kernel_params = torch.ones(X.shape[1] + 2, dtype=torch.float64)
    # TODO: Move this function into a method of `GPRegressor`

    def _default_gpr() -> GPRegressor:
        return GPRegressor(
            is_categorical=torch.from_numpy(is_categorical),
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            inverse_squared_lengthscales=default_kernel_params[:-2].clone(),
            kernel_scale=default_kernel_params[-2].clone(),
            noise_var=default_kernel_params[-1].clone(),
        )

    default_gpr_cache = _default_gpr()
    if gpr_cache is None:
        gpr_cache = _default_gpr()

    error = None
    # First try optimizing the kernel params with the provided kernel parameters in gpr_cache,
    # but if it fails, rerun the optimization with the default kernel parameters above.
    # This increases the robustness of the optimization.
    for gpr_cache_to_use in [gpr_cache, default_gpr_cache]:
        try:
            return GPRegressor(
                is_categorical=torch.from_numpy(is_categorical),
                X_train=torch.from_numpy(X),
                y_train=torch.from_numpy(Y),
                inverse_squared_lengthscales=gpr_cache_to_use.inverse_squared_lengthscales,
                kernel_scale=gpr_cache_to_use.kernel_scale,
                noise_var=gpr_cache_to_use.noise_var,
            )._fit_kernel_params(
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel parameters failed: \n{error}\n"
        "The default initial kernel parameters will be used instead."
    )
    default_gpr = _default_gpr()
    default_gpr._cache_matrix()
    return default_gpr
