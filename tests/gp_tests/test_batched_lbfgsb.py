from typing import Any
from typing import Callable

import numpy as np
import pytest
from scipy.optimize import fmin_l_bfgs_b

from optuna._gp.batched_lbfgsb import batched_lbfgsb


# (B,D) -> (B,) or (D,) -> ()
def rastrigin_and_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = 10.0
    dim = x.shape[-1]
    _2pi_x = 2 * np.pi * x

    fval = A * dim + np.sum(x**2 - A * np.cos(_2pi_x), axis=-1)
    grad = 2 * x + 2 * np.pi * A * np.sin(_2pi_x)
    return fval, grad


def styblinski_tang_and_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Styblinski-Tang function, which has multiple local minima.
    # Global minimum is at f(-2.903534), where f(-2.903534) = -39.16599 * dim
    fval = np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1) / 2
    grad = 2 * x**3 - 16 * x + 2.5
    return fval, grad


def X0_and_bounds(dim: int, n_localopts: int) -> tuple[np.ndarray, np.ndarray]:
    R = 5.12
    rng = np.random.RandomState(0)
    X0 = rng.random((n_localopts, dim)) * 2 * R - R
    bounds = np.array([[-R, R]] * dim)
    return X0, bounds


def _verify_results(
    X0: np.ndarray, func_and_grad: Callable, kwargs_ours: Any, kwargs_scipy: Any
) -> None:
    xs_opt1, fvals_opt1, n_iters1 = batched_lbfgsb(
        func_and_grad=func_and_grad, x0_batched=X0, **kwargs_ours
    )
    xs_opt2 = []
    fvals_opt2 = []
    n_iters2 = []
    for x0 in X0:
        x_opt, fval, info = fmin_l_bfgs_b(func_and_grad, x0=x0, **kwargs_scipy)
        xs_opt2.append(x_opt)
        fvals_opt2.append(fval.item())
        n_iters2.append(info["nit"])

    assert np.allclose(xs_opt1, np.array(xs_opt2))
    assert np.allclose(fvals_opt1, np.array(fvals_opt2))
    assert np.all(xs_opt1 == np.array(xs_opt2))


test_params = [
    (rastrigin_and_grad, {}, {}),
    (styblinski_tang_and_grad, {}, {}),
    (rastrigin_and_grad, {"max_evals": 3}, {"maxfun": 3}),
    (styblinski_tang_and_grad, {"max_evals": 3}, {"maxfun": 3}),
    (rastrigin_and_grad, {"max_iters": 3}, {"maxiter": 3}),
    (styblinski_tang_and_grad, {"max_iters": 3}, {"maxiter": 3}),
]


@pytest.mark.parametrize("func_and_grad,kwargs_ours,kwargs_scipy", test_params)
def test_batched_lbfgsb(func_and_grad: Callable, kwargs_ours: Any, kwargs_scipy: Any) -> None:
    dim = 10
    n_localopts = 10
    X0, bounds = X0_and_bounds(dim=dim, n_localopts=n_localopts)
    kwargs_ours.update(bounds=bounds)
    kwargs_scipy.update(bounds=bounds)
    _verify_results(X0, func_and_grad, kwargs_ours, kwargs_scipy)


@pytest.mark.parametrize("func_and_grad,kwargs_ours,kwargs_scipy", test_params)
@pytest.mark.parametrize("lower_bound,upper_bound", [(-np.inf, None), (None, np.inf), (-np.inf, np.inf), (None, None)])
def test_batched_lbfgsb_without_bounds(
    func_and_grad: Callable,
    kwargs_ours: Any,
    kwargs_scipy: Any,
    lower_bound: float | None,
    upper_bound: float | None,
) -> None:
    dim = 10
    n_localopts = 10
    X0, bounds = X0_and_bounds(dim=dim, n_localopts=n_localopts)
    if lower_bound is not None:
        bounds[:, 0] = lower_bound
    if upper_bound is not None:
        bounds[:, 1] = upper_bound
    kwargs_ours.update(bounds=bounds)
    kwargs_scipy.update(bounds=bounds)
    _verify_results(X0, func_and_grad, kwargs_ours, kwargs_scipy)
