from __future__ import annotations

from collections.abc import Callable
import importlib
import sys
from typing import Any

import numpy as np
import pytest
from scipy.optimize import fmin_l_bfgs_b

from optuna._gp.batched_lbfgsb import batched_lbfgsb


RADIUS = 5.12


def rastrigin_and_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 1:
        x = x[None]
    A = 10.0
    dim = x.shape[-1]
    _2pi_x = 2 * np.pi * x

    fval = A * dim + np.sum(x**2 - A * np.cos(_2pi_x), axis=-1)
    grad = 2 * x + 2 * np.pi * A * np.sin(_2pi_x)
    return fval, grad


def styblinski_tang_and_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Styblinski-Tang function, which has multiple local minima.
    if x.ndim == 1:
        x = x[None]
    fval = np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1) / 2
    grad = 2 * x**3 - 16 * x + 2.5
    return fval, grad


def X0_and_bounds(
    dim: int, n_localopts: int, lower_bound: float, upper_bound: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    X0 = rng.random((n_localopts, dim)) * 2 * RADIUS - RADIUS
    bounds = np.array([[lower_bound, upper_bound]] * dim)
    return X0, bounds


def _verify_results(
    X0: np.ndarray,
    func_and_grad: Callable,
    kwargs_ours: Any,
    kwargs_scipy: Any,
    batched_lbfgsb_func: Callable,
) -> None:
    xs_opt1, fvals_opt1, n_iters1 = batched_lbfgsb_func(
        func_and_grad=func_and_grad, x0_batched=X0, **kwargs_ours
    )
    xs_opt2 = []
    fvals_opt2 = []
    n_iters2 = []
    for x0 in X0:
        x_opt, fval, info = fmin_l_bfgs_b(func=func_and_grad, x0=x0, **kwargs_scipy)
        xs_opt2.append(x_opt)
        fvals_opt2.append(float(fval))
        n_iters2.append(info["nit"])

    assert np.all(n_iters1 == np.array(n_iters2))
    assert np.all(fvals_opt1 == np.array(fvals_opt2))
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
@pytest.mark.parametrize("use_greenlet", [True, False])
@pytest.mark.parametrize(
    "lower_bound,upper_bound",
    [(-RADIUS, RADIUS), (-np.inf, RADIUS), (-RADIUS, np.inf), (-np.inf, np.inf)],
)
@pytest.mark.parametrize("dim, n_localopts", [(10, 10), (1, 10), (10, 1), (1, 1)])
def test_batched_lbfgsb(
    monkeypatch: pytest.MonkeyPatch,
    func_and_grad: Callable,
    kwargs_ours: Any,
    kwargs_scipy: Any,
    use_greenlet: bool,
    lower_bound: float,
    upper_bound: float,
    dim: int,
    n_localopts: int,
) -> None:
    if not use_greenlet:
        monkeypatch.setitem(sys.modules, "greenlet", None)

    import optuna._gp.batched_lbfgsb as optimization_module

    importlib.reload(optimization_module)
    assert optimization_module._greenlet_imports.is_successful() == use_greenlet

    X0, bounds = X0_and_bounds(
        dim=dim, n_localopts=n_localopts, lower_bound=lower_bound, upper_bound=upper_bound
    )
    kwargs_ours.update(bounds=bounds)
    kwargs_scipy.update(bounds=bounds)
    _verify_results(
        X0,
        func_and_grad,
        kwargs_ours,
        kwargs_scipy,
        batched_lbfgsb_func=optimization_module.batched_lbfgsb,
    )


def test_batched_lbfgsb_invalid_input() -> None:
    batch_size = 3
    dimension = 2
    x0_batched = np.random.rand(batch_size, dimension)

    # x0_batched validation
    with pytest.raises(ValueError):
        batched_lbfgsb(
            func_and_grad=lambda x: (np.sum(x, axis=1), np.ones_like(x)),
            x0_batched=x0_batched[0],  # not 2D
        )

    # batched_args validation
    def dummy_func_and_grad(x: np.ndarray, _arg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(x, axis=1), np.ones_like(x)

    with pytest.raises(AssertionError):
        batched_lbfgsb(
            func_and_grad=dummy_func_and_grad,
            x0_batched=x0_batched,
            batched_args=([0] * (batch_size + 1),),  # wrong length
        )

    # bounds validation
    invalid_bounds = [(0.0, 1.0)]  # length is not equal to dimension
    with pytest.raises(AssertionError):
        batched_lbfgsb(
            func_and_grad=dummy_func_and_grad,
            x0_batched=x0_batched,
            batched_args=(list(range(batch_size)),),
            bounds=invalid_bounds,
        )
