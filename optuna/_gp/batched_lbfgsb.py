from __future__ import annotations

from typing import Any
from typing import List
from typing import TYPE_CHECKING

import numpy as np

from optuna._imports import try_import


with try_import() as _imports:
    from greenlet import greenlet

if TYPE_CHECKING:
    from typing import Protocol

    import scipy.optimize as so

    class FuncAndGrad(Protocol):
        def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raise NotImplementedError

else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")


def _batched_lbfgsb(
    func_and_grad: FuncAndGrad,
    x0_batched: np.ndarray,
    bounds: list | None,
    m: int,
    factr: float,
    pgtol: float,
    max_evals: int,
    max_iters: int,
    max_line_search: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert (
        x0_batched.ndim == 2
    ), f"The shape of x0 must be (batch_size, dim), but got {x0_batched.shape}."
    batch_size = len(x0_batched)
    x_opts: List[Any] = [None] * batch_size
    fval_opts: List[Any] = [None] * batch_size
    n_iterations: List[Any] = [None] * batch_size

    def run(i: int) -> None:
        def _func_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            fval, grad = greenlet.getcurrent().parent.switch(x)
            # NOTE(nabenabe): copy is necessary to convert grad to writable.
            return float(fval), grad.copy()

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=_func_and_grad,
            x0=x0_batched[i],
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            maxfun=max_evals,
            maxiter=max_iters,
            maxls=max_line_search,
            iprint=-1,
        )
        x_opts[i] = x_opt
        fval_opts[i] = fval_opt
        n_iterations[i] = info["nit"]

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_batched = [gl.switch(i) for i, gl in enumerate(greenlets)]

    while len(x_batched := [x for x in x_batched if x is not None]) > 0:
        fvals, grads = func_and_grad(np.asarray(x_batched))
        x_batched = [gl.switch((fvals[i], grads[i])) for i, gl in enumerate(greenlets)]
        greenlets = [gl for x, gl in zip(x_batched, greenlets) if x is not None]

    return np.array(x_opts), np.array(fval_opts), np.array(n_iterations)


def batched_lbfgsb(
    func_and_grad: FuncAndGrad,
    x0_batched: np.ndarray,
    bounds: list | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def func_and_grad_1D_wrapper(scaled_x: np.ndarray) -> tuple[float, np.ndarray]:
        """A wrapper for `func_and_grad` to handle 1D inputs.

        This is used as a fallback to sequential optimization when the batched
        L-BFGS-B is not available. It adapts the batched `func_and_grad` for
        use with optimizers like `scipy.optimize.fmin_l_bfgs_b` that expect
        a 1D input array.
        """
        assert scaled_x.ndim == 1
        fval, grad = func_and_grad(scaled_x[None])
        return fval.item(), grad.ravel()

    if _imports.is_successful():
        scaled_cont_x_opts, neg_fval_opts, n_iterations = _batched_lbfgsb(
            func_and_grad=func_and_grad,
            x0_batched=x0_batched,
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            max_evals=max_evals,
            max_iters=max_iters,
            max_line_search=max_line_search,
        )

    # fallback to sequential optimization if SciPy version is not supported
    else:
        scaled_cont_x_opts = np.zeros_like(x0_batched)
        neg_fval_opts = np.zeros(x0_batched.shape[0])
        n_iterations = np.zeros(x0_batched.shape[0], dtype=int)
        for batch_index, x0 in enumerate(x0_batched):
            scaled_cont_x_opt, neg_fval_opt, info = so.fmin_l_bfgs_b(
                func=func_and_grad_1D_wrapper,
                x0=x0,
                bounds=bounds,
                m=m,
                factr=factr,
                pgtol=pgtol,
                maxfun=max_evals,
                maxiter=max_iters,
                maxls=max_line_search,
            )
            scaled_cont_x_opts[batch_index] = scaled_cont_x_opt
            neg_fval_opts[batch_index] = neg_fval_opt
            n_iterations[batch_index] = info["nit"]

    return scaled_cont_x_opts, neg_fval_opts, n_iterations
