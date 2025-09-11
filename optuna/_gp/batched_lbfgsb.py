from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._imports import try_import
from optuna.logging import get_logger


with try_import() as _greenlet_imports:
    from greenlet import greenlet

if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy.optimize as so

else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")


if not _greenlet_imports.is_successful():
    _logger = get_logger(__name__)
    _logger.warning(
        "The 'greenlet' package is unavailable, falling back to sequential L-BFGS-B optimization. "
        "This may lead to slower suggestions."
    )


def _batched_lbfgsb(
    func_and_grad: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    bounds: list[tuple[float, float]] | None,
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
    xs_opt = np.empty_like(x0_batched)
    fvals_opt = np.empty(batch_size, dtype=float)
    n_iterations = np.empty(batch_size, dtype=int)
    is_remaining_batch = np.ones(batch_size, dtype=bool)

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
        )
        xs_opt[i] = x_opt
        fvals_opt[i] = fval_opt
        n_iterations[i] = info["nit"]
        is_remaining_batch[i] = False

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_batched = [gl.switch(i) for i, gl in enumerate(greenlets)]

    while np.any(is_remaining_batch):
        remaining_batch_indices = np.where(is_remaining_batch)[0]
        fvals, grads = func_and_grad(np.asarray(x_batched), np.asarray(remaining_batch_indices))

        x_batched = []
        next_greenlets = []
        for i, gl in enumerate(greenlets):
            x = gl.switch((fvals[i], grads[i]))
            if x is None:
                continue
            x_batched.append(x)
            next_greenlets.append(gl)
        greenlets = next_greenlets
    return xs_opt, fvals_opt, n_iterations


def batched_lbfgsb(
    func_and_grad: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if _greenlet_imports.is_successful() and len(x0_batched) > 1:
        # NOTE(Kaichi-Irie): when batch size is 1, using greenlet causes context-switch overhead.
        xs_opt, fvals_opt, n_iterations = _batched_lbfgsb(
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

    # fall back to sequential optimization if greenlet is not available.
    else:
        xs_opt = np.empty_like(x0_batched)
        fvals_opt = np.empty(x0_batched.shape[0], dtype=float)
        n_iterations = np.empty(x0_batched.shape[0], dtype=int)
        for i, x0 in enumerate(x0_batched):
            batch_indices = np.array([i])
            xs_opt[i], fvals_opt[i], info = so.fmin_l_bfgs_b(
                func=func_and_grad,  # type: ignore[arg-type]
                args=(batch_indices,),  # type: ignore[arg-type]
                x0=x0,
                bounds=bounds,
                m=m,
                factr=factr,
                pgtol=pgtol,
                maxfun=max_evals,
                maxiter=max_iters,
                maxls=max_line_search,
            )
            n_iterations[i] = info["nit"]

    return xs_opt, fvals_opt, n_iterations
