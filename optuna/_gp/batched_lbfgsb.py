from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._imports import try_import
from optuna.logging import get_logger


with try_import() as _greenlet_imports:
    from greenlet import greenlet

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

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
    func_and_grad: Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    args_list: list[Any] | None,
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

    def run(i: int) -> None:
        def _func_and_grad(x: np.ndarray, *args: Any) -> tuple[float, np.ndarray]:
            fval, grad = greenlet.getcurrent().parent.switch(x, args if len(args) else None)
            # NOTE(nabenabe): copy is necessary to convert grad to writable.
            return float(fval), grad.copy()

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=_func_and_grad,
            x0=x0_batched[i],
            args=args_list[i] if args_list is not None else (),
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
        greenlet.getcurrent().parent.switch(None, None)

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_and_args = [gl.switch(i) for i, gl in enumerate(greenlets)]
    args_batched = [args for _, args in x_and_args] if args_list is not None else None
    while (x_batched := np.array([x for x, _ in x_and_args if x is not None])).size:
        fvals, grads = func_and_grad(x_batched, *() if args_batched is None else (args_batched,))
        x_and_args = [gl.switch((fvals[i], grads[i])) for i, gl in enumerate(greenlets)]
        args_batched = (
            None if args_batched is None else [args for x, args in x_and_args if x is not None]
        )
        greenlets = [gl for (x, _), gl in zip(x_and_args, greenlets) if x is not None]

    return xs_opt, fvals_opt, n_iterations


def batched_lbfgsb(
    func_and_grad: Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    args_list: list[Any] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args_list is not None and x0_batched.ndim > 2:
        raise ValueError("x0_batched must be 2D when args_list is provided.")
    if args_list is not None and len(args_list) != len(x0_batched):
        raise ValueError(
            f"The length of args_list must be equal to the batch size, "
            f"but got len(args_list)={len(args_list)} and batch size={len(x0_batched)}."
        )
    x0_batched = x0_batched.reshape(-1, x0_batched.shape[-1])  # Make 3+D array 2D.
    if _greenlet_imports.is_successful() and len(x0_batched) > 1:
        # NOTE(Kaichi-Irie): when batch size is 1, using greenlet causes context-switch overhead.
        xs_opt, fvals_opt, n_iterations = _batched_lbfgsb(
            func_and_grad=func_and_grad,
            x0_batched=x0_batched,
            args_list=args_list,
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

        def _func_and_grad_wrapper(x: np.ndarray, *args: Any) -> tuple[float, np.ndarray]:
            fval, grad = func_and_grad(x, *([args],) if len(args) else ())
            return fval.item(), grad

        xs_opt = np.empty_like(x0_batched)
        fvals_opt = np.empty(x0_batched.shape[0], dtype=float)
        n_iterations = np.empty(x0_batched.shape[0], dtype=int)
        for i, x0 in enumerate(x0_batched):
            xs_opt[i], fvals_opt[i], info = so.fmin_l_bfgs_b(
                func=_func_and_grad_wrapper,
                x0=x0,
                args=(args_list[i],) if args_list is not None else (),  # type: ignore[arg-type]
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
