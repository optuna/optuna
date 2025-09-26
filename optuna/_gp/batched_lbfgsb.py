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
    func_and_grad: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    args_tuple: tuple[Any] | None,
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
            args=tuple(arg[i] for arg in args_tuple) if args_tuple is not None else (),
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

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_and_argsT_pairs = [gl.switch(i) for i, gl in enumerate(greenlets)]

    while x_and_argsT_pairs:
        x_batched_list = [pair[0] for pair in x_and_argsT_pairs if pair is not None]
        args_tuple_transposed = [pair[1] for pair in x_and_argsT_pairs if pair is not None]
        x_batched = np.array(x_batched_list)

        if args_tuple_transposed and all(a is not None for a in args_tuple_transposed):
            current_args_tuple = tuple(zip(*args_tuple_transposed))
            fvals, grads = func_and_grad(x_batched, *current_args_tuple)
        else:
            fvals, grads = func_and_grad(x_batched)

        results = [gl.switch((fvals[i], grads[i])) for i, gl in enumerate(greenlets)]

        live_pairs_and_greenlets = [
            (pair, gl) for pair, gl in zip(results, greenlets) if pair is not None
        ]

        if not live_pairs_and_greenlets:
            break

        x_and_argsT_pairs, greenlets = map(list, zip(*live_pairs_and_greenlets))

    return xs_opt, fvals_opt, n_iterations


def batched_lbfgsb(
    func_and_grad: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    args_tuple: tuple[Any] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args_tuple is not None and x0_batched.ndim > 2:
        raise ValueError("x0_batched must be 2D when args_tuple is provided.")
    x0_batched = x0_batched.reshape(-1, x0_batched.shape[-1])  # Make 3+D array 2D.
    if _greenlet_imports.is_successful() and len(x0_batched) > 1:
        # NOTE(Kaichi-Irie): when batch size is 1, using greenlet causes context-switch overhead.
        xs_opt, fvals_opt, n_iterations = _batched_lbfgsb(
            func_and_grad=func_and_grad,
            x0_batched=x0_batched,
            args_tuple=args_tuple,
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
        # args: (P,dim) -> (P,1,dim)
        def _func_and_grad_wrapper(x: np.ndarray, *args: Any) -> tuple[float, np.ndarray]:
            if args:
                args_ = ([arg] for arg in args)  # (P,1,dim)
                fval, grad = func_and_grad(x, *args_)
            else:
                fval, grad = func_and_grad(x)
            return fval.item(), grad

        xs_opt = np.empty_like(x0_batched)
        fvals_opt = np.empty(x0_batched.shape[0], dtype=float)
        n_iterations = np.empty(x0_batched.shape[0], dtype=int)
        for i, x0 in enumerate(x0_batched):
            xs_opt[i], fvals_opt[i], info = so.fmin_l_bfgs_b(
                func=_func_and_grad_wrapper,
                x0=x0,
                args=tuple(arg[i] for arg in args_tuple) if args_tuple is not None else (),
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
